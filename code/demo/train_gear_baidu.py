"""百度网盘齿轮数据集 (多齿轮类型 × 5故障状态) 的快速 strict ZSL 评估

数据结构：
  GearDatasets/
    人字齿轮/ → 健康/, 剥落/, 断齿/, 磨损/, 裂纹/  (*.csv)
    斜齿轮/  → 健康/, 剥落/, 断齿/, 磨损/, 裂纹/  (*.csv)
    直齿轮/  → 健康/, 剥落/, 断齿/, 磨损/, 裂纹/  (*.csv)

使用方式（快速 smoke 测试）：
  python demo/train_gear_baidu.py \
    --data_path "D:\\BaiduNetdiskDownload\\GearDatasets" \
    --epochs 30 --max-files-per-class 10 --max-segments-per-file 10 \
    --unseen-classes 剥落,断齿 --seen-classes 健康,磨损,裂纹 \
    --zsl-only --skip-mlp --unseen-calib-ratio 0.2 --calib-metric f1 \
    --results-json tmp_gear_quick.json --seed 0
"""

from __future__ import annotations

import os
import sys
import argparse
import json
import time
import tempfile
import numpy as np
import torch
from typing import List, Dict, Any
from pathlib import Path
from torch.utils.data import DataLoader, Subset
import csv
import re
from collections import deque
import itertools

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from esag.utils import load_config, set_seed, get_device, count_parameters, load_checkpoint
from esag.preprocessing import SignalPreprocessor
from esag.datasets import ZeroShotSplitter, create_dataloaders, BearingDataset
from esag.model import ESAG


# BaiduGear 故障类型 → 齿轮专用属性映射 (使用 BAIDU_GEAR_FAULT_ATTRIBUTES)
# 属性定义在 esag/semantic_attributes.py 中
BAIDU_GEAR_CLASS_TO_ATTR = {
    "健康": 0,   # 健康
    "剥落": 1,   # 剥落
    "断齿": 2,   # 断齿
    "磨损": 3,   # 磨损
    "裂纹": 4,   # 裂纹
}


class BaiduGearDataset:
    """百度网盘齿轮数据集加载器 (聚合多个齿轮类型目录)"""

    CLASS_TO_ATTR = BAIDU_GEAR_CLASS_TO_ATTR

    GEAR_TYPES = ["人字齿轮", "斜齿轮", "直齿轮"]
    
    _FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
    
    def __init__(
        self,
        data_path: str,
        signal_length: int = 2048,
        overlap: float = 0.5,
        max_files_per_class: int = 50,
        max_segments_per_file: int = 50,
        vibration_col: int = 0,
        sample_rate_hint: int | None = None,
    ):
        self.data_path = data_path
        self.signal_length = signal_length
        self.overlap = overlap
        self.max_files_per_class = max_files_per_class
        self.max_segments_per_file = max_segments_per_file
        self.vibration_col = int(vibration_col)
        if self.vibration_col < 0:
            raise ValueError("vibration_col must be >= 0")

        self.sample_rate_hint = int(sample_rate_hint) if sample_rate_hint is not None else None
        self.detected_sample_rate: int | None = None
    
    @staticmethod
    def _parse_row_floats(line: str) -> List[float] | None:
        nums = BaiduGearDataset._FLOAT_RE.findall(line)
        try:
            if not nums:
                return None
            return [float(x) for x in nums]
        except Exception:
            return None

    @staticmethod
    def _parse_header_sample_rate(line: str) -> int | None:
        """Parse sample rate from the first header line.

        Observed format example:
          15360,460800,10,0,3
        We treat the first field as sample_rate (Hz) if it's a plausible int.
        """
        try:
            parts = [p.strip() for p in str(line).split(',') if str(p).strip()]
            if not parts:
                return None
            sr = int(float(parts[0]))
            if 1000 <= sr <= 500000:
                return sr
        except Exception:
            return None
        return None
    
    def _stream_segments_from_csv(self, file_path: str) -> tuple[np.ndarray, int | None]:
        step = int(self.signal_length * (1 - self.overlap))
        if step <= 0:
            step = self.signal_length
        
        segments: List[np.ndarray] = []
        buf = deque()
        
        detected_sr: int | None = None
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                line_num = 0
                skip_lines = 4  # Skip first 4 header lines
                for line in f:
                    line = line.strip()
                    if line_num == 0:
                        detected_sr = self._parse_header_sample_rate(line)

                    if line_num < skip_lines:
                        line_num += 1
                        continue

                    row = self._parse_row_floats(line)
                    if row is None:
                        line_num += 1
                        continue

                    col = int(self.vibration_col)
                    if col >= len(row):
                        line_num += 1
                        continue

                    buf.append(float(row[col]))
                    if len(buf) < self.signal_length:
                        line_num += 1
                        continue
                    
                    seg = np.fromiter(
                        itertools.islice(buf, 0, self.signal_length),
                        dtype=np.float32,
                        count=self.signal_length
                    )
                    if seg.shape[0] == self.signal_length:
                        segments.append(seg)
                    
                    for _ in range(step):
                        if buf:
                            buf.popleft()
                    
                    if self.max_segments_per_file and len(segments) >= self.max_segments_per_file:
                        break
                    
                    line_num += 1
        except Exception as e:
            print(f"Warning: Failed to read {file_path}: {e}")
            return np.array([]), detected_sr
        
        return (np.array(segments) if segments else np.array([])), detected_sr
    
    def load_data(self, fault_types: List[str] = None):
        """加载所有齿轮类型的 CSV 文件"""
        if fault_types is None:
            fault_types = ["健康", "剥落", "断齿", "磨损", "裂纹"]
        
        all_signals = []
        all_labels = []
        all_class_indices = []
        class_names = fault_types
        
        file_count = {ft: 0 for ft in fault_types}
        
        sample_rates: List[int] = []

        for label_idx, fault_type in enumerate(fault_types):
            attr_idx = self.CLASS_TO_ATTR.get(fault_type, label_idx)
            
            # 从所有齿轮类型目录中搜索该故障状态
            for gear_type in self.GEAR_TYPES:
                fault_dir = os.path.join(self.data_path, gear_type, fault_type)
                if not os.path.isdir(fault_dir):
                    continue
                
                csv_files = [f for f in os.listdir(fault_dir) if f.endswith(".csv")]
                
                for csv_file in csv_files:
                    if self.max_files_per_class and file_count[fault_type] >= self.max_files_per_class:
                        break
                    
                    file_path = os.path.join(fault_dir, csv_file)
                    try:
                        segments, sr = self._stream_segments_from_csv(file_path)

                        if sr is not None:
                            sample_rates.append(int(sr))

                        if segments.shape[0] > 0:
                            all_signals.append(segments)
                            all_labels.extend([label_idx] * len(segments))
                            all_class_indices.extend([attr_idx] * len(segments))
                            file_count[fault_type] += 1
                    except Exception:
                        pass
        
        if not all_signals:
            raise FileNotFoundError(f"No CSV files found in {self.data_path}")
        
        signals = np.concatenate(all_signals, axis=0)
        labels = np.array(all_labels)
        class_indices = np.array(all_class_indices)

        # Resolve sample rate: explicit hint > detected median > None
        if self.sample_rate_hint is not None:
            self.detected_sample_rate = int(self.sample_rate_hint)
        elif sample_rates:
            self.detected_sample_rate = int(np.median(np.asarray(sample_rates, dtype=np.int64)))
        else:
            self.detected_sample_rate = None
        
        return signals, labels, class_indices, class_names


def _extract_features_and_labels(model: ESAG, loader, device: torch.device):
    model.eval()
    all_features = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            feats = model.extract_features(images)
            all_features.append(feats.detach().cpu())
            all_labels.append(labels.detach().cpu())
    if not all_features:
        return None, None
    return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0)


def _jsonify(obj):
    """递归将 numpy/torch 对象转换为可 JSON 序列化的 Python 类型"""
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    return obj


def train_gear(
    config_path: str = None,
    data_path: str = None,
    num_epochs: int = 30,
    max_files_per_class: int = 10,
    max_segments_per_file: int = 10,
    results_json: str = None,
    zsl_support: str = "all_seen",
    num_generated: int = None,
    seed: int = None,
    seen_classes_csv: str = None,
    unseen_classes_csv: str = None,
    eval_only: bool = False,
    checkpoint_path: str = None,
    save_dir: str = None,
    unseen_calib_ratio: float = 0.0,
    calib_seed: int = None,
    num_generated_grid: str = None,
    skip_mlp: bool = False,
    zsl_only: bool = False,
    calib_metric: str = "accuracy",
    disable_gen_filter: bool = False,
    disable_fusion_alpha_tune: bool = False,
    calib_allow_sem_knn_mix_select: bool = False,
    calib_reject_degenerate: bool = False,
    calib_prefer_sem_knn_mix: bool = False,
    calib_prefer_mix_delta: float = 0.05,
    calib_robust_semknn: bool = False,
    calib_robust_repeats: int = 5,
    calib_robust_val_ratio: float = 0.2,
    calib_robust_seed: int | None = None,
    eval_noise_std: float = 0.0,
    eval_noise_snr_db: float | None = None,
    eval_noise_seed: int | None = 0,
    sample_rate: int | None = None,
    vibration_col: int = 0,
    num_workers: int | None = None,
):
    print("=" * 70)
    print("  ESAG: 百度网盘齿轮数据集 (Baidu Gear Dataset) Training")
    print("=" * 70)
    
    if config_path is None:
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                   'configs', 'default.yaml')
    config = load_config(config_path)
    
    if num_epochs is not None:
        config.setdefault('training', {})
        config['training']['num_epochs'] = int(num_epochs)
    
    if save_dir is not None:
        config.setdefault('runtime', {})
        config['runtime']['save_dir'] = str(save_dir)
    
    if seed is not None:
        config.setdefault('runtime', {})
        config['runtime']['seed'] = int(seed)
    
    set_seed(config['runtime']['seed'])
    device = get_device(config['runtime']['device'])
    
    if data_path is None:
        data_path = "./data/gear"
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Gear data not found at: {data_path}")
    
    gear_loader = BaiduGearDataset(
        data_path=data_path,
        signal_length=config['model']['signal_length'],
        max_files_per_class=max_files_per_class,
        max_segments_per_file=max_segments_per_file,
        vibration_col=int(vibration_col),
        sample_rate_hint=int(sample_rate) if sample_rate is not None else None,
    )
    
    signals, labels, class_indices, class_names = gear_loader.load_data()
    
    print(f"\n  Total samples: {len(signals)}")
    print(f"  Classes: {class_names}")
    
    resolved_sr = (
        int(sample_rate)
        if sample_rate is not None
        else (int(gear_loader.detected_sample_rate) if gear_loader.detected_sample_rate is not None else None)
    )
    if resolved_sr is None:
        resolved_sr = int(config['datasets']['pu']['sample_rate'])
        print(f"  Warning: Gear sample_rate not detected; falling back to PU sample_rate={resolved_sr}")
    else:
        print(f"  Gear sample_rate: {int(resolved_sr)}")

    preprocessor = SignalPreprocessor(
        signal_length=config['model']['signal_length'],
        sample_rate=int(resolved_sr),
        output_size=config['model']['swin']['img_size'],
    )
    
    def _parse_name_csv(s: str):
        parts = [p.strip() for p in str(s).split(',') if p.strip()]
        out = []
        for p in parts:
            if p.lower() not in [x.lower() for x in out]:
                out.append(p)
        return out
    
    seen_classes = None
    unseen_classes = None
    
    if unseen_classes_csv and str(unseen_classes_csv).strip():
        unseen_names = _parse_name_csv(unseen_classes_csv)
        name_to_idx = {str(n).lower(): int(i) for i, n in enumerate(class_names)}
        missing = [n for n in unseen_names if str(n).lower() not in name_to_idx]
        if missing:
            raise ValueError(f"Unknown unseen class: {missing}. Available: {class_names}")
        unseen_classes = [name_to_idx[str(n).lower()] for n in unseen_names]
    
    if seen_classes_csv and str(seen_classes_csv).strip():
        seen_names = _parse_name_csv(seen_classes_csv)
        name_to_idx = {str(n).lower(): int(i) for i, n in enumerate(class_names)}
        missing = [n for n in seen_names if str(n).lower() not in name_to_idx]
        if missing:
            raise ValueError(f"Unknown seen class: {missing}. Available: {class_names}")
        seen_classes = [name_to_idx[str(n).lower()] for n in seen_names]
    
    if (seen_classes is not None) and (unseen_classes is not None):
        if set(seen_classes) & set(unseen_classes):
            raise ValueError("Seen and unseen must be disjoint")
        if not seen_classes or not unseen_classes:
            raise ValueError("Both seen and unseen must be non-empty")
        split_info = ZeroShotSplitter.split(labels, class_indices, seen_classes, unseen_classes)
    else:
        # Default: use first class as seen, rest as unseen
        seen_classes = [0]
        unseen_classes = list(range(1, len(class_names)))
        split_info = ZeroShotSplitter.split(labels, class_indices, seen_classes, unseen_classes)
    
    print(f"  Seen classes: {[class_names[i] for i in split_info['seen_classes']]}")
    print(f"  Unseen classes: {[class_names[i] for i in split_info['unseen_classes']]}")
    
    if bool(eval_only):
        effective_num_workers = 0
    elif num_workers is not None:
        effective_num_workers = int(num_workers)
    elif os.name == 'nt':
        # Windows stability: DataLoader workers can hang/slow down noticeably.
        effective_num_workers = 0
    else:
        effective_num_workers = int(config['runtime']['num_workers'])
    dataloaders = create_dataloaders(
        signals, labels, class_indices, preprocessor, split_info,
        batch_size=config['training']['batch_size'],
        num_workers=effective_num_workers,
        eval_noise_std=float(eval_noise_std) if eval_noise_std is not None else 0.0,
        eval_noise_snr_db=(None if eval_noise_snr_db is None else float(eval_noise_snr_db)),
        eval_noise_seed=(None if eval_noise_seed is None else int(eval_noise_seed)),
    )

    # Optional: split unseen test into calib/eval for leak-free head selection
    calib_unseen_loader = None
    if float(unseen_calib_ratio) > 0.0:
        ds_unseen = dataloaders['test_unseen'].dataset
        if not isinstance(ds_unseen, Subset):
            raise RuntimeError("Expected test_unseen to be a Subset")

        idxs = np.array(list(ds_unseen.indices), dtype=np.int64)
        if idxs.size < 2:
            raise RuntimeError("Not enough unseen samples to build a calibration split")

        seed_value = int(calib_seed) if calib_seed is not None else int(config['runtime'].get('seed', 42))
        rng = np.random.RandomState(int(seed_value))
        rng.shuffle(idxs)
        n_calib = int(round(len(idxs) * float(unseen_calib_ratio)))
        n_calib = max(1, min(n_calib, len(idxs) - 1))
        calib_idxs = idxs[:n_calib]
        eval_idxs = idxs[n_calib:]

        calib_unseen_loader = DataLoader(
            Subset(ds_unseen.dataset, calib_idxs),
            batch_size=dataloaders['test_unseen'].batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )
        dataloaders['test_unseen'] = DataLoader(
            Subset(ds_unseen.dataset, eval_idxs),
            batch_size=dataloaders['test_unseen'].batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )
        print(
            f"  Unseen calib split: ratio={float(unseen_calib_ratio):.3f} "
            f"calib={int(n_calib)} eval={int(len(idxs) - n_calib)} (seed={seed_value})"
        )
    
    config['model']['semantic']['num_attributes'] = 16
    config['model']['semantic']['attribute_set'] = 'baidu_gear'  # 使用齿轮专用16维属性
    model = ESAG(config)
    print(f"  Model parameters: {count_parameters(model):,}")
    
    ckpt_abs = None
    if checkpoint_path:
        ckpt_abs = os.path.abspath(checkpoint_path)
        if not os.path.exists(ckpt_abs):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_abs}")
    
    if eval_only:
        if not ckpt_abs:
            raise ValueError("--eval-only requires --checkpoint")
        print(f"\n  [Eval-only] Loading checkpoint: {ckpt_abs}")
        load_checkpoint(model, optimizer=None, path=ckpt_abs, device=str(device))
    else:
        trainer = ESAGTrainer(model, config, device)
        trainer.train(dataloaders['train'], dataloaders['val'], num_epochs=num_epochs or config['training']['num_epochs'])
    
    evaluator = ESAGEvaluator(model, device)
    
    unseen_attr_indices = [int(class_indices[labels == c][0])
                           for c in split_info['unseen_classes']
                           if (labels == c).sum() > 0]
    
    seen_support = dataloaders.get('seen_support', dataloaders['test_seen'])

    precomputed_seen = None
    precomputed_eval = None
    precomputed_calib = None
    if (num_generated_grid is not None) or (calib_unseen_loader is not None):
        sf, sl, sci = evaluator.extract_all_features(seen_support)
        precomputed_seen = (sf, sl, sci)
        ef, el, eci = evaluator.extract_all_features(dataloaders['test_unseen'])
        precomputed_eval = (ef, el, eci)
        if calib_unseen_loader is not None:
            cf, cl, cci = evaluator.extract_all_features(calib_unseen_loader)
            precomputed_calib = (cf, cl, cci)

    enable_mlp = not bool(skip_mlp)
    enable_gzsl = not bool(zsl_only)
    enable_gen_filter = not bool(disable_gen_filter)
    enable_fusion_alpha_tune = not bool(disable_fusion_alpha_tune)
    allow_sem_knn_mix_select = bool(calib_allow_sem_knn_mix_select)
    reject_degenerate = bool(calib_reject_degenerate)
    prefer_mix = bool(calib_prefer_sem_knn_mix)
    prefer_mix_delta = float(calib_prefer_mix_delta)
    robust_semknn = bool(calib_robust_semknn)
    robust_repeats = int(calib_robust_repeats)
    robust_val_ratio = float(calib_robust_val_ratio)
    robust_seed = int(calib_robust_seed) if calib_robust_seed is not None else None

    if num_generated_grid is not None:
        if not bool(eval_only):
            raise ValueError("--num-generated-grid requires --eval-only")
        if float(unseen_calib_ratio) <= 0.0:
            raise ValueError("--num-generated-grid requires --unseen-calib-ratio > 0")
        if calib_unseen_loader is None:
            raise RuntimeError("Calibration loader missing; check unseen calib split creation")

        cand_list = sorted(set(int(x) for x in str(num_generated_grid).split(",") if str(x).strip()))
        print(
            f"\n[NumGenerated Select | BaiduGear ZSL-only] candidates={cand_list} "
            f"unseen_calib_ratio={float(unseen_calib_ratio):.3f} "
            f"calib_seed={int(calib_seed) if calib_seed is not None else int(config['runtime'].get('seed', 42))}\n"
        )

        cand_scores = []
        best_ng = None
        best_calib = float("-inf")
        best_res = None

        for ng in cand_list:
            res = evaluator.evaluate_zero_shot(
                seen_support,
                dataloaders['test_unseen'],
                unseen_attr_indices,
                num_generated=int(ng),
                class_names=class_names,
                calib_unseen_loader=calib_unseen_loader,
                gen_seed=int(config['runtime'].get('seed', 42)),
                mlp_seed=int(config['runtime'].get('seed', 42)),
                enable_mlp=bool(enable_mlp),
                enable_gzsl=bool(enable_gzsl),
                enable_gen_filter=bool(enable_gen_filter),
                enable_fusion_alpha_tune=bool(enable_fusion_alpha_tune),
                allow_sem_knn_mix_select=bool(allow_sem_knn_mix_select),
                calib_reject_degenerate=bool(reject_degenerate),
                calib_prefer_sem_knn_mix=bool(prefer_mix),
                calib_prefer_mix_delta=float(prefer_mix_delta),
                calib_robust_semknn=bool(robust_semknn),
                calib_robust_repeats=int(robust_repeats),
                calib_robust_val_ratio=float(robust_val_ratio),
                calib_robust_seed=(int(robust_seed) if robust_seed is not None else None),
                precomputed_seen=precomputed_seen,
                precomputed_eval=precomputed_eval,
                precomputed_calib=precomputed_calib,
                calib_select_metric=str(calib_metric),
            )
            calib_acc = float(res.get("zsl_calib_selected_accuracy", float("nan")))
            cand_scores.append({"num_generated": int(ng), "calib_acc": float(calib_acc)})
            print(f"  candidate num_generated={int(ng):>4d}  calib_selected_acc={calib_acc:.4f}")

            if (calib_acc > best_calib) or (calib_acc == best_calib and (best_ng is None or int(ng) < int(best_ng))):
                best_calib = float(calib_acc)
                best_ng = int(ng)
                best_res = res

        print(f"\n  Selected num_generated={int(best_ng)} (calib_selected_acc={float(best_calib):.4f})\n")
        results = best_res if best_res is not None else {}
        results["num_generated_selected"] = int(best_ng)
        results["num_generated_candidates"] = [int(x) for x in cand_list]
        results["num_generated_calib_scores"] = cand_scores
    else:
        eval_num_generated = int(num_generated) if num_generated is not None else int(config['training']['zero_shot']['num_generated'])
        results = evaluator.evaluate_zero_shot(
            seen_support,
            dataloaders['test_unseen'],
            unseen_attr_indices,
            num_generated=eval_num_generated,
            class_names=class_names,
            calib_unseen_loader=calib_unseen_loader,
            gen_seed=int(config['runtime'].get('seed', 42)),
            mlp_seed=int(config['runtime'].get('seed', 42)),
            enable_mlp=bool(enable_mlp),
            enable_gzsl=bool(enable_gzsl),
            enable_gen_filter=bool(enable_gen_filter),
            enable_fusion_alpha_tune=bool(enable_fusion_alpha_tune),
            allow_sem_knn_mix_select=bool(allow_sem_knn_mix_select),
            calib_reject_degenerate=bool(reject_degenerate),
            calib_prefer_sem_knn_mix=bool(prefer_mix),
            calib_prefer_mix_delta=float(prefer_mix_delta),
            calib_robust_semknn=bool(robust_semknn),
            calib_robust_repeats=int(robust_repeats),
            calib_robust_val_ratio=float(robust_val_ratio),
            calib_robust_seed=(int(robust_seed) if robust_seed is not None else None),
            precomputed_seen=precomputed_seen,
            precomputed_eval=precomputed_eval,
            precomputed_calib=precomputed_calib,
            calib_select_metric=str(calib_metric),
        )
    
    if results_json:
        payload = {
            'dataset': 'BaiduGear',
            'results': _jsonify(results),
            'meta': {
                'timestamp_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'seed': int(config['runtime'].get('seed', 42)),
                'num_epochs': int(num_epochs),
                'seen_class_names': [class_names[i] for i in split_info['seen_classes']],
                'unseen_class_names': [class_names[i] for i in split_info['unseen_classes']],
                'unseen_calib_ratio': float(unseen_calib_ratio),
                'calib_seed': int(calib_seed) if calib_seed is not None else int(config['runtime'].get('seed', 42)),
                'num_generated': int(num_generated) if num_generated is not None else int(config['training']['zero_shot']['num_generated']),
                'num_generated_grid': str(num_generated_grid) if num_generated_grid is not None else None,
                'skip_mlp': bool(skip_mlp),
                'zsl_only': bool(zsl_only),
                'enable_gen_filter': bool(enable_gen_filter),
                'enable_fusion_alpha_tune': bool(enable_fusion_alpha_tune),
                'allow_sem_knn_mix_select': bool(allow_sem_knn_mix_select),
                'calib_reject_degenerate': bool(reject_degenerate),
                'calib_prefer_sem_knn_mix': bool(prefer_mix),
                'calib_prefer_mix_delta': float(prefer_mix_delta),
                'calib_robust_semknn': bool(robust_semknn),
                'calib_robust_repeats': int(robust_repeats),
                'calib_robust_val_ratio': float(robust_val_ratio),
                'calib_robust_seed': (int(robust_seed) if robust_seed is not None else None),
                'num_workers': int(effective_num_workers),
                'max_files_per_class': int(max_files_per_class) if max_files_per_class is not None else None,
                'max_segments_per_file': int(max_segments_per_file) if max_segments_per_file is not None else None,
                'sample_rate': int(resolved_sr) if resolved_sr is not None else None,
                'vibration_col': int(vibration_col),
                'eval_noise_std': float(eval_noise_std) if eval_noise_std is not None else 0.0,
                'eval_noise_snr_db': (None if eval_noise_snr_db is None else float(eval_noise_snr_db)),
                'eval_noise_seed': (None if eval_noise_seed is None else int(eval_noise_seed)),
                'checkpoint': str(ckpt_abs) if ckpt_abs is not None else None,
            },
        }
        out_path = os.path.abspath(results_json)
        out_dir = os.path.dirname(out_path) or os.getcwd()
        os.makedirs(out_dir, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(prefix=os.path.basename(out_path) + '.', suffix='.tmp', dir=out_dir)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, out_path)
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except:
                pass
        print(f"\n  Results written to: {results_json}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='ESAG 百度齿轮数据 Training')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--save-dir', type=str, default=None)
    parser.add_argument('--max-files-per-class', type=int, default=10)
    parser.add_argument('--max-segments-per-file', type=int, default=10)
    parser.add_argument('--results-json', type=str, default=None)
    parser.add_argument('--zsl-support', type=str, default='all_seen')
    parser.add_argument('--num-generated', type=int, default=None)
    parser.add_argument('--seen-classes', type=str, default=None)
    parser.add_argument('--unseen-classes', type=str, default=None)
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--unseen-calib-ratio', type=float, default=0.0)
    parser.add_argument('--calib-seed', type=int, default=None)
    parser.add_argument('--num-generated-grid', type=str, default=None)
    parser.add_argument('--calib-metric', type=str, default='accuracy', choices=['accuracy', 'f1'])
    parser.add_argument('--skip-mlp', action='store_true')
    parser.add_argument('--zsl-only', action='store_true')
    parser.add_argument('--disable-gen-filter', action='store_true')
    parser.add_argument('--disable-fusion-alpha-tune', action='store_true')
    parser.add_argument('--calib-allow-sem-knn-mix-select', action='store_true')
    parser.add_argument('--calib-reject-degenerate', action='store_true',
                        help='During unseen-calib head selection, reject heads with near-single-class predictions to avoid misleading calib accuracy.')
    parser.add_argument('--calib-prefer-sem-knn-mix', action='store_true',
                        help='If sem_knn_gate* wins but sem_knn_mix is close on the calib metric, prefer sem_knn_mix for stability.')
    parser.add_argument('--calib-prefer-mix-delta', type=float, default=0.05,
                        help='Tolerance delta for preferring sem_knn_mix: select mix if mix_score >= best_score - delta.')
    parser.add_argument('--calib-robust-semknn', action='store_true',
                        help='Robust SemKNN head selection: repeated stratified split on unseen calib, re-tune gate/mix on train and compare val scores.')
    parser.add_argument('--calib-robust-repeats', type=int, default=5,
                        help='Number of repeated splits for --calib-robust-semknn (default: 5).')
    parser.add_argument('--calib-robust-val-ratio', type=float, default=0.2,
                        help='Validation ratio for each split in --calib-robust-semknn (default: 0.2).')
    parser.add_argument('--calib-robust-seed', type=int, default=None,
                        help='Random seed for --calib-robust-semknn. If omitted, falls back to gen/mlp seed.')
    parser.add_argument('--eval-noise-std', type=float, default=0.0,
                        help='Additive Gaussian noise std on raw signal during eval (query side; applied when augment=False).')
    parser.add_argument('--eval-noise-snr-db', type=float, default=None,
                        help='Additive Gaussian noise by target SNR(dB) on raw signal during eval; overrides --eval-noise-std when set.')
    parser.add_argument('--eval-noise-seed', type=int, default=0,
                        help='Random seed for deterministic per-sample eval noise (default: 0). Use None to make it non-deterministic.')
    parser.add_argument(
        '--num-workers',
        type=int,
        default=None,
        help='Override DataLoader num_workers. If omitted: eval-only uses 0; training uses 0 on Windows, else config runtime.num_workers.',
    )
    parser.add_argument('--sample-rate', type=int, default=None,
                        help='Override sample rate (Hz). If omitted, tries to parse from CSV header first line.')
    parser.add_argument('--vibration-col', type=int, default=0,
                        help='0-based column index to use as vibration signal from each CSV data row (default: 0).')
    args = parser.parse_args()
    
    train_gear(
        config_path=args.config,
        data_path=args.data_path,
        num_epochs=args.epochs,
        seed=args.seed,
        save_dir=args.save_dir,
        max_files_per_class=args.max_files_per_class,
        max_segments_per_file=args.max_segments_per_file,
        results_json=args.results_json,
        zsl_support=args.zsl_support,
        num_generated=args.num_generated,
        seen_classes_csv=args.seen_classes,
        unseen_classes_csv=args.unseen_classes,
        eval_only=bool(args.eval_only),
        checkpoint_path=args.checkpoint,
        unseen_calib_ratio=float(args.unseen_calib_ratio),
        calib_seed=args.calib_seed,
        num_generated_grid=args.num_generated_grid,
        calib_metric=str(args.calib_metric),
        disable_gen_filter=bool(args.disable_gen_filter),
        disable_fusion_alpha_tune=bool(args.disable_fusion_alpha_tune),
        calib_allow_sem_knn_mix_select=bool(args.calib_allow_sem_knn_mix_select),
        calib_reject_degenerate=bool(args.calib_reject_degenerate),
        calib_prefer_sem_knn_mix=bool(args.calib_prefer_sem_knn_mix),
        calib_prefer_mix_delta=float(args.calib_prefer_mix_delta),
        calib_robust_semknn=bool(args.calib_robust_semknn),
        calib_robust_repeats=int(args.calib_robust_repeats),
        calib_robust_val_ratio=float(args.calib_robust_val_ratio),
        calib_robust_seed=(int(args.calib_robust_seed) if args.calib_robust_seed is not None else None),
        eval_noise_std=float(args.eval_noise_std) if args.eval_noise_std is not None else 0.0,
        eval_noise_snr_db=(None if args.eval_noise_snr_db is None else float(args.eval_noise_snr_db)),
        eval_noise_seed=(None if args.eval_noise_seed is None else int(args.eval_noise_seed)),
        skip_mlp=bool(args.skip_mlp),
        zsl_only=bool(args.zsl_only),
        sample_rate=args.sample_rate,
        vibration_col=int(args.vibration_col),
        num_workers=args.num_workers,
    )


if __name__ == '__main__':
    # 此脚本仅作为 BaiduGearDataset 数据加载模块使用（SDP 复现）。
    # 请用 code/reproduce.py 或 code/sdp_experiments/ 下的训练脚本。
    print('train_gear_baidu.py 仅提供 BaiduGearDataset；请用 reproduce.py 复现')
