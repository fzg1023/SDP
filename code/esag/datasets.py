"""
轴承故障数据集加载器
支持: CWRU, PU (Paderborn), JNU 以及模拟数据
"""

import os
import glob
import re
import itertools
from collections import deque
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from scipy.io import loadmat
from typing import Dict, List, Tuple, Optional

from .preprocessing import SignalPreprocessor


# ==============================================================================
# 基础数据集类
# ==============================================================================

class BearingDataset(Dataset):
    """
    通用轴承故障数据集基类
    """

    def __init__(
        self,
        signals: np.ndarray,
        labels: np.ndarray,
        class_indices: np.ndarray,
        preprocessor: SignalPreprocessor,
        augment: bool = False,
        eval_noise_std: float = 0.0,
        eval_noise_snr_db: Optional[float] = None,
        eval_noise_seed: Optional[int] = None,
    ):
        """
        Args:
            signals: (N, signal_length) 原始振动信号
            labels: (N,) 数字标签
            class_indices: (N,) 属性矩阵中的索引
            preprocessor: 信号预处理器
            augment: 是否做数据增强
        """
        self.signals = signals
        self.labels = labels
        self.class_indices = class_indices
        self.preprocessor = preprocessor
        self.augment = augment

        # Evaluation-time noise injection (for robustness curves).
        # Applied ONLY when augment=False to keep training augmentation behavior unchanged.
        self.eval_noise_std = float(eval_noise_std) if eval_noise_std is not None else 0.0
        self.eval_noise_snr_db = None if eval_noise_snr_db is None else float(eval_noise_snr_db)
        self.eval_noise_seed = None if eval_noise_seed is None else int(eval_noise_seed)

    def _apply_eval_noise(self, signal: np.ndarray, *, idx: int) -> np.ndarray:
        if bool(self.augment):
            return signal

        snr_db = self.eval_noise_snr_db
        noise_std = float(self.eval_noise_std)
        if (snr_db is None) and (noise_std <= 0.0):
            return signal

        # Deterministic per-sample noise for reproducible eval.
        if self.eval_noise_seed is None:
            rng = np.random
        else:
            rng = np.random.RandomState(int(self.eval_noise_seed) + int(idx))

        sig = np.asarray(signal, dtype=np.float32)
        if snr_db is not None:
            x = sig - float(sig.mean())
            rms = float(np.sqrt(np.mean(x * x)) + 1e-8)
            # SNR(dB) = 20*log10(rms_signal / rms_noise)
            rms_noise = float(rms / (10.0 ** (float(snr_db) / 20.0)))
            noise_std = float(rms_noise)

        if noise_std <= 0.0 or not np.isfinite(noise_std):
            return sig

        noise = rng.randn(int(sig.shape[0])).astype(np.float32) * np.float32(noise_std)
        return (sig + noise).astype(np.float32)

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal = self.signals[idx]
        label = self.labels[idx]
        class_idx = self.class_indices[idx]

        signal = self._apply_eval_noise(signal, idx=int(idx))

        # 信号预处理 -> 双通道图像
        if self.augment:
            image = self.preprocessor.process_with_augmentation(signal)
        else:
            image = self.preprocessor.process_single(signal)

        return {
            'image': torch.from_numpy(image),
            'label': torch.tensor(label, dtype=torch.long),
            'class_index': torch.tensor(class_idx, dtype=torch.long),
        }


# ==============================================================================
# CWRU 数据集加载器
# ==============================================================================

class CWRUDataset:
    """
    Case Western Reserve University 轴承数据集
    12k / 48k 采样率，4种载荷 (0-3 HP)
    10种故障类型：Normal, IR007/014/021, B007/014/021, OR007/014/021
    """

    # 故障类型到文件关键字的映射
    FAULT_MAP = {
        'Normal': ['normal', 'Normal'],
        'IR007': ['IR007', 'ir007'],
        'IR014': ['IR014', 'ir014'],
        'IR021': ['IR021', 'ir021'],
        'B007': ['B007', 'ball007'],
        'B014': ['B014', 'ball014'],
        'B021': ['B021', 'ball021'],
        'OR007': ['OR007', 'or007'],
        'OR014': ['OR014', 'or014'],
        'OR021': ['OR021', 'or021'],
    }

    # 故障类型到属性矩阵索引的映射
    CLASS_TO_ATTR = {
        'Normal': 0,   # Normal
        'IR007': 1,    # IR_slight
        'IR014': 2,    # IR_medium
        'IR021': 3,    # IR_severe
        'OR007': 4,    # OR_slight
        'OR014': 5,    # OR_medium
        'OR021': 6,    # OR_severe
        'B007': 7,     # B_slight
        'B014': 8,     # B_medium
        'B021': 9,     # B_severe
    }

    def __init__(self, data_path: str, signal_length: int = 2048,
                 overlap: float = 0.5, sample_rate: int = 12000,
                 loads: list = None):
        self.data_path = data_path
        self.signal_length = signal_length
        self.overlap = overlap
        self.sample_rate = sample_rate
        self.loads = loads or [0, 1, 2, 3]

    def load_mat_file(self, file_path: str) -> Optional[np.ndarray]:
        """加载单个 .mat 文件"""
        try:
            mat_data = loadmat(file_path)
            # CWRU 数据 key 通常包含 'DE_time', 'FE_time' 等
            for key in mat_data:
                if 'DE_time' in key:
                    return mat_data[key].flatten()
            # 尝试其它常见 key
            for key in mat_data:
                if isinstance(mat_data[key], np.ndarray) and mat_data[key].ndim <= 2:
                    data = mat_data[key].flatten()
                    if len(data) > self.signal_length:
                        return data
            return None
        except Exception as e:
            print(f"Warning: Failed to load {file_path}: {e}")
            return None

    def segment_signal(self, signal: np.ndarray) -> np.ndarray:
        """将长信号分段"""
        step = int(self.signal_length * (1 - self.overlap))
        segments = []
        for start in range(0, len(signal) - self.signal_length + 1, step):
            seg = signal[start:start + self.signal_length]
            segments.append(seg)
        return np.array(segments)

    def load_data(self, fault_types: list = None, loads: list = None
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        加载 CWRU 数据集
        
        Returns:
            signals: (N, signal_length)
            labels: (N,) 数字标签
            class_indices: (N,) 属性矩阵索引
            class_names: 类别名称列表
        """
        if fault_types is None:
            fault_types = list(self.FAULT_MAP.keys())
        if loads is None:
            loads = self.loads

        all_signals = []
        all_labels = []
        all_class_indices = []
        class_names = fault_types

        for label_idx, fault_type in enumerate(fault_types):
            keywords = self.FAULT_MAP[fault_type]
            attr_idx = self.CLASS_TO_ATTR[fault_type]

            # 搜索匹配的文件
            for load in loads:
                found = False
                for keyword in keywords:
                    pattern = os.path.join(self.data_path,
                                           f'**/*{keyword}*{load}*.mat')
                    files = glob.glob(pattern, recursive=True)
                    if not files:
                        pattern = os.path.join(self.data_path,
                                               f'**/*{keyword}*.mat')
                        files = glob.glob(pattern, recursive=True)

                    for f in files:
                        data = self.load_mat_file(f)
                        if data is not None:
                            segments = self.segment_signal(data)
                            all_signals.append(segments)
                            all_labels.extend([label_idx] * len(segments))
                            all_class_indices.extend([attr_idx] * len(segments))
                            found = True
                            break
                    if found:
                        break

        if not all_signals:
            raise FileNotFoundError(
                f"No data found in {self.data_path}. "
                f"Please download CWRU dataset first."
            )

        signals = np.concatenate(all_signals, axis=0)
        labels = np.array(all_labels)
        class_indices = np.array(all_class_indices)

        return signals, labels, class_indices, class_names


# ==============================================================================
# PU (Paderborn) 数据集加载器
# ==============================================================================

class PUDataset:
    """Paderborn University (PU) 轴承数据集加载器。

    该仓库默认按如下目录结构组织 PU 数据：
      pu/
        Healthy/*.mat
        IR/*.mat
        OR/*.mat
        Compound/*.mat

    每个 .mat 文件通常包含一个顶层变量（与文件名一致），其内容是结构体，
    其中 `Y` 字段包含多个通道，常见振动通道名为 `vibration_1`。
    """

    CLASS_TO_ATTR = {
        'Healthy': 0,     # Normal
        'IR': 2,          # IR_medium
        'OR': 5,          # OR_medium
        'Compound': 10,   # IR_OR
    }

    def __init__(
        self,
        data_path: str,
        signal_length: int = 2048,
        overlap: float = 0.5,
        sample_rate: int = 64000,
        max_files_per_class: Optional[int] = 50,
        max_segments_per_file: Optional[int] = 50,
    ):
        self.data_path = data_path
        self.signal_length = signal_length
        self.overlap = overlap
        self.sample_rate = sample_rate
        self.max_files_per_class = max_files_per_class
        self.max_segments_per_file = max_segments_per_file

    def _load_mat_signal(self, file_path: str) -> Optional[np.ndarray]:
        """加载单个 PU .mat 文件中的振动信号（优先 vibration 通道）。"""
        try:
            mat_data = loadmat(file_path)
            keys = [k for k in mat_data.keys() if not k.startswith('__')]
            if not keys:
                return None

            root = mat_data[keys[0]]
            if not (isinstance(root, np.ndarray) and root.size > 0):
                return None

            rec = root[0, 0]
            if not (hasattr(rec, 'dtype') and rec.dtype.names and ('Y' in rec.dtype.names)):
                return None

            Y = rec['Y']
            if not (isinstance(Y, np.ndarray) and Y.ndim == 2 and Y.shape[0] == 1):
                return None

            best = None
            best_len = 0
            for i in range(Y.shape[1]):
                yi = Y[0, i]
                if not (hasattr(yi, 'dtype') and yi.dtype.names and ('Data' in yi.dtype.names) and ('Name' in yi.dtype.names)):
                    continue

                name = yi['Name']
                try:
                    name_str = str(name.squeeze()).lower()
                except Exception:
                    name_str = str(name).lower()

                data = yi['Data']
                if data is None:
                    continue
                data_arr = np.array(data).reshape(-1)
                if data_arr.size <= self.signal_length:
                    continue

                # 优先选择 vibration 通道；否则选最长的数值序列
                score = 2 if ('vibration' in name_str) else 1
                if (score == 2 and data_arr.size > best_len) or (best is None) or (data_arr.size > best_len and score == 1):
                    best = data_arr
                    best_len = data_arr.size

            if best is None:
                return None

            return best.astype(np.float32, copy=False)
        except Exception as e:
            print(f"Warning: Failed to load {file_path}: {e}")
            return None

    def segment_signal(self, signal: np.ndarray) -> np.ndarray:
        """将长信号分段，并可限制每个文件的 segment 数量。"""
        step = int(self.signal_length * (1 - self.overlap))
        if step <= 0:
            step = self.signal_length

        segments = []
        for start in range(0, len(signal) - self.signal_length + 1, step):
            segments.append(signal[start:start + self.signal_length])
            if self.max_segments_per_file is not None and len(segments) >= int(self.max_segments_per_file):
                break
        return np.asarray(segments, dtype=np.float32)

    def load_data(self, fault_types: list = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """加载 PU 数据集。

        Returns:
            signals: (N, signal_length)
            labels: (N,) 数字标签（与 fault_types 顺序一致）
            class_indices: (N,) 属性矩阵索引
            class_names: 类别名称列表
        """
        if fault_types is None:
            fault_types = ['Healthy', 'OR', 'IR', 'Compound']

        all_signals = []
        all_labels: List[int] = []
        all_class_indices: List[int] = []
        class_names = list(fault_types)

        for label_idx, fault_type in enumerate(fault_types):
            fault_dir = os.path.join(self.data_path, fault_type)
            files = sorted(glob.glob(os.path.join(fault_dir, '*.mat')))
            if self.max_files_per_class is not None:
                files = files[: int(self.max_files_per_class)]

            attr_idx = self.CLASS_TO_ATTR.get(fault_type, 0)
            for f in files:
                sig = self._load_mat_signal(f)
                if sig is None:
                    continue
                segments = self.segment_signal(sig)
                if segments.size == 0:
                    continue
                all_signals.append(segments)
                all_labels.extend([label_idx] * len(segments))
                all_class_indices.extend([attr_idx] * len(segments))

        if not all_signals:
            raise FileNotFoundError(
                f"No PU data found in {self.data_path}. "
                f"Expected subfolders like Healthy/IR/OR/Compound with .mat files."
            )

        signals = np.concatenate(all_signals, axis=0)
        labels = np.asarray(all_labels, dtype=np.int64)
        class_indices = np.asarray(all_class_indices, dtype=np.int64)
        return signals, labels, class_indices, class_names


# ======================================================================
# SEU (Southeast University) Gear/Bearing dataset (DDS) loader
# ======================================================================


class SEUGearDataset:
    """Southeast University (SEU) Gearbox dataset loader (gearset).

    This dataset is commonly distributed as a folder containing `gearset/` with
    large CSV files like:
      - Health_20_0.csv, Health_30_2.csv
      - Chipped_20_0.csv, ...
      - Miss_20_0.csv, Root_20_0.csv, Surface_20_0.csv, etc.

    Each CSV has a text header followed by a line `Data`, then numeric rows.
    Numeric rows contain 8 channels; prior README indicates channels 2/3/4 are
    effective (planetary gearbox vibration x/y/z).

    Notes on parsing:
    - Some lines may lack whitespace between adjacent floats (e.g. "0.00265-0.0203"),
      so we parse numbers via regex rather than whitespace splitting.
    - To keep runtime feasible, this loader supports `max_segments_per_file` early-stop
      and does streaming segmentation (does not load entire CSV into memory).
    """

    DEFAULT_CLASSES = ["Health", "Chipped", "Miss", "Root", "Surface"]

    # For SEU-Gear attribute set, we expect class_indices == label_idx (0..4).
    CLASS_TO_ATTR = {
        "Health": 0,
        "Chipped": 1,
        "Miss": 2,
        "Root": 3,
        "Surface": 4,
    }

    _FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

    def __init__(
        self,
        data_path: str,
        signal_length: int = 2048,
        overlap: float = 0.5,
        vibration_channel: int = 2,
        conditions: Optional[List[str]] = None,
        max_files_per_class: Optional[int] = None,
        max_segments_per_file: Optional[int] = 50,
    ):
        self.data_path = data_path
        self.signal_length = int(signal_length)
        self.overlap = float(overlap)
        self.vibration_channel = int(vibration_channel)
        self.conditions = conditions
        self.max_files_per_class = max_files_per_class
        self.max_segments_per_file = max_segments_per_file

        if self.vibration_channel not in (2, 3, 4):
            raise ValueError("SEUGearDataset vibration_channel must be one of {2,3,4} (effective channels)")

    def _resolve_gearset_dir(self) -> str:
        # Accept either the dataset root (containing gearset/) or the gearset folder itself.
        if os.path.isdir(os.path.join(self.data_path, "gearset")):
            return os.path.join(self.data_path, "gearset")
        return self.data_path

    @staticmethod
    def _iter_after_data_marker(f):
        in_data = False
        for line in f:
            if not in_data:
                if str(line).strip() == "Data":
                    in_data = True
                continue
            yield line

    @classmethod
    def _parse_row_floats(cls, line: str) -> Optional[List[float]]:
        # Extract floats even when adjacent numbers are glued (no whitespace).
        nums = cls._FLOAT_RE.findall(line)
        if len(nums) < 8:
            return None
        try:
            return [float(x) for x in nums[:8]]
        except Exception:
            return None

    def _stream_segments_from_csv(self, file_path: str) -> np.ndarray:
        step = int(self.signal_length * (1 - self.overlap))
        if step <= 0:
            step = self.signal_length

        col = self.vibration_channel - 1  # channel2->idx1, channel3->idx2, channel4->idx3

        segments: List[np.ndarray] = []
        buf = deque()

        # Use text mode; the CSV is ASCII-like.
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in self._iter_after_data_marker(f):
                row = self._parse_row_floats(line)
                if row is None:
                    continue

                buf.append(float(row[col]))
                if len(buf) < self.signal_length:
                    continue

                # Emit one segment and slide by `step`.
                seg = np.fromiter(itertools.islice(buf, 0, self.signal_length), dtype=np.float32, count=self.signal_length)
                if seg.shape[0] != self.signal_length:
                    continue
                segments.append(seg)

                for _ in range(step):
                    if not buf:
                        break
                    buf.popleft()

                if self.max_segments_per_file is not None and len(segments) >= int(self.max_segments_per_file):
                    break

        return np.asarray(segments, dtype=np.float32)

    def load_data(self, fault_types: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        if fault_types is None:
            fault_types = list(self.DEFAULT_CLASSES)

        gear_dir = self._resolve_gearset_dir()
        if not os.path.isdir(gear_dir):
            raise FileNotFoundError(f"SEU gearset directory not found: {gear_dir}")

        all_signals: List[np.ndarray] = []
        all_labels: List[int] = []
        all_class_indices: List[int] = []
        class_names = list(fault_types)

        for label_idx, fault_type in enumerate(fault_types):
            patterns: List[str] = []
            if self.conditions:
                for cond in self.conditions:
                    patterns.append(os.path.join(gear_dir, f"{fault_type}_{cond}.csv"))
            else:
                patterns.append(os.path.join(gear_dir, f"{fault_type}_*.csv"))

            files: List[str] = []
            for pat in patterns:
                files.extend(sorted(glob.glob(pat)))

            # De-dup
            files = sorted(list(dict.fromkeys(files)))
            if self.max_files_per_class is not None:
                files = files[: int(self.max_files_per_class)]

            attr_idx = int(self.CLASS_TO_ATTR.get(fault_type, label_idx))
            for fp in files:
                segs = self._stream_segments_from_csv(fp)
                if segs.size == 0:
                    continue
                all_signals.append(segs)
                all_labels.extend([label_idx] * int(segs.shape[0]))
                all_class_indices.extend([attr_idx] * int(segs.shape[0]))

        if not all_signals:
            raise FileNotFoundError(
                f"No SEU gear data found under {gear_dir}. "
                f"Expected files like Health_20_0.csv, Chipped_30_2.csv, etc."
            )

        signals = np.concatenate(all_signals, axis=0)
        labels = np.asarray(all_labels, dtype=np.int64)
        class_indices = np.asarray(all_class_indices, dtype=np.int64)
        return signals, labels, class_indices, class_names


# ==============================================================================
# 模拟数据生成器（用于 Demo / 测试）
# ==============================================================================

class SimulatedBearingData:
    """
    模拟轴承故障数据生成器
    基于物理模型生成逼真的轴承振动信号，用于 Demo 和测试
    """

    def __init__(self, sample_rate: int = 12000, signal_length: int = 2048,
                 shaft_freq: float = 30.0):
        """
        Args:
            sample_rate: 采样率
            signal_length: 信号长度
            shaft_freq: 轴转频 (Hz)
        """
        self.sample_rate = sample_rate
        self.signal_length = signal_length
        self.shaft_freq = shaft_freq

        # 标准轴承故障特征频率系数 (相对于轴转频)
        # 基于典型 6205 轴承参数
        self.bpfo_coef = 3.585   # 外圈故障特征频率系数
        self.bpfi_coef = 5.415   # 内圈故障特征频率系数
        self.bsf_coef = 2.357    # 滚动体故障特征频率系数
        self.ftf_coef = 0.398    # 保持架故障特征频率系数

    def _generate_base_signal(self) -> np.ndarray:
        """生成基础振动信号（正常轴承）"""
        t = np.arange(self.signal_length) / self.sample_rate
        # 轴转频分量
        signal = 0.1 * np.sin(2 * np.pi * self.shaft_freq * t)
        # 添加少量随机噪声
        signal += 0.02 * np.random.randn(self.signal_length)
        return signal, t

    def generate_normal(self, noise_level: float = 0.05) -> np.ndarray:
        """生成正常信号"""
        signal, t = self._generate_base_signal()
        signal += noise_level * np.random.randn(self.signal_length)
        return signal

    def generate_inner_race_fault(self, severity: float = 0.5,
                                  noise_level: float = 0.05) -> np.ndarray:
        """
        生成内圈故障信号
        特征：周期性冲击 + 幅值调制（由轴转频调制）
        """
        signal, t = self._generate_base_signal()
        fault_freq = self.bpfi_coef * self.shaft_freq

        # 周期性冲击
        impulse_period = 1.0 / fault_freq
        impulse_times = np.arange(0, t[-1], impulse_period)
        for imp_t in impulse_times:
            idx = int(imp_t * self.sample_rate)
            if idx < self.signal_length:
                # 衰减冲击
                decay = np.exp(-300 * np.arange(min(200, self.signal_length - idx)) / self.sample_rate)
                impact = severity * np.random.randn() * decay * \
                         np.sin(2 * np.pi * 3000 * np.arange(len(decay)) / self.sample_rate)
                signal[idx:idx + len(decay)] += impact

        # 幅值调制（内圈旋转导致故障区域进出载荷区）
        modulation = 1.0 + 0.5 * severity * np.sin(2 * np.pi * self.shaft_freq * t)
        signal *= modulation

        signal += noise_level * np.random.randn(self.signal_length)
        return signal

    def generate_outer_race_fault(self, severity: float = 0.5,
                                  noise_level: float = 0.05) -> np.ndarray:
        """
        生成外圈故障信号
        特征：周期性冲击（幅值较稳定，无明显调制）
        """
        signal, t = self._generate_base_signal()
        fault_freq = self.bpfo_coef * self.shaft_freq

        impulse_period = 1.0 / fault_freq
        impulse_times = np.arange(0, t[-1], impulse_period)
        for imp_t in impulse_times:
            idx = int(imp_t * self.sample_rate)
            if idx < self.signal_length:
                decay = np.exp(-250 * np.arange(min(200, self.signal_length - idx)) / self.sample_rate)
                impact = severity * (0.8 + 0.2 * np.random.randn()) * decay * \
                         np.sin(2 * np.pi * 2500 * np.arange(len(decay)) / self.sample_rate)
                signal[idx:idx + len(decay)] += impact

        signal += noise_level * np.random.randn(self.signal_length)
        return signal

    def generate_ball_fault(self, severity: float = 0.5,
                            noise_level: float = 0.05) -> np.ndarray:
        """
        生成滚动体故障信号
        特征：2倍 BSF 冲击 + 随机幅值变化
        """
        signal, t = self._generate_base_signal()
        fault_freq = 2 * self.bsf_coef * self.shaft_freq  # 2倍BSF

        impulse_period = 1.0 / fault_freq
        impulse_times = np.arange(0, t[-1], impulse_period)
        for imp_t in impulse_times:
            idx = int(imp_t * self.sample_rate)
            if idx < self.signal_length:
                decay = np.exp(-350 * np.arange(min(150, self.signal_length - idx)) / self.sample_rate)
                impact = severity * np.random.randn() * decay * \
                         np.sin(2 * np.pi * 3500 * np.arange(len(decay)) / self.sample_rate)
                signal[idx:idx + len(decay)] += impact

        # 保持架调制
        cage_mod = 1.0 + 0.3 * severity * np.sin(2 * np.pi * self.ftf_coef * self.shaft_freq * t)
        signal *= cage_mod

        signal += noise_level * np.random.randn(self.signal_length)
        return signal

    def generate_compound_fault(self, fault_types: list,
                                severities: list = None,
                                noise_level: float = 0.05) -> np.ndarray:
        """
        生成复合故障信号
        """
        if severities is None:
            severities = [0.5] * len(fault_types)

        signal = self.generate_normal(noise_level=0.01)

        generators = {
            'IR': self.generate_inner_race_fault,
            'OR': self.generate_outer_race_fault,
            'B': self.generate_ball_fault,
        }

        for fault_type, severity in zip(fault_types, severities):
            if fault_type in generators:
                fault_signal = generators[fault_type](severity, noise_level=0)
                # 减去基础信号避免重复
                base, _ = self._generate_base_signal()
                signal += (fault_signal - base)

        signal += noise_level * np.random.randn(self.signal_length)
        return signal

    def generate_dataset(self, samples_per_class: int = 200,
                         noise_level: float = 0.05,
                         loads: list = None
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        生成完整的模拟数据集
        
        Args:
            samples_per_class: 每类样本数
            noise_level: 噪声水平
            loads: 载荷列表（通过改变轴频模拟）
        
        Returns:
            signals, labels, class_indices, class_names
        """
        if loads is None:
            loads = [25.0, 30.0, 35.0]  # 不同轴频模拟不同载荷

        class_configs = [
            ('Normal', 'normal', None, 0, 0),
            ('IR_slight', 'IR', 0.3, 1, 1),
            ('IR_medium', 'IR', 0.6, 2, 2),
            ('IR_severe', 'IR', 0.9, 3, 3),
            ('OR_slight', 'OR', 0.3, 4, 4),
            ('OR_medium', 'OR', 0.6, 5, 5),
            ('OR_severe', 'OR', 0.9, 6, 6),
            ('B_slight', 'B', 0.3, 7, 7),
            ('B_medium', 'B', 0.6, 8, 8),
            ('B_severe', 'B', 0.9, 9, 9),
            ('IR_OR', 'compound_IR_OR', None, 10, 10),
            ('IR_B', 'compound_IR_B', None, 11, 11),
            ('OR_B', 'compound_OR_B', None, 12, 12),
        ]

        all_signals = []
        all_labels = []
        all_class_indices = []
        class_names = [c[0] for c in class_configs]

        generators = {
            'normal': lambda s, nl: self.generate_normal(nl),
            'IR': lambda s, nl: self.generate_inner_race_fault(s, nl),
            'OR': lambda s, nl: self.generate_outer_race_fault(s, nl),
            'B': lambda s, nl: self.generate_ball_fault(s, nl),
            'compound_IR_OR': lambda s, nl: self.generate_compound_fault(['IR', 'OR'], noise_level=nl),
            'compound_IR_B': lambda s, nl: self.generate_compound_fault(['IR', 'B'], noise_level=nl),
            'compound_OR_B': lambda s, nl: self.generate_compound_fault(['OR', 'B'], noise_level=nl),
        }

        for class_name, gen_key, severity, label_idx, attr_idx in class_configs:
            for load_freq in loads:
                self.shaft_freq = load_freq
                n_per_load = samples_per_class // len(loads)
                for _ in range(n_per_load):
                    if severity is not None:
                        sig = generators[gen_key](severity, noise_level)
                    else:
                        sig = generators[gen_key](0, noise_level)
                    all_signals.append(sig)
                    all_labels.append(label_idx)
                    all_class_indices.append(attr_idx)

        signals = np.array(all_signals, dtype=np.float32)
        labels = np.array(all_labels)
        class_indices = np.array(all_class_indices)

        return signals, labels, class_indices, class_names


# ==============================================================================
# 零样本数据集分割
# ==============================================================================

class ZeroShotSplitter:
    """
    零样本学习数据集分割器
    将类别分为已知 (seen) 和未知 (unseen)
    """

    @staticmethod
    def split(labels: np.ndarray, class_indices: np.ndarray,
              seen_classes: list, unseen_classes: list
              ) -> Dict[str, np.ndarray]:
        """
        按已知/未知类别分割数据
        
        Args:
            labels: 所有样本的标签
            class_indices: 所有样本的属性索引
            seen_classes: 已知类别的标签列表
            unseen_classes: 未知类别的标签列表
        
        Returns:
            dict with 'seen_indices', 'unseen_indices'
        """
        seen_mask = np.isin(labels, seen_classes)
        unseen_mask = np.isin(labels, unseen_classes)

        return {
            'seen_indices': np.where(seen_mask)[0],
            'unseen_indices': np.where(unseen_mask)[0],
            'seen_classes': seen_classes,
            'unseen_classes': unseen_classes,
        }

    @staticmethod
    def auto_split(labels: np.ndarray, seen_ratio: float = 0.7,
                   seed: int = 42) -> Dict:
        """
        自动按比例分割已知/未知类别
        """
        np.random.seed(seed)
        unique_classes = np.unique(labels)
        n_seen = max(2, int(len(unique_classes) * seen_ratio))
        perm = np.random.permutation(unique_classes)
        seen_classes = sorted(perm[:n_seen].tolist())
        unseen_classes = sorted(perm[n_seen:].tolist())

        return ZeroShotSplitter.split(labels, labels, seen_classes,
                                      unseen_classes)


def create_dataloaders(signals: np.ndarray, labels: np.ndarray,
                       class_indices: np.ndarray,
                       preprocessor: SignalPreprocessor,
                       split_info: dict,
                       batch_size: int = 64,
                       num_workers: int = 4,
                       pin_memory: bool = True,
                       drop_last_train: bool = True,
                       eval_noise_std: float = 0.0,
                       eval_noise_snr_db: Optional[float] = None,
                       eval_noise_seed: Optional[int] = None,
                       ) -> Dict[str, DataLoader]:
    """
    创建训练/测试 DataLoader
    """
    seen_idx = split_info['seen_indices']
    unseen_idx = split_info['unseen_indices']

    # 训练集：已知类别的 80%，验证集：剩余 20%
    # NOTE: 为了避免“监督/seen 指标”被训练集泄露，额外提供 `seen_support`(train-only) 与
    # `test_seen_holdout`(val-only) 两个 loader。
    n_train = int(len(seen_idx) * 0.8)
    perm = np.random.permutation(len(seen_idx))
    train_indices = seen_idx[perm[:n_train]]
    val_indices = seen_idx[perm[n_train:]]

    # 构建 Dataset
    full_dataset_clean = BearingDataset(
        signals, labels, class_indices, preprocessor, augment=False,
        eval_noise_std=0.0, eval_noise_snr_db=None, eval_noise_seed=None,
    )
    full_dataset_eval = BearingDataset(
        signals, labels, class_indices, preprocessor, augment=False,
        eval_noise_std=float(eval_noise_std),
        eval_noise_snr_db=eval_noise_snr_db,
        eval_noise_seed=eval_noise_seed,
    )
    train_dataset = BearingDataset(
        signals, labels, class_indices, preprocessor, augment=True
    )

    train_loader = DataLoader(
        Subset(train_dataset, train_indices),
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last_train,
    )

    # 训练集（不增强、可复现）——用于 zero-shot 支持集 / 监督线性探针
    seen_support_loader = DataLoader(
        Subset(full_dataset_clean, train_indices),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        Subset(full_dataset_eval, val_indices),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    test_seen_loader = DataLoader(
        Subset(full_dataset_clean, seen_idx),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    # Seen holdout（与 train 不重叠；默认与 val 相同）
    test_seen_holdout_loader = DataLoader(
        Subset(full_dataset_eval, val_indices),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    test_unseen_loader = DataLoader(
        Subset(full_dataset_eval, unseen_idx),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return {
        'train': train_loader,
        'val': val_loader,
        'seen_support': seen_support_loader,
        'test_seen': test_seen_loader,
        'test_seen_holdout': test_seen_holdout_loader,
        'test_unseen': test_unseen_loader,
    }
