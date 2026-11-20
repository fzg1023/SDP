# -*- coding: utf-8 -*-
"""
NEU-Gear 补全 10 对 + 机理原型对照（P1-3 补全 10 对 / P0-1 加对照）。

同一训练模型，测试时分别用两种原型：
  1) 样本原型 attr_raw  = 每类 20% 标定样本的属性均值（数据驱动，含未见类支持样本）
  2) 机理原型 attr_mech = 仅由先验运动学量构造：齿轮振动无论健康/故障都以啮频为谱峰，
     故频率位置维 e_centroid/e_peak_freq 对所有类取同一啮频值；其余 19 维的类特定值
     无法由简单运动学先验推导、取全局均值（z-score 后为 0）。因而机理原型对所有类相同、
     不可区分，各对均退化为随机（≈50%）。诊断已证实健康类谱峰≈758Hz（≈45 齿）、并非
     转频 16.67Hz，故不采用「健康=转频、故障=啮频」的错误假设，保持诚实。

5 种子（42/49/56/63/70），逐 pair 记录两种原型的 per-seed / mean / std。
结果存 results_gear_10pairs_mechproto.json。
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "demo"))
sys.path.insert(0, HERE)

from esag.utils import load_config, set_seed, get_device
from esag.preprocessing import SignalPreprocessor
from demo.train_gear_baidu import BaiduGearDataset
from train_fix_v10 import image_features, CachedImageDataset, build, train_seen, eval_scores

CFG = os.path.join(ROOT, "configs", "light_semantic.yaml")
GEAR_DATA = os.environ.get("SDP_GEAR_DATA", "D:/BaiduNetdiskDownload/GearDatasets")
CLASS_NAMES = ["健康", "剥落", "断齿", "磨损", "裂纹"]
# 全部 C(5,2)=10 对（固定顺序）
ALL_PAIRS = [
    ("健康", "剥落"), ("健康", "断齿"), ("健康", "磨损"), ("健康", "裂纹"),
    ("剥落", "断齿"), ("剥落", "磨损"), ("剥落", "裂纹"),
    ("断齿", "磨损"), ("断齿", "裂纹"),
    ("磨损", "裂纹"),
]
MAX_FILES = 50
MAX_SEGMENTS = 50
ATTR_HOLDOUT = 0.2
ANNOT_SEED = 42
FS = 15360
NYQ = FS / 2.0
# 齿轮机理参数（转速 1000rpm -> 转频 16.67Hz；啮频 = 转频 × 主动轮齿数）
FR = 1000.0 / 60.0          # 转频 Hz
Z_DRIVER = 55                # 主动轮齿数（假定值；诊断显示谱峰≈758Hz 对应约 45 齿，
                              # 但啮频值对所有类相同，其精确值不影响「不可区分」结论）
FM = FR * Z_DRIVER           # 啮频 Hz
FR_NORM = FR / NYQ
FM_NORM = FM / NYQ
# 21 维索引（见 train_fix_v10.image_features）：e_centroid=13, e_peak_freq=19
IDX_CENTROID = 13
IDX_PEAKFREQ = 19


def build_mech_proto(n_classes=5, dim=21):
    """构造机理原型（z-score 空间）。

    唯一可先验确定的是齿轮啮频 FM：健康与故障齿轮的振动均以啮频为谱峰（诊断证实健康类
    谱峰≈758Hz、并非转频 16.67Hz）。故频率位置维对所有类取同一啮频值，其余维取全局均值
    （z=0）。原型对所有类相同 → 不可区分 → 各对退化为随机（50%）。
    """
    mech = np.zeros((n_classes, dim))
    mech[:, IDX_CENTROID] = FM_NORM
    mech[:, IDX_PEAKFREQ] = FM_NORM
    return mech


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--drop-path", type=float, default=0.3)
    ap.add_argument("--backbone", type=str, default="small")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--pairs", type=str, default=None,
                    help="逗号分隔的类名对，如 健康,剥落,断齿,裂纹；缺省跑全部 10 对")
    ap.add_argument("--results-json", type=str,
                    default=os.path.join(HERE, "results_gear_10pairs_mechproto.json"))
    args = ap.parse_args()

    if args.pairs:
        names = [p.strip() for p in args.pairs.split(",") if p.strip()]
        pairs = [tuple(names[i:i + 2]) for i in range(0, len(names), 2)]
    else:
        pairs = ALL_PAIRS

    config = load_config(CFG)
    device = get_device("auto")
    seeds = [42 + i * 7 for i in range(args.seeds)]
    print(f"[info] FR={FR:.2f}Hz FM={FM:.2f}Hz FR_NORM={FR_NORM:.4f} FM_NORM={FM_NORM:.4f}",
          flush=True)

    loader = BaiduGearDataset(data_path=GEAR_DATA, signal_length=2048,
                              max_files_per_class=MAX_FILES, max_segments_per_file=MAX_SEGMENTS,
                              vibration_col=0)
    signals, labels, class_indices, class_names = loader.load_data()
    n2i = {str(n): i for i, n in enumerate(class_names)}
    print(f"[data] n={len(signals)} classes={class_names}", flush=True)

    feat = np.stack([image_features(s) for s in signals])

    rng = np.random.RandomState(ANNOT_SEED)
    attr_pos, zsl_pos = [], []
    for c in range(5):
        cidx = np.where(labels == c)[0]
        rng.shuffle(cidx)
        n_attr = max(1, int(round(len(cidx) * ATTR_HOLDOUT)))
        attr_pos.extend(cidx[:n_attr])
        zsl_pos.extend(cidx[n_attr:])
    attr_pos = np.array(attr_pos, dtype=np.int64)
    zsl_pos = np.array(zsl_pos, dtype=np.int64)

    mu = feat[attr_pos].mean(0)
    sd = feat[attr_pos].std(0) + 1e-12
    Z = (feat - mu) / sd
    attr_raw = np.stack([Z[attr_pos][labels[attr_pos] == i].mean(0) for i in range(5)])
    attr_mech = build_mech_proto()
    attr_t = torch.tensor(attr_raw, dtype=torch.float32, device=device)
    attr_m = torch.tensor(attr_mech, dtype=torch.float32, device=device)
    feat_zsl = torch.tensor(Z[zsl_pos], dtype=torch.float32)

    preprocessor = SignalPreprocessor(signal_length=2048, sample_rate=FS, output_size=64,
                                      channel_mode="dual")
    print("[prep] precomputing images...", flush=True)
    images_all = torch.from_numpy(
        np.stack([preprocessor.process_single(signals[i]) for i in zsl_pos]).astype(np.float32))
    labels_zsl = torch.tensor(labels[zsl_pos], dtype=torch.long)
    print(f"[prep] zsl images={tuple(images_all.shape)}", flush=True)

    rows = []
    for a_name, b_name in pairs:
        a, b = n2i[a_name], n2i[b_name]
        seen_names = [n for n in CLASS_NAMES if n not in (a_name, b_name)]
        seen_classes = [n2i[n] for n in seen_names]
        seen_pos = np.where(np.isin(labels[zsl_pos], seen_classes))[0]
        sel = (labels_zsl == a) | (labels_zsl == b)
        true = torch.tensor([0 if int(x) == a else 1 for x in labels_zsl[sel].tolist()])

        per_seed_sample, per_seed_mech = [], []
        print(f"\n=== {a_name}/{b_name} | seen={seen_names} ===", flush=True)
        for s in seeds:
            set_seed(s)
            ds = CachedImageDataset(images_all, labels_zsl, feat_zsl, seen_pos)
            train_loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=0,
                                      drop_last=True)
            model = build(config, device, attr_t.shape[1], drop_path=args.drop_path,
                          backbone=args.backbone)
            train_seen(model, train_loader, attr_t, seen_classes, device, args.epochs,
                       os.path.join(HERE, "checkpoints", f"mech10_{a_name}_{b_name}_s{s}"),
                       lr=args.lr, wd=args.wd)
            model.eval()
            sc_sample = eval_scores(model, images_all, attr_t, device)
            sc_mech = eval_scores(model, images_all, attr_m, device)
            pred_sample = sc_sample[sel][:, torch.tensor([a, b])].argmax(1)
            pred_mech = sc_mech[sel][:, torch.tensor([a, b])].argmax(1)
            acc_sample = float((pred_sample == true).float().mean().item())
            acc_mech = float((pred_mech == true).float().mean().item())
            per_seed_sample.append(acc_sample)
            per_seed_mech.append(acc_mech)
            print(f"  [seed {s}] sample={acc_sample * 100:.2f}% mech={acc_mech * 100:.2f}%",
                  flush=True)

        rows.append({
            "pair": f"{a_name}/{b_name}", "seen": seen_names,
            "sample_per_seed": [round(x, 4) for x in per_seed_sample],
            "mech_per_seed": [round(x, 4) for x in per_seed_mech],
            "sample_mean": round(float(np.mean(per_seed_sample)), 4),
            "sample_std": round(float(np.std(per_seed_sample)), 4),
            "mech_mean": round(float(np.mean(per_seed_mech)), 4),
            "mech_std": round(float(np.std(per_seed_mech)), 4),
        })
        print(f"  [{a_name}/{b_name}] sample={np.mean(per_seed_sample) * 100:.2f}% "
              f"mech={np.mean(per_seed_mech) * 100:.2f}%", flush=True)

    arr_s = np.array([r["sample_per_seed"] for r in rows])
    arr_m = np.array([r["mech_per_seed"] for r in rows])
    out = {
        "note": "NEU-Gear 10 对 × 5 种子；样本原型 vs 机理原型对照",
        "seeds": seeds, "epochs": args.epochs, "backbone": args.backbone,
        "mech_proto": {"FR_Hz": FR, "FM_Hz": FM, "Z_driver": Z_DRIVER,
                       "note": "啮频为唯一先验量；健康/故障均以啮频为谱峰，故对所有类取同一值→不可区分",
                       "dims_used": ["e_centroid", "e_peak_freq"],
                       "other_dims": "global mean (z-score 0)"},
        "rows": rows,
        "sample_overall_mean": round(float(arr_s.mean()), 4),
        "sample_overall_std": round(float(arr_s.mean(0).std()), 4),
        "mech_overall_mean": round(float(arr_m.mean()), 4),
        "mech_overall_std": round(float(arr_m.mean(0).std()), 4),
    }
    json.dump(out, open(args.results_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n[已另存] {args.results_json}")
    print(f"sample 10对均值={arr_s.mean() * 100:.2f}% ± {arr_s.mean(0).std() * 100:.2f}%")
    print(f"mech   10对均值={arr_m.mean() * 100:.2f}% ± {arr_m.mean(0).std() * 100:.2f}%")


if __name__ == "__main__":
    main()
