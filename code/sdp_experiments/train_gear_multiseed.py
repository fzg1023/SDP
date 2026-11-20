"""
BaiduGear 主实验 —— 5 seed 重跑，记录逐 seed 准确率 + 均值 ± std（第 2 项）。

复用 v10 机制：20% holdout 标定属性、80% ZSL、逐样本 MSE、small CNN、语义点积。
5 对 × N seed，逐 pair 输出 per-seed 列表、mean、std。
结果另存 results_gear_multiseed.json。
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np
import torch
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
ALL_PAIRS = [("健康", "裂纹"), ("剥落", "磨损"), ("断齿", "磨损"),
             ("剥落", "裂纹"), ("剥落", "断齿")]
MAX_FILES = 50
MAX_SEGMENTS = 50
ATTR_HOLDOUT = 0.2
ANNOT_SEED = 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--drop-path", type=float, default=0.3)
    ap.add_argument("--backbone", type=str, default="small")
    ap.add_argument("--channel-mode", type=str, default="dual",
                    choices=["dual", "env_only", "cwt_only"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--results-json", type=str,
                    default=os.path.join(HERE, "results_gear_multiseed.json"))
    args = ap.parse_args()

    config = load_config(CFG)
    device = get_device("auto")
    seeds = [42 + i * 7 for i in range(args.seeds)]

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
    attr_t = torch.tensor(attr_raw, dtype=torch.float32, device=device)
    feat_zsl = torch.tensor(Z[zsl_pos], dtype=torch.float32)

    preprocessor = SignalPreprocessor(signal_length=2048, sample_rate=15360, output_size=64,
                                      channel_mode=args.channel_mode)
    print(f"[prep] precomputing images (channel_mode={args.channel_mode})...", flush=True)
    images_all = torch.from_numpy(
        np.stack([preprocessor.process_single(signals[i]) for i in zsl_pos]).astype(np.float32))
    labels_zsl = torch.tensor(labels[zsl_pos], dtype=torch.long)
    print(f"[prep] zsl images={tuple(images_all.shape)}", flush=True)

    rows = []
    all_per_seed = []
    for a_name, b_name in ALL_PAIRS:
        a, b = n2i[a_name], n2i[b_name]
        seen_names = [n for n in CLASS_NAMES if n not in (a_name, b_name)]
        seen_classes = [n2i[n] for n in seen_names]
        seen_pos = np.where(np.isin(labels[zsl_pos], seen_classes))[0]
        sel = (labels_zsl == a) | (labels_zsl == b)
        true = torch.tensor([0 if int(x) == a else 1 for x in labels_zsl[sel].tolist()])

        per_seed = []
        print(f"\n=== {a_name}/{b_name} | seen={seen_names} ===", flush=True)
        for s in seeds:
            set_seed(s)
            ds = CachedImageDataset(images_all, labels_zsl, feat_zsl, seen_pos)
            train_loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=0,
                                      drop_last=True)
            model = build(config, device, attr_t.shape[1], drop_path=args.drop_path,
                          backbone=args.backbone)
            train_seen(model, train_loader, attr_t, seen_classes, device, args.epochs,
                       os.path.join(HERE, "checkpoints", f"multiseed_{a_name}_{b_name}_s{s}"),
                       lr=args.lr, wd=args.wd)
            model.eval()
            sc = eval_scores(model, images_all, attr_t, device)
            pred = sc[sel][:, torch.tensor([a, b])].argmax(1)
            acc = float((pred == true).float().mean().item())
            per_seed.append(acc)
            print(f"  [seed {s}] acc={acc * 100:.2f}%", flush=True)

        mean = float(np.mean(per_seed))
        std = float(np.std(per_seed))
        all_per_seed.append(per_seed)
        rows.append({"pair": f"{a_name}/{b_name}", "seen": seen_names,
                     "per_seed": [round(x, 4) for x in per_seed],
                     "mean": round(mean, 4), "std": round(std, 4)})
        print(f"  [{a_name}/{b_name}] mean={mean * 100:.2f}% std={std * 100:.2f}%", flush=True)

    # 5 对平均（先对各 seed 求 5 对均值，再算跨 seed 的 mean/std）
    arr = np.array(all_per_seed)  # (n_pairs, n_seeds)
    pair_mean = arr.mean(0)       # 每个 seed 的 5 对均值
    grand_mean = float(pair_mean.mean())
    grand_std = float(pair_mean.std())
    print(f"\n=== 5 对 × {len(seeds)} seed 总体 = {grand_mean * 100:.2f}% ± {grand_std * 100:.2f}% ===",
          flush=True)

    out = {"note": "BaiduGear 主实验 5-seed（逐 seed 记录）", "seeds": seeds,
           "epochs": args.epochs, "backbone": args.backbone,
           "rows": rows, "overall_mean": round(grand_mean, 4), "overall_std": round(grand_std, 4)}
    json.dump(out, open(args.results_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n[已另存] {args.results_json}")


if __name__ == "__main__":
    main()
