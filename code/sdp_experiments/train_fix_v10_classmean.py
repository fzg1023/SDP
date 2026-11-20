"""
类均值回归消融 —— 干净重跑（3 seed × 5 对 × 50 epoch），证明「根因塌缩」。

与 train_fix_v10.py 唯一区别：训练回归目标从【样本自身特征】换成【类均值原型 attr_t[label]】。
其余完全一致：图像域 21 维特征、语义点积、严格 inductive、3 seed、无 CE、无早停。

预期：类均值回归把 seen 类压成紧簇 → 投影器学成 seen 分类器 → unseen 塌缩到接近随机（~30%）。

用法：
  D:/Anaconda/envs/esag/python.exe -u train_fix_v10_classmean.py --epochs 50 --seeds 3
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "demo"))
sys.path.insert(0, HERE)

from esag.utils import load_config, set_seed, get_device
from esag.preprocessing import SignalPreprocessor
from demo.train_gear_baidu import BaiduGearDataset
from train_fix_v10 import image_features, CachedImageDataset, build, eval_scores

CFG = os.path.join(ROOT, "configs", "light_semantic.yaml")
GEAR_DATA = os.environ.get("SDP_GEAR_DATA", "D:/BaiduNetdiskDownload/GearDatasets")
CLASS_NAMES = ["健康", "剥落", "断齿", "磨损", "裂纹"]
ALL_PAIRS = [("健康", "裂纹"), ("剥落", "磨损"), ("断齿", "磨损"),
             ("剥落", "裂纹"), ("剥落", "断齿")]
MAX_FILES = 50
MAX_SEGMENTS = 50
ATTR_HOLDOUT = 0.2
ANNOT_SEED = 42


def train_seen_classmean(model, loader, attr_t, device, epochs, save_dir,
                         lr=5e-4, wd=1e-3):
    """回归目标 = 类均值原型 attr_t[label]（关键差异：非逐样本特征）。"""
    params = list(model.backbone.parameters()) + list(model.attr_head.parameters())
    optimizer = AdamW(params, lr=lr, weight_decay=wd)
    warmup = 5
    scheduler = SequentialLR(
        optimizer,
        schedulers=[LinearLR(optimizer, start_factor=0.1, total_iters=warmup),
                    CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup), eta_min=1e-6)],
        milestones=[warmup])
    os.makedirs(save_dir, exist_ok=True)
    for ep in range(1, epochs + 1):
        model.train()
        tr = 0.0
        nb = 0
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            y_attr = attr_t[labels]                 # 类均值原型（塌缩根因）
            g = model.attr_head(model.backbone(images))
            loss = F.mse_loss(g, y_attr)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            tr += loss.item()
            nb += 1
        scheduler.step()
        if ep % 10 == 0 or ep == epochs:
            print(f"    [train] epoch {ep:3d}/{epochs}  reg={tr / nb:.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--drop-path", type=float, default=0.3)
    ap.add_argument("--backbone", type=str, default="swin_lite")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--results-json", type=str,
                    default=os.path.join(HERE, "results_classmean_ablation.json"))
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
    feat_zsl = torch.tensor(Z[zsl_pos], dtype=torch.float32)  # 仅用于缓存数据集（本实验不用作目标）

    preprocessor = SignalPreprocessor(signal_length=2048, sample_rate=15360, output_size=64)
    print("[prep] precomputing CWT images for ZSL split...", flush=True)
    images_all = torch.from_numpy(
        np.stack([preprocessor.process_single(signals[i]) for i in zsl_pos]).astype(np.float32))
    labels_zsl = torch.tensor(labels[zsl_pos], dtype=torch.long)
    print(f"[prep] zsl images={images_all.shape}", flush=True)

    rows = []
    for a_name, b_name in ALL_PAIRS:
        a, b = n2i[a_name], n2i[b_name]
        seen_names = [n for n in CLASS_NAMES if n not in (a_name, b_name)]
        seen_classes = [n2i[n] for n in seen_names]
        seen_pos = np.where(np.isin(labels[zsl_pos], seen_classes))[0]
        print(f"\n=== 不可见对 {a_name}/{b_name} | seen={seen_names} (n_seen={len(seen_pos)}) ===", flush=True)

        scores_acc = None
        for s in seeds:
            set_seed(s)
            ds = CachedImageDataset(images_all, labels_zsl, feat_zsl, seen_pos)
            train_loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=0, drop_last=True)
            model = build(config, device, attr_t.shape[1], drop_path=args.drop_path,
                          backbone=args.backbone)
            save_dir = os.path.join(HERE, "checkpoints", f"classmean_{a_name}_{b_name}_s{s}")
            print(f"  [seed {s}] training...", flush=True)
            train_seen_classmean(model, train_loader, attr_t, device, args.epochs, save_dir,
                                 lr=args.lr, wd=args.wd)
            model.eval()
            sc = eval_scores(model, images_all, attr_t, device)
            scores_acc = sc if scores_acc is None else scores_acc + sc

        scores = scores_acc / len(seeds)
        sel = (labels_zsl == a) | (labels_zsl == b)
        pred = scores[sel][:, torch.tensor([a, b])].argmax(1)
        true = torch.tensor([0 if int(x) == a else 1 for x in labels_zsl[sel].tolist()])
        acc = float((pred == true).float().mean().item())
        n = int(len(true))
        Pn = F.normalize(attr_t, p=2, dim=-1).cpu()
        dot = float((Pn[a] @ Pn[b]).item())
        rows.append({"pair": f"{a_name}/{b_name}", "attr_cos": round(dot, 3),
                     "unseen_accuracy": round(acc, 4), "n": n})
        print(f"  [{a_name}/{b_name}] 属性余弦={dot:.3f}  unseen 准确率={acc * 100:.2f}% "
              f"(n={n}, seeds={len(seeds)})", flush=True)

    print("\n| 不可见对 | 属性余弦 | unseen 准确率 |")
    print("|----------|----------|---------------|")
    for r in rows:
        print(f"| {r['pair']} | {r['attr_cos']:.3f} | {r['unseen_accuracy'] * 100:.2f}% |")
    mean = float(np.mean([r["unseen_accuracy"] for r in rows]))
    print(f"\n均值准确率（类均值回归） = {mean * 100:.2f}%")

    out = {"note": "ESAG 类均值回归消融（干净 3 seed × 5 对，回归目标=类均值原型）",
           "seeds": seeds, "epochs": args.epochs, "lr": args.lr, "wd": args.wd,
           "drop_path": args.drop_path, "backbone": args.backbone,
           "attr_holdout": ATTR_HOLDOUT, "annot_seed": ANNOT_SEED,
           "attr_dim": int(attr_t.shape[1]), "rows": rows, "mean_accuracy": round(mean, 4)}
    json.dump(out, open(args.results_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n[已另存] {args.results_json}")


if __name__ == "__main__":
    main()
