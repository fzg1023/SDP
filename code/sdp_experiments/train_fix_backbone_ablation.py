"""
ESAG 骨干消融：证明 Swin 骨干的必要性（干净协议，唯一变量 = 骨干网络）。

协议与 train_fix_v10.py 完全一致：
  21 维图像域物理特征 + 逐样本 MSE 回归 + 语义点积 + 严格 inductive
  （20% 标定 / 80% ZSL，5 对 × 3 种子 × 50 轮，无早停、无分类头选择）。

对比骨干（ESAG 内置，只改 backbone_type，其余全同）：
  swin_lite : 轻量 Swin Transformer（当前主结果，~10.72M）
  swin      : Swin-Tiny 全尺寸（~27.6M）
  medium    : ResNet-18（~11.6M）
  small     : 轻量 CNN（~2M）

用法：
  D:/Anaconda/envs/esag/python.exe -u train_fix_backbone_ablation.py --seeds 3
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
from train_fix_v10 import (image_features, CachedImageDataset, build,
                           train_seen, eval_scores)

CFG = os.path.join(ROOT, "configs", "light_semantic.yaml")
GEAR_DATA = os.environ.get("SDP_GEAR_DATA", "D:/BaiduNetdiskDownload/GearDatasets")
CLASS_NAMES = ["健康", "剥落", "断齿", "磨损", "裂纹"]
ALL_PAIRS = [("健康", "裂纹"), ("剥落", "磨损"), ("断齿", "磨损"),
             ("剥落", "裂纹"), ("剥落", "断齿")]
MAX_FILES = 50
MAX_SEGMENTS = 50
ATTR_HOLDOUT = 0.2
ANNOT_SEED = 42
DEFAULT_BACKBONES = ["swin_lite", "swin", "medium", "small"]


def run_backbone(backbone, config_base, device, seeds, epochs, drop_path,
                 signals, labels, class_indices, class_names, attr_t,
                 feat_zsl, images_all, labels_zsl, n2i, save_dir):
    """对单个骨干跑 5 对 × 3 种子，返回 rows + mean。"""
    rows = []
    for a_name, b_name in ALL_PAIRS:
        a, b = n2i[a_name], n2i[b_name]
        seen_names = [n for n in CLASS_NAMES if n not in (a_name, b_name)]
        seen_classes = [n2i[n] for n in seen_names]
        seen_pos = np.where(np.isin(labels_zsl, seen_classes))[0]
        print(f"\n=== [{backbone}] 不可见对 {a_name}/{b_name} | seen={seen_names} "
              f"(n_seen={len(seen_pos)}) ===", flush=True)

        scores_acc = None
        for s in seeds:
            set_seed(s)
            config = load_config(CFG)  # 每次 fresh，避免 build 的 config 变异串扰
            ds = CachedImageDataset(images_all, labels_zsl, feat_zsl, seen_pos)
            train_loader = DataLoader(ds, batch_size=64, shuffle=True,
                                      num_workers=0, drop_last=True)
            model = build(config, device, attr_t.shape[1], drop_path=drop_path,
                          backbone=backbone)
            train_seen(model, train_loader, attr_t, seen_classes, device, epochs,
                       save_dir, lr=5e-4, wd=1e-3, ce_weight=0.0, loss_type="mse")
            model.eval()
            sc = eval_scores(model, images_all, attr_t, device)
            scores_acc = sc if scores_acc is None else scores_acc + sc

        scores = scores_acc / len(seeds)
        sel = (labels_zsl == a) | (labels_zsl == b)
        pred = scores[sel][:, torch.tensor([a, b])].argmax(1)
        true = torch.tensor([0 if int(x) == a else 1
                             for x in labels_zsl[sel].tolist()])
        acc = float((pred == true).float().mean().item())
        n = int(len(true))
        Pn = F.normalize(attr_t, p=2, dim=-1).cpu()
        dot = float((Pn[a] @ Pn[b]).item())
        rows.append({"pair": f"{a_name}/{b_name}", "attr_cos": round(dot, 3),
                     "unseen_accuracy": round(acc, 4), "n": n})
        print(f"  [{a_name}/{b_name}] 属性余弦={dot:.3f}  "
              f"unseen 准确率={acc * 100:.2f}% (n={n}, seeds={len(seeds)})", flush=True)

    mean = float(np.mean([r["unseen_accuracy"] for r in rows]))
    print(f"\n[{backbone}] 均值准确率 = {mean * 100:.2f}%", flush=True)
    return rows, mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", type=str, default=",".join(DEFAULT_BACKBONES))
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--drop-path", type=float, default=0.3)
    ap.add_argument("--results-json", type=str,
                    default=os.path.join(HERE, "results_backbone_ablation.json"))
    args = ap.parse_args()

    backbones = [b.strip() for b in args.backbones.split(",") if b.strip()]
    seeds = [42 + i * 7 for i in range(args.seeds)]

    config_base = load_config(CFG)
    device = get_device("auto")
    print(f"[env] device={device} backbones={backbones} seeds={seeds} "
          f"epochs={args.epochs}", flush=True)

    # ---- 数据 + 特征 + 原型（只算一次，与 v10 完全一致）----
    loader = BaiduGearDataset(data_path=GEAR_DATA, signal_length=2048,
                              max_files_per_class=MAX_FILES,
                              max_segments_per_file=MAX_SEGMENTS, vibration_col=0)
    signals, labels, class_indices, class_names = loader.load_data()
    n2i = {str(n): i for i, n in enumerate(class_names)}
    print(f"[data] n={len(signals)} classes={class_names}", flush=True)

    feat = np.stack([image_features(s) for s in signals])  # (N,21)

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

    preprocessor = SignalPreprocessor(signal_length=2048, sample_rate=15360,
                                      output_size=64)
    print("[prep] precomputing CWT images for ZSL split...", flush=True)
    images_all = torch.from_numpy(
        np.stack([preprocessor.process_single(signals[i])
                  for i in zsl_pos]).astype(np.float32))
    labels_zsl = torch.tensor(labels[zsl_pos], dtype=torch.long)
    print(f"[prep] zsl images={images_all.shape}", flush=True)

    # ---- 逐骨干跑 ----
    summary = {}
    for backbone in backbones:
        save_dir = os.path.join(HERE, "checkpoints", f"ablation_{backbone}")
        rows, mean = run_backbone(
            backbone, config_base, device, seeds, args.epochs, args.drop_path,
            signals, labels, class_indices, class_names, attr_t, feat_zsl,
            images_all, labels_zsl, n2i, save_dir)
        summary[backbone] = {"mean_accuracy": round(mean, 4), "rows": rows}

    # ---- 汇总对比表 ----
    print("\n\n========== BACKBONE 消融汇总 ==========")
    print(f"{'backbone':<12} {'mean unseen acc':>16}")
    print("-" * 30)
    for b in backbones:
        print(f"{b:<12} {summary[b]['mean_accuracy'] * 100:>15.2f}%")
    print("=" * 36)

    out = {
        "note": "骨干消融：唯一变量 = backbone（21维图像域特征+逐样本MSE+语义点积+严格inductive）",
        "seeds": seeds, "epochs": args.epochs, "drop_path": args.drop_path,
        "attr_holdout": ATTR_HOLDOUT, "annot_seed": ANNOT_SEED,
        "attr_dim": int(attr_t.shape[1]), "backbones": backbones,
        "summary": summary,
    }
    json.dump(out, open(args.results_json, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n[已保存] {args.results_json}")


if __name__ == "__main__":
    main()
