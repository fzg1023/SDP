"""
ESAG 真零样本修复 CWRU 版 —— 复用 v10 机制（图像域物理特征 + 逐样本 MSE 回归 + 语义点积）。

口径（论文自带 7/3，写死在 train_cwru.py 的标准 split）：
  seen   = Normal, IR007, IR014, B007, B014, OR007, OR014（7 类：正常 + 轻微/中度）
  unseen = IR021, B021, OR021（3 类重度损伤）
  严格 inductive：20% holdout 标定属性，80% ZSL；训练只用 seen；测试 unseen 纯点积；
  不早停、不选种子、不碰 unseen 标签。

报：3-way unseen 准确率 + 逐类 recall + 3 个 unseen 二分类对 + 直接匹配上界（无模型）。

用法：
  D:/Anaconda/envs/esag/python.exe -u train_fix_cwru.py --epochs 50 --seeds 1
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
from esag.datasets import CWRUDataset
from train_fix_v10 import image_features, CachedImageDataset, build, train_seen, eval_scores

CFG = os.path.join(ROOT, "configs", "light_semantic.yaml")   # swin_lite
CWRU_DATA = os.environ.get("SDP_CWRU_DATA", "D:/EsAg/data/cwru")
FAULT_TYPES = ["Normal", "IR007", "IR014", "IR021", "B007", "B014",
               "B021", "OR007", "OR014", "OR021"]
UNSEEN_NAMES = ["IR021", "B021", "OR021"]
LOADS = [0, 1, 2, 3]
SAMPLE_RATE = 12000
SIGNAL_LENGTH = 2048
ATTR_HOLDOUT = 0.2
ANNOT_SEED = 42


def multiway_acc(scores_mat, labels, classes):
    """scores_mat: (N, K) 全类点积分数；classes: 参与分类的类 id 列表。
    返回在 classes 上的 multi-way 准确率与样本数。"""
    sel = np.isin(labels.numpy(), classes)
    sub = scores_mat[sel][:, classes]
    pred = sub.argmax(1)
    true = torch.tensor([classes.index(int(x)) for x in labels[sel].tolist()])
    return float((pred == true).float().mean().item()), int(sel.sum())


def per_class_recall(scores_mat, labels, classes):
    """在 classes 的 multi-way 下，逐类 recall。"""
    sel = np.isin(labels.numpy(), classes)
    sub = scores_mat[sel][:, classes]
    pred = sub.argmax(1)
    out = {}
    for j, c in enumerate(classes):
        mask = labels[sel] == c
        if mask.sum() > 0:
            out[int(c)] = float((pred[mask] == j).float().mean().item())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--drop-path", type=float, default=0.3)
    ap.add_argument("--backbone", type=str, default="swin_lite")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--results-json", type=str,
                    default=os.path.join(HERE, "results_strict_zsl_cwru.json"))
    args = ap.parse_args()

    config = load_config(CFG)
    device = get_device("auto")
    seeds = [42 + i * 7 for i in range(args.seeds)]

    # ---- 加载 CWRU ----
    loader = CWRUDataset(data_path=CWRU_DATA, signal_length=SIGNAL_LENGTH,
                         sample_rate=SAMPLE_RATE, loads=LOADS)
    signals, labels, class_indices, class_names = loader.load_data(fault_types=FAULT_TYPES)
    n2i = {n: i for i, n in enumerate(class_names)}
    n_classes = len(class_names)
    unseen_classes = [n2i[n] for n in UNSEEN_NAMES]
    seen_classes = [i for i in range(n_classes) if i not in unseen_classes]
    print(f"[data] n={len(signals)} classes={class_names}", flush=True)
    print(f"[data] seen={[class_names[i] for i in seen_classes]} "
          f"unseen={[class_names[i] for i in unseen_classes]}", flush=True)

    # ---- 图像域物理特征 ----
    feat = np.stack([image_features(s) for s in signals])  # (N,21)

    # ---- 20% 标定 / 80% ZSL ----
    rng = np.random.RandomState(ANNOT_SEED)
    attr_pos, zsl_pos = [], []
    for c in range(n_classes):
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
    attr_raw = np.stack([Z[attr_pos][labels[attr_pos] == i].mean(0) for i in range(n_classes)])
    attr_t = torch.tensor(attr_raw, dtype=torch.float32, device=device)   # (10,21)
    feat_zsl = torch.tensor(Z[zsl_pos], dtype=torch.float32)              # 逐样本回归目标
    print(f"[attr] {n_classes} 类原型来自 {len(attr_pos)} 个标定样本", flush=True)

    # ---- CWT 图（仅 ZSL 80%）----
    preprocessor = SignalPreprocessor(signal_length=SIGNAL_LENGTH, sample_rate=SAMPLE_RATE,
                                      output_size=64)
    print("[prep] precomputing CWT images for ZSL split...", flush=True)
    images_all = torch.from_numpy(
        np.stack([preprocessor.process_single(signals[i]) for i in zsl_pos]).astype(np.float32))
    labels_zsl = torch.tensor(labels[zsl_pos], dtype=torch.long)
    print(f"[prep] zsl images={images_all.shape}", flush=True)

    # ---- 训练（只用 seen 7 类）----
    seen_pos = np.where(np.isin(labels[zsl_pos], seen_classes))[0]
    scores_acc = None
    for s in seeds:
        set_seed(s)
        ds = CachedImageDataset(images_all, labels_zsl, feat_zsl, seen_pos)
        train_loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=0, drop_last=True)
        model = build(config, device, attr_t.shape[1], drop_path=args.drop_path,
                      backbone=args.backbone)
        save_dir = os.path.join(HERE, "checkpoints", f"cwru_s{s}")
        print(f"[seed {s}] training on {len(seen_pos)} seen samples...", flush=True)
        train_seen(model, train_loader, attr_t, seen_classes, device, args.epochs, save_dir,
                   lr=args.lr, wd=args.wd, ce_weight=0.0, loss_type="mse")
        model.eval()
        sc = eval_scores(model, images_all, attr_t, device)
        scores_acc = sc if scores_acc is None else scores_acc + sc

    scores = scores_acc / len(seeds)  # (N_zsl, 10) 平均分数

    # ---- 直接匹配上界（无模型，样本真实特征 vs 原型，全部 CPU）----
    scores_raw = F.normalize(feat_zsl, p=2, dim=-1).cpu() @ F.normalize(attr_t, p=2, dim=-1).cpu().t()

    # ---- 评估 ----
    Pn = F.normalize(attr_t, p=2, dim=-1).cpu()

    # 3-way unseen
    acc3, n3 = multiway_acc(scores, labels_zsl, unseen_classes)
    raw3, _ = multiway_acc(scores_raw, labels_zsl, unseen_classes)
    rec3 = per_class_recall(scores, labels_zsl, unseen_classes)
    raw_rec3 = per_class_recall(scores_raw, labels_zsl, unseen_classes)

    print("\n=== 3-way unseen（IR021/B021/OR021）===", flush=True)
    print(f"  学习模型 3-way 准确率 = {acc3*100:.2f}% (n={n3})", flush=True)
    print(f"  直接匹配上界           = {raw3*100:.2f}%", flush=True)
    for c in unseen_classes:
        print(f"    {class_names[c]:6s} recall: model={rec3.get(c, 0)*100:5.2f}%  "
              f"ceiling={raw_rec3.get(c, 0)*100:5.2f}%", flush=True)

    # 3 个 unseen 二分类对
    pairs = [(unseen_classes[0], unseen_classes[1]),
             (unseen_classes[0], unseen_classes[2]),
             (unseen_classes[1], unseen_classes[2])]
    rows = []
    print("\n| unseen 对 | 属性余弦 | 模型准确率 | 直接匹配上界 |", flush=True)
    print("|-----------|----------|-----------|--------------|", flush=True)
    for a, b in pairs:
        acc, n = multiway_acc(scores, labels_zsl, [a, b])
        raw, _ = multiway_acc(scores_raw, labels_zsl, [a, b])
        dot = float((Pn[a] @ Pn[b]).item())
        rows.append({"pair": f"{class_names[a]}/{class_names[b]}", "attr_cos": round(dot, 3),
                     "unseen_accuracy": round(acc, 4), "ceiling": round(raw, 4), "n": n})
        print(f"| {class_names[a]}/{class_names[b]} | {dot:.3f} | {acc*100:.2f}% | {raw*100:.2f}% |",
              flush=True)

    out = {
        "note": "ESAG 真零样本修复 CWRU 版（论文自带 7/3：seen 7 类 / unseen 3 重度）",
        "dataset": "CWRU", "seeds": seeds, "epochs": args.epochs, "lr": args.lr, "wd": args.wd,
        "drop_path": args.drop_path, "backbone": args.backbone,
        "attr_holdout": ATTR_HOLDOUT, "annot_seed": ANNOT_SEED, "attr_dim": int(attr_t.shape[1]),
        "seen": [class_names[i] for i in seen_classes],
        "unseen": [class_names[i] for i in unseen_classes],
        "three_way_accuracy": round(acc3, 4), "three_way_ceiling": round(raw3, 4),
        "three_way_recall": {class_names[k]: round(v, 4) for k, v in rec3.items()},
        "three_way_recall_ceiling": {class_names[k]: round(v, 4) for k, v in raw_rec3.items()},
        "pairwise": rows,
    }
    json.dump(out, open(args.results_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n[已另存] {args.results_json}", flush=True)
    print(f"\n=== CWRU 3-way unseen 准确率 = {acc3*100:.2f}%（上界 {raw3*100:.2f}%）===", flush=True)


if __name__ == "__main__":
    main()
