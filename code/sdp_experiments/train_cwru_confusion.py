"""
P0-2：CWRU 三分类混淆矩阵（5 seed 汇总）—— 供重画图 8。

复用 train_cwru_multiseed.py 的 5-seed 协议（small CNN, 50 epoch, 逐样本 MSE），
额外记录每个 seed 在 unseen 三类上的逐样本预测，汇总成 5 seed 混淆矩阵：
  - counts：真实类 -> 预测类 的样本计数（5 seed 合计）
  - 也输出逐类 recall 与总体 3-way（与 results_cwru_multiseed.json 对齐）

结果另存 results_cwru_confusion.json + cwru_confusion.npz。
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
from esag.datasets import CWRUDataset
from train_fix_v10 import image_features, CachedImageDataset, build, train_seen, eval_scores

CFG = os.path.join(ROOT, "configs", "light_semantic.yaml")
CWRU_DATA = os.environ.get("SDP_CWRU_DATA", "D:/EsAg/data/cwru")
FAULT_TYPES = ["Normal", "IR007", "IR014", "IR021", "B007", "B014",
               "B021", "OR007", "OR014", "OR021"]
UNSEEN_NAMES = ["IR021", "B021", "OR021"]
LOADS = [0, 1, 2, 3]
SAMPLE_RATE = 12000
SIGNAL_LENGTH = 2048
ATTR_HOLDOUT = 0.2
ANNOT_SEED = 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--backbone", type=str, default="small")
    ap.add_argument("--results-json", type=str,
                    default=os.path.join(HERE, "results_cwru_confusion.json"))
    args = ap.parse_args()

    config = load_config(CFG)
    device = get_device("auto")
    seeds = [42 + i * 7 for i in range(args.seeds)]

    loader = CWRUDataset(data_path=CWRU_DATA, signal_length=SIGNAL_LENGTH,
                         sample_rate=SAMPLE_RATE, loads=LOADS)
    signals, labels, class_indices, class_names = loader.load_data(fault_types=FAULT_TYPES)
    n2i = {n: i for i, n in enumerate(class_names)}
    unseen_classes = [n2i[n] for n in UNSEEN_NAMES]
    seen_classes = [i for i in range(len(class_names)) if i not in unseen_classes]
    print(f"[data] n={len(signals)} unseen={UNSEEN_NAMES}", flush=True)

    feat = np.stack([image_features(s) for s in signals])
    rng = np.random.RandomState(ANNOT_SEED)
    attr_pos, zsl_pos = [], []
    for c in range(len(class_names)):
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
    attr_raw = np.stack([Z[attr_pos][labels[attr_pos] == i].mean(0) for i in range(len(class_names))])
    attr_t = torch.tensor(attr_raw, dtype=torch.float32, device=device)
    feat_zsl = torch.tensor(Z[zsl_pos], dtype=torch.float32)

    preprocessor = SignalPreprocessor(signal_length=SIGNAL_LENGTH, sample_rate=SAMPLE_RATE,
                                      output_size=64)
    print("[prep] precomputing CWT images...", flush=True)
    images_all = torch.from_numpy(
        np.stack([preprocessor.process_single(signals[i]) for i in zsl_pos]).astype(np.float32))
    labels_zsl = torch.tensor(labels[zsl_pos], dtype=torch.long)
    print(f"[prep] zsl images={tuple(images_all.shape)}", flush=True)

    seen_pos = np.where(np.isin(labels[zsl_pos], seen_classes))[0]

    # 累计混淆矩阵：真类(0/1/2) -> 预测类(0/1/2)
    conf = np.zeros((3, 3), dtype=np.int64)
    per_seed = []
    for s in seeds:
        set_seed(s)
        ds = CachedImageDataset(images_all, labels_zsl, feat_zsl, seen_pos)
        train_loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=0, drop_last=True)
        model = build(config, device, attr_t.shape[1], drop_path=0.3, backbone=args.backbone)
        train_seen(model, train_loader, attr_t, seen_classes, device, args.epochs,
                   os.path.join(HERE, "checkpoints", f"cwru_confusion_s{s}"),
                   lr=5e-4, wd=1e-3)
        model.eval()
        sc = eval_scores(model, images_all, attr_t, device)
        sel = np.isin(labels_zsl.numpy(), unseen_classes)
        sub = sc[sel][:, torch.tensor(unseen_classes)]
        pred = sub.argmax(1).numpy()
        true = torch.tensor([unseen_classes.index(int(x)) for x in labels_zsl[sel].tolist()]).numpy()
        c = np.zeros((3, 3), dtype=np.int64)
        for t, p in zip(true, pred):
            c[int(t), int(p)] += 1
        conf += c
        acc = float(np.diag(c).sum() / c.sum())
        per_seed.append({"seed": s, "conf": c.tolist(), "acc": round(acc, 4)})
        print(f"[seed {s}] 3-way={acc * 100:.2f}%  conf(row=真类)=\n{c}", flush=True)

    total = int(conf.sum())
    acc_all = float(np.diag(conf).sum() / total)
    recall = [float(conf[i, i] / conf[i].sum()) for i in range(3)]
    # 错误流向（每类被误判到哪个类，百分比）
    flow = {}
    for i, name in enumerate(UNSEEN_NAMES):
        row = conf[i]
        off = {UNSEEN_NAMES[j]: round(float(row[j] / row.sum()), 4) for j in range(3) if j != i}
        flow[name] = {"recall": round(recall[i], 4), "misclassified_to": off}

    print(f"\n=== CWRU 5-seed 混淆矩阵（合计 {total} 样本）===")
    print("        预测: IR021  B021  OR021")
    for i, name in enumerate(UNSEEN_NAMES):
        print(f"  真 {name}: " + "  ".join(f"{conf[i,j]:6d}" for j in range(3)))
    print(f"总体 3-way = {acc_all * 100:.2f}%")
    for i, name in enumerate(UNSEEN_NAMES):
        print(f"  {name} recall = {recall[i] * 100:.2f}%")

    out = {"note": "CWRU 三分类混淆矩阵（5 seed 汇总，small CNN）", "seeds": seeds,
           "epochs": args.epochs, "backbone": args.backbone, "total": total,
           "confusion_matrix": conf.tolist(), "three_way_acc": round(acc_all, 4),
           "recall": {UNSEEN_NAMES[i]: round(recall[i], 4) for i in range(3)},
           "flow": flow, "per_seed": per_seed}
    json.dump(out, open(args.results_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    np.savez(os.path.join(HERE, "cwru_confusion.npz"), conf=conf, seeds=np.array(seeds))
    print(f"\n[已另存] {args.results_json} + cwru_confusion.npz")


if __name__ == "__main__":
    main()
