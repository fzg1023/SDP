"""
CWRU 留故障部位（leave-fault-location-out）更严格对照 —— 第 1 项（可选）。

与「留严重度」不同：这里把一个故障部位（内圈 IR / 滚动体 B / 外圈 OR）**整个**作为 unseen，
训练只用其余 7 类。测试 = 在 unseen 部位的 3 个严重度（007/014/021）之间做 3-way 分类。

三个子实验：分别 hold IR / B / OR。逐 seed 记录 3-way 准确率 + 逐类 recall，accuracy-averaging。
不改模型：small CNN + 逐样本 MSE + 语义点积 + 严格 inductive（20% holdout 标定属性、80% ZSL）。

用法：
  D:/Anaconda/envs/esag/python.exe -u train_cwru_leave_location.py --epochs 50 --seeds 3 --backbone small
结果另存 results_cwru_leave_location.json。
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
LOADS = [0, 1, 2, 3]
SAMPLE_RATE = 12000
SIGNAL_LENGTH = 2048
ATTR_HOLDOUT = 0.2
ANNOT_SEED = 42

# 每个被留出的部位 → 其 3 个严重度类名
LOCATIONS = {
    "IR": ["IR007", "IR014", "IR021"],
    "B":  ["B007", "B014", "B021"],
    "OR": ["OR007", "OR014", "OR021"],
}


def multiway_acc(scores_mat, labels, classes):
    sel = np.isin(labels.numpy(), classes)
    sub = scores_mat[sel][:, classes]
    pred = sub.argmax(1)
    true = torch.tensor([classes.index(int(x)) for x in labels[sel].tolist()])
    return float((pred == true).float().mean().item()), int(sel.sum())


def per_class_recall(scores_mat, labels, classes):
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
    ap.add_argument("--backbone", type=str, default="small")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--hold", type=str, default="", help="IR|B|OR；留空则三个都跑")
    ap.add_argument("--results-json", type=str,
                    default=os.path.join(HERE, "results_cwru_leave_location.json"))
    args = ap.parse_args()

    config = load_config(CFG)
    device = get_device("auto")
    seeds = [42 + i * 7 for i in range(args.seeds)]
    holds = [args.hold] if args.hold else list(LOCATIONS.keys())

    loader = CWRUDataset(data_path=CWRU_DATA, signal_length=SIGNAL_LENGTH,
                         sample_rate=SAMPLE_RATE, loads=LOADS)
    signals, labels, class_indices, class_names = loader.load_data(fault_types=FAULT_TYPES)
    n2i = {n: i for i, n in enumerate(class_names)}
    n_classes = len(class_names)
    print(f"[data] n={len(signals)} classes={class_names}", flush=True)

    feat = np.stack([image_features(s) for s in signals])
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
    attr_t = torch.tensor(attr_raw, dtype=torch.float32, device=device)
    feat_zsl = torch.tensor(Z[zsl_pos], dtype=torch.float32)

    preprocessor = SignalPreprocessor(signal_length=SIGNAL_LENGTH, sample_rate=SAMPLE_RATE,
                                      output_size=64)
    print("[prep] precomputing CWT images...", flush=True)
    images_all = torch.from_numpy(
        np.stack([preprocessor.process_single(signals[i]) for i in zsl_pos]).astype(np.float32))
    labels_zsl = torch.tensor(labels[zsl_pos], dtype=torch.long)
    print(f"[prep] zsl images={tuple(images_all.shape)}", flush=True)

    # 直接匹配上界（无模型，样本真实特征 vs 原型，全 CPU）
    Pn = torch.nn.functional.normalize(attr_t, p=2, dim=-1).cpu()
    scores_raw = torch.nn.functional.normalize(feat_zsl, p=2, dim=-1).cpu() @ Pn.t()

    rows = []
    all3 = []
    for loc in holds:
        unseen_names = LOCATIONS[loc]
        unseen_classes = [n2i[n] for n in unseen_names]
        seen_classes = [i for i in range(n_classes) if i not in unseen_classes]
        seen_pos = np.where(np.isin(labels[zsl_pos], seen_classes))[0]
        print(f"\n=== hold {loc} | seen={[class_names[i] for i in seen_classes]} "
              f"unseen={unseen_names} ===", flush=True)

        acc3_list, rec3_list = [], []
        for s in seeds:
            set_seed(s)
            ds = CachedImageDataset(images_all, labels_zsl, feat_zsl, seen_pos)
            train_loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=0,
                                      drop_last=True)
            model = build(config, device, attr_t.shape[1], drop_path=args.drop_path,
                          backbone=args.backbone)
            train_seen(model, train_loader, attr_t, seen_classes, device, args.epochs,
                       os.path.join(HERE, "checkpoints", f"loc_{loc}_s{s}"),
                       lr=args.lr, wd=args.wd)
            model.eval()
            sc = eval_scores(model, images_all, attr_t, device)
            acc3, n3 = multiway_acc(sc, labels_zsl, unseen_classes)
            rec3 = per_class_recall(sc, labels_zsl, unseen_classes)
            acc3_list.append(acc3)
            rec3_list.append(rec3)
            print(f"  [seed {s}] 3-way={acc3 * 100:.2f}%  rec="
                  f"{ {class_names[k]: round(v*100,1) for k,v in rec3.items()} }", flush=True)

        mean3 = float(np.mean(acc3_list))
        std3 = float(np.std(acc3_list))
        raw3, _ = multiway_acc(scores_raw, labels_zsl, unseen_classes)
        all3.append(mean3)
        rec_mean = {class_names[k]: round(float(np.mean([r.get(k, 0.0) for r in rec3_list])), 4)
                    for k in unseen_classes}
        rows.append({"hold": loc, "unseen": unseen_names,
                     "seen": [class_names[i] for i in seen_classes],
                     "three_way_mean": round(mean3, 4), "three_way_std": round(std3, 4),
                     "three_way_ceiling": round(raw3, 4),
                     "three_way_per_seed": [round(x, 4) for x in acc3_list],
                     "recall_mean": rec_mean})
        print(f"  [{loc}] 3-way = {mean3 * 100:.2f}% ± {std3 * 100:.2f}%  "
              f"(ceiling {raw3 * 100:.2f}%, chance 33.3%)", flush=True)

    grand = float(np.mean(all3)) if all3 else 0.0
    print(f"\n=== 三部位平均 = {grand * 100:.2f}% ===", flush=True)
    out = {"note": "CWRU 留故障部位更严格对照（不改模型；chance=33.3%）",
           "seeds": seeds, "epochs": args.epochs, "backbone": args.backbone,
           "holds": holds, "rows": rows, "mean_over_locations": round(grand, 4)}
    json.dump(out, open(args.results_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n[已另存] {args.results_json}")


if __name__ == "__main__":
    main()
