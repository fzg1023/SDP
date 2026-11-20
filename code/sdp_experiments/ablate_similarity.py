"""
语义点积必要性消融（第 4 项）。

同一投影特征 q(x) + 同一类原型 p_c 下，比较 4 种零样本原型匹配度量：
  cosine    归一化点积（本文 argmax normalize(q)·normalize(p_c)）
  raw_dot   未归一化内积 argmax q·p_c
  euclidean L2 距离最近邻 argmin ||q-p_c||_2
  manhattan L1 距离最近邻 argmin sum|q-p_c|

全部零样本（unseen 无训练），端到端 small CNN，BaiduGear 5 对 × 3 seed。
结果另存 results_similarity_ablation.json。
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
from train_fix_v10 import image_features, CachedImageDataset, build, train_seen

CFG = os.path.join(ROOT, "configs", "light_semantic.yaml")
GEAR_DATA = os.environ.get("SDP_GEAR_DATA", "D:/BaiduNetdiskDownload/GearDatasets")
CLASS_NAMES = ["健康", "剥落", "断齿", "磨损", "裂纹"]
ALL_PAIRS = [("健康", "裂纹"), ("剥落", "磨损"), ("断齿", "磨损"),
             ("剥落", "裂纹"), ("剥落", "断齿")]
MAX_FILES = 50
MAX_SEGMENTS = 50
ATTR_HOLDOUT = 0.2
ANNOT_SEED = 42
METRICS = ["cosine", "raw_dot", "euclidean", "manhattan"]


@torch.no_grad()
def q_raw_all(model, images, device, batch_size=256):
    outs = []
    for i in range(0, len(images), batch_size):
        outs.append(model.attr_head(model.backbone(images[i:i + batch_size].to(device))).cpu())
    return torch.cat(outs, 0)


def metric_scores(q, p):
    q = q.double()
    p = p.double().to(q.device)
    qn = F.normalize(q, p=2, dim=-1)
    pn = F.normalize(p, p=2, dim=-1)
    cosine = qn @ pn.t()
    raw_dot = q @ p.t()
    pnorm2 = (p ** 2).sum(-1)
    euclidean = 2 * raw_dot - pnorm2.unsqueeze(0)   # argmax == argmin ||q-p||
    manhattan = -torch.cdist(q, p, p=1.0)
    return {"cosine": cosine, "raw_dot": raw_dot, "euclidean": euclidean,
            "manhattan": manhattan}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--backbone", type=str, default="small")
    ap.add_argument("--results-json", type=str,
                    default=os.path.join(HERE, "results_similarity_ablation.json"))
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

    preprocessor = SignalPreprocessor(signal_length=2048, sample_rate=15360, output_size=64)
    print("[prep] precomputing CWT images...", flush=True)
    images_all = torch.from_numpy(
        np.stack([preprocessor.process_single(signals[i]) for i in zsl_pos]).astype(np.float32))
    labels_zsl = torch.tensor(labels[zsl_pos], dtype=torch.long)
    print(f"[prep] zsl images={tuple(images_all.shape)}", flush=True)

    # 每个 metric 记录每个 pair 的 3-seed 均值
    acc_by_metric = {m: [] for m in METRICS}
    rows = []

    for a_name, b_name in ALL_PAIRS:
        a, b = n2i[a_name], n2i[b_name]
        seen_names = [n for n in CLASS_NAMES if n not in (a_name, b_name)]
        seen_classes = [n2i[n] for n in seen_names]
        seen_pos = np.where(np.isin(labels[zsl_pos], seen_classes))[0]
        sel = (labels_zsl == a) | (labels_zsl == b)
        true = torch.tensor([0 if int(x) == a else 1 for x in labels_zsl[sel].tolist()])

        acc_per_metric_seed = {m: [] for m in METRICS}
        for s in seeds:
            set_seed(s)
            ds = CachedImageDataset(images_all, labels_zsl, feat_zsl, seen_pos)
            train_loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=0,
                                      drop_last=True)
            model = build(config, device, attr_t.shape[1], backbone=args.backbone)
            train_seen(model, train_loader, attr_t, seen_classes, device, args.epochs,
                       os.path.join(HERE, "checkpoints", f"sim_{a_name}_{b_name}_s{s}"))
            model.eval()
            q_raw = q_raw_all(model, images_all, device)
            sc = metric_scores(q_raw, attr_t)
            for m in METRICS:
                sub = sc[m][sel][:, torch.tensor([a, b])]
                pred = sub.argmax(1)
                acc_per_metric_seed[m].append(float((pred == true).float().mean().item()))

        row = {"pair": f"{a_name}/{b_name}"}
        line = f"  [{a_name}/{b_name}] "
        for m in METRICS:
            mean_m = float(np.mean(acc_per_metric_seed[m]))
            row[m] = round(mean_m, 4)
            acc_by_metric[m].append(mean_m)
            line += f"{m}={mean_m * 100:.2f}% "
        rows.append(row)
        print(line, flush=True)

    print("\n=== 5 对平均（各度量）===", flush=True)
    summary = {}
    for m in METRICS:
        mean = float(np.mean(acc_by_metric[m]))
        summary[m] = round(mean, 4)
        print(f"  {m}: {mean * 100:.2f}%", flush=True)

    out = {"note": "语义点积必要性消融：同一投影特征/原型下 4 种零样本匹配度量，5 对 × 3 seed",
           "seeds": seeds, "epochs": args.epochs, "backbone": args.backbone,
           "metrics": METRICS, "rows": rows, "mean": summary}
    json.dump(out, open(args.results_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n[已另存] {args.results_json}")


if __name__ == "__main__":
    main()
