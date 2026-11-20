"""
近年 ZSL 故障诊断方法复现（齿轮 5 对，冻结骨干协议，与 baseline_compare.py 完全一致）。

复现方法（忠实实现其核心机制，跑在我们的统一 inductive 协议下）：
  - VAE  (Oh & Jung, MSSP 2026)  类感知变分自编码器语义提取：冻结特征 -> VAE(重建+KL+隐层对齐属性)
        -> 隐变量余弦到类原型。
  - GCN  (J. Wang et al., S&A:A 2026) DACNN+GCN 属性学习：冻结特征 -> 样本 k-NN 图上 2 层 GCN(残差)
        -> 属性回归 -> 余弦分类。
  - DAP  (锚点，闭式岭回归) —— 用于核对冻结特征与既有 baseline_compare 结果一致（应≈66.44%）。

协议（严格 inductive，与 baseline_compare 相同）：
  每对 unseen (a,b)，seen=其余 3 类；在 seen 上 CE 预训练 small backbone(50ep) 冻结提取特征；
  属性原型 attr_t(21维，20%标定集类均值) 所有方法共享；不早停、不碰 unseen 标签、固定 3 seed。
结果另存 results_baseline_recent.json。
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "demo"))
sys.path.insert(0, HERE)

from esag.utils import load_config, set_seed, get_device
from esag.preprocessing import SignalPreprocessor
from demo.train_gear_baidu import BaiduGearDataset
from train_fix_v10 import image_features, CachedImageDataset, build
from baseline_compare import (train_classifier_backbone, extract_features,
                              normalize, run_dap)

CFG = os.path.join(ROOT, "configs", "light_semantic.yaml")
GEAR_DATA = os.environ.get("SDP_GEAR_DATA", "D:/BaiduNetdiskDownload/GearDatasets")
CLASS_NAMES = ["健康", "剥落", "断齿", "磨损", "裂纹"]
ALL_PAIRS = [("健康", "裂纹"), ("剥落", "磨损"), ("断齿", "磨损"),
             ("剥落", "裂纹"), ("剥落", "断齿")]
MAX_FILES = 50
MAX_SEGMENTS = 50
ATTR_HOLDOUT = 0.2
ANNOT_SEED = 42
BACKBONE_EPOCHS = 50


def run_vae(X_train, A_train, X_test, attr_t, latent=21, hidden=256, steps=400,
            lr=1e-3, seed=42, beta=1e-3, lam_align=1.0):
    """类感知 VAE 语义提取（Oh & Jung 2026 核心）：VAE 隐层对齐属性 + 重建 + KL。"""
    set_seed(seed)
    d, a = X_train.shape[1], A_train.shape[1]
    dev = X_train.device
    enc = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU()).to(dev)
    mu_head = nn.Linear(hidden, latent).to(dev)
    lv_head = nn.Linear(hidden, latent).to(dev)
    dec = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU(), nn.Linear(hidden, hidden),
                        nn.ReLU(), nn.Linear(hidden, d)).to(dev)
    params = list(enc.parameters()) + list(mu_head.parameters()) + \
        list(lv_head.parameters()) + list(dec.parameters())
    opt = AdamW(params, lr=lr, weight_decay=1e-4)
    for _ in range(steps):
        h = enc(X_train)
        mu, lv = mu_head(h), lv_head(h)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
        recon = dec(z)
        loss_recon = F.mse_loss(recon, X_train)
        loss_kl = -0.5 * (1 + lv - mu.pow(2) - lv.exp()).sum(1).mean()
        loss_align = F.mse_loss(z, A_train)          # 类感知：隐变量贴近样本自身属性
        loss = loss_recon + beta * loss_kl + lam_align * loss_align
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        mu_test = mu_head(enc(X_test))
        return normalize(mu_test) @ normalize(attr_t).t()


def run_gcn(X_train, A_train, X_test, attr_t, hidden=256, k=10, steps=400,
            lr=1e-3, seed=42):
    """DACNN+GCN 属性学习（J. Wang 2026 核心）：样本 k-NN 图 2 层 GCN(残差) -> 属性回归。"""
    set_seed(seed)
    d, a = X_train.shape[1], A_train.shape[1]
    n = X_train.shape[0]
    sim = X_train @ X_train.t()
    knn = sim.topk(min(k + 1, n), dim=1).indices[:, 1:]           # 排除自身
    adj = torch.zeros_like(sim)
    adj.scatter_(1, knn, 1.0)
    adj = adj + torch.eye(n, device=X_train.device)                # 自环
    deg = adj.sum(1).clamp(min=1).pow(-0.5)
    A_hat = deg[:, None] * adj * deg[None, :]                      # 对称归一化
    W0 = nn.Linear(d, hidden).to(X_train.device)
    W1 = nn.Linear(hidden, a).to(X_train.device)
    W_res = nn.Linear(d, a).to(X_train.device)                     # 输入->输出跳连
    opt = AdamW(list(W0.parameters()) + list(W1.parameters()) + list(W_res.parameters()),
                lr=lr, weight_decay=1e-3)
    for _ in range(steps):
        H = F.relu(A_hat @ W0(X_train))
        Z = A_hat @ W1(H) + W_res(X_train)                         # 残差连接
        loss = F.mse_loss(Z, A_train)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        # inductive：测试样本无图，用自环（等价于把学习到的变换直接作用到 X_test）
        H = F.relu(W0(X_test))
        Z = W1(H) + W_res(X_test)
        return normalize(Z) @ normalize(attr_t).t()


METHODS = ["DAP", "VAE", "GCN"]
RUNNERS = {"DAP": run_dap, "VAE": run_vae, "GCN": run_gcn}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=str, default=None)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--backbone-epochs", type=int, default=BACKBONE_EPOCHS)
    ap.add_argument("--backbone", type=str, default="small")
    ap.add_argument("--results-json", type=str,
                    default=os.path.join(HERE, "results_baseline_recent.json"))
    args = ap.parse_args()

    if args.pairs:
        names = [p.strip() for p in args.pairs.split(",") if p.strip()]
        pairs = [tuple(names[i:i + 2]) for i in range(0, len(names), 2)]
    else:
        pairs = ALL_PAIRS

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
    print("[prep] precomputing CWT images for ZSL split...", flush=True)
    images_all = torch.from_numpy(
        np.stack([preprocessor.process_single(signals[i]) for i in zsl_pos]).astype(np.float32))
    labels_zsl = torch.tensor(labels[zsl_pos], dtype=torch.long)
    print(f"[prep] zsl images={images_all.shape}", flush=True)

    all_rows = []
    for a_name, b_name in pairs:
        a, b = n2i[a_name], n2i[b_name]
        seen_names = [n for n in CLASS_NAMES if n not in (a_name, b_name)]
        seen_classes = [n2i[n] for n in seen_names]
        seen_pos = np.where(np.isin(labels[zsl_pos], seen_classes))[0]
        sel = (labels_zsl == a) | (labels_zsl == b)
        true = torch.tensor([0 if int(x) == a else 1 for x in labels_zsl[sel].tolist()],
                            device=device)
        print(f"\n=== {a_name}/{b_name} | seen={seen_names} ===", flush=True)

        method_acc = {m: [] for m in METHODS}
        for s in seeds:
            print(f"  [seed {s}] training classifier backbone...", flush=True)
            backbone = train_classifier_backbone(config, device, images_all, labels_zsl,
                                                 seen_pos, s, epochs=args.backbone_epochs,
                                                 backbone=args.backbone)
            X_all = extract_features(backbone, images_all, device).to(device)
            X_all = F.normalize(X_all, p=2, dim=-1)
            X_train = X_all[torch.tensor(seen_pos)]
            X_test = X_all[sel]
            A_train = feat_zsl[torch.tensor(seen_pos)].to(device)

            for m in METHODS:
                res = RUNNERS[m](X_train, A_train, X_test, attr_t)
                acc = float((res[:, torch.tensor([a, b])].argmax(1) == true).float().mean().item())
                method_acc[m].append(acc)

        row = {"pair": f"{a_name}/{b_name}", "seen": seen_names, "n": int(len(true))}
        line = f"  [{a_name}/{b_name}] | "
        for m in METHODS:
            mean_acc = float(np.mean(method_acc[m]))
            row[m] = round(mean_acc, 4)
            line += f"{m}={mean_acc * 100:.2f}% "
        all_rows.append(row)
        print(line, flush=True)

    print("\n=== 均值（5 对平均）===", flush=True)
    summary = {}
    for m in METHODS:
        summary[m] = round(float(np.mean([r[m] for r in all_rows])), 4)
        print(f"  {m}: {summary[m] * 100:.2f}%", flush=True)

    out = {"note": "近年 ZSL 方法复现（冻结 small backbone，严格 inductive，3 seed × 5 对）",
           "seeds": seeds, "backbone": args.backbone, "backbone_epochs": args.backbone_epochs,
           "methods": METHODS, "rows": all_rows, "mean": summary}
    json.dump(out, open(args.results_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n[已另存] {args.results_json}")


if __name__ == "__main__":
    main()
