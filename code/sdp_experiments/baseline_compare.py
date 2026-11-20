"""
baseline 公平对比 —— 共享冻结 backbone 协议。

协议（严格 inductive，与 v10 主实验一致）：
  1) 每对 unseen (a,b)，seen = 其余 3 类。
  2) 在 seen 3 类上【分类预训练】Swin-Lite backbone（CE，50 epoch，3 seed），冻结。
  3) 用冻结 backbone 提取全部 ZSL 样本的 512 维特征 X。
  4) 属性原型 attr_t（21 维，数据派生，来自 20% 标注集类均值）所有方法共享。
  5) 每个方法在 seen 样本上学习「视觉特征 → 属性空间」的映射，unseen 用语义点积测试。

方法（忠实实现标准算法）：
  - DAP   独立属性岭回归（逐样本属性），点积最近邻。
  - SAE   语义自编码器（逐样本属性编码 + 重建），点积。
  - ESZSL 兼容函数 + one-hot 标签（类级属性），闭式解。
  - ALE   属性标签嵌入 rank loss（类级属性），SGD。
  - SJE   结构化 max-margin（类级属性），SGD。
  - Ours  非线性 MLP projector 逐样本属性回归（MSE），点积。

不做早停、不碰 unseen 标签、固定 seed。结果另存 results_baseline_compare.json。
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np
import torch
import torch.nn as nn
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
from train_fix_v10 import image_features, CachedImageDataset, build

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


# ---------------------------------------------------------------- backbone 分类预训练
def train_classifier_backbone(config, device, images_all, labels_zsl, seen_pos, seed,
                              epochs=BACKBONE_EPOCHS, lr=5e-4, wd=1e-3,
                              backbone="swin_lite"):
    set_seed(seed)
    seen_classes = sorted(set(labels_zsl[seen_pos].tolist()))
    n_cls = len(seen_classes)
    cls_map = {int(c): i for i, c in enumerate(seen_classes)}
    # 用 attr_head 当分类头：Linear(feature_dim, n_cls)
    model = build(config, device, n_cls, drop_path=0.3, backbone=backbone)
    dummy_feats = torch.zeros(len(labels_zsl), 1)
    ds = CachedImageDataset(images_all, labels_zsl, dummy_feats, seen_pos)
    loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=0, drop_last=True)
    params = list(model.backbone.parameters()) + list(model.attr_head.parameters())
    optimizer = AdamW(params, lr=lr, weight_decay=wd)
    warmup = 5
    scheduler = SequentialLR(
        optimizer,
        schedulers=[LinearLR(optimizer, start_factor=0.1, total_iters=warmup),
                    CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup), eta_min=1e-6)],
        milestones=[warmup])
    for ep in range(1, epochs + 1):
        model.train()
        tot = 0.0
        nb = 0
        for batch in loader:
            img = batch["image"].to(device)
            lab = batch["label"].to(device)
            logits = model.attr_head(model.backbone(img))
            y = torch.tensor([cls_map[int(l)] for l in lab.cpu().tolist()],
                             device=device, dtype=torch.long)
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            tot += loss.item()
            nb += 1
        scheduler.step()
    model.eval()
    return model.backbone


@torch.no_grad()
def extract_features(backbone, images_all, device, batch_size=256):
    outs = []
    for i in range(0, len(images_all), batch_size):
        f = backbone(images_all[i:i + batch_size].to(device))
        outs.append(f.cpu())
    return torch.cat(outs, 0)  # (N, 512)


# ---------------------------------------------------------------- baseline 方法
def normalize(t):
    return F.normalize(t, p=2, dim=-1)


def pairwise_acc(scores_unseen, true):
    pred = scores_unseen.argmax(1)
    return float((pred == true).float().mean().item())


@torch.no_grad()
def test_dotprod(score_fn, X_test, attr_t, a, b):
    """统一测试：unseen 两类的点积分数 + 准确率。"""
    scores = score_fn(X_test)          # (N_test, 5)
    sel = [a, b]
    P = normalize(attr_t)              # (5,21)
    # score_fn 已返回 (N,5) 或按需算
    sc = scores[:, torch.tensor(sel)]
    pred = sc.argmax(1)
    return pred


def run_dap(X_train, A_train, X_test, attr_t, lam=1e-3):
    """独立属性岭回归（逐样本属性）→ 点积。"""
    n, d = X_train.shape
    I = torch.eye(d, device=X_train.device) * lam
    W = torch.linalg.solve(X_train.t() @ X_train + I, X_train.t() @ A_train)  # (512,21)
    A_hat = X_test @ W
    P = normalize(attr_t)
    return normalize(A_hat) @ P.t()


def run_eszsl(X_train, Y_train, X_test, A_seen, attr_full, gamma=1e3, lam=1e-3):
    """ESZSL 兼容函数（类级属性 + one-hot），闭式解。

    A_seen: seen 类原型 (k,21)，与 Y_train 列顺序一致；attr_full: 完整 5 类原型（测试用）。
    """
    n, d = X_train.shape
    # W = (XᵀX + γI)⁻¹ Xᵀ Y A (AᵀA + λI)⁻¹
    G = torch.linalg.inv(X_train.t() @ X_train + gamma * torch.eye(d, device=X_train.device))
    H = torch.linalg.inv(A_seen.t() @ A_seen + lam * torch.eye(A_seen.shape[1], device=X_train.device))
    W = G @ X_train.t() @ Y_train @ A_seen @ H       # (512,21)
    P = normalize(attr_full)
    return normalize(X_test @ W) @ P.t()


def run_sae(X_train, A_train, X_test, attr_t, lam=1e-2, steps=300, lr=1e-3, seed=42):
    """语义自编码器（逐样本属性编码 + 重建），梯度下降。"""
    set_seed(seed)
    n, d = X_train.shape
    a = A_train.shape[1]
    W = torch.randn(a, d, device=X_train.device) * 0.01   # 编码器 (21,512)
    W.requires_grad_(True)
    opt = AdamW([W], lr=lr, weight_decay=1e-4)
    for _ in range(steps):
        code = X_train @ W.t()          # (n,21)
        recon = code @ W                # (n,512)
        loss = F.mse_loss(recon, X_train) + lam * F.mse_loss(code, A_train)
        opt.zero_grad()
        loss.backward()
        opt.step()
    code_test = X_test @ W.t()
    P = normalize(attr_t)
    return normalize(code_test) @ P.t()


def run_ale(X_train, Y_train, X_test, A_seen, attr_full, margin=0.2, steps=300, lr=1e-3, seed=42):
    """属性标签嵌入 rank loss（类级属性），SGD。

    A_seen: seen 类原型 (k,21) 用于训练；attr_full: 完整 5 类原型用于测试。
    """
    set_seed(seed)
    n, d = X_train.shape
    a = A_seen.shape[1]
    W = torch.randn(d, a, device=X_train.device) * 0.01
    W.requires_grad_(True)
    opt = AdamW([W], lr=lr, weight_decay=1e-4)
    P_seen = normalize(A_seen)
    labels = Y_train.argmax(1)
    for _ in range(steps):
        S = X_train @ W @ P_seen.t()         # (n,k) score
        sy = S.gather(1, labels.unsqueeze(1))   # (n,1)
        margin_loss = torch.clamp(margin - sy + S, min=0)  # (n,k)
        mask = torch.ones_like(S, dtype=torch.bool)
        mask.scatter_(1, labels.unsqueeze(1), False)
        loss = margin_loss[mask].mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return normalize(X_test @ W) @ normalize(attr_full).t()


def run_sje(X_train, Y_train, X_test, A_seen, attr_full, margin=0.2, steps=300, lr=1e-3, seed=42):
    """结构化 max-margin（类级属性），SGD。

    A_seen: seen 类原型 (k,21) 用于训练；attr_full: 完整 5 类原型用于测试。
    """
    set_seed(seed)
    n, d = X_train.shape
    a = A_seen.shape[1]
    W = torch.randn(d, a, device=X_train.device) * 0.01
    W.requires_grad_(True)
    opt = AdamW([W], lr=lr, weight_decay=1e-4)
    P_seen = normalize(A_seen)
    labels = Y_train.argmax(1)
    for _ in range(steps):
        S = X_train @ W @ P_seen.t()          # (n,k)
        sy = S.gather(1, labels.unsqueeze(1))
        margin_loss = torch.clamp(margin - sy + S, min=0)
        mask = torch.ones_like(S, dtype=torch.bool)
        mask.scatter_(1, labels.unsqueeze(1), False)
        margin_loss[~mask] = 0.0
        loss = margin_loss.max(1).values.mean()   # max over wrong classes
        opt.zero_grad()
        loss.backward()
        opt.step()
    return normalize(X_test @ W) @ normalize(attr_full).t()


def run_ours(X_train, A_train, X_test, attr_t, steps=300, lr=1e-3, seed=42, hidden=256):
    """Ours（公平版）：非线性 MLP projector 逐样本属性回归（MSE）→ 点积。"""
    set_seed(seed)
    n, d = X_train.shape
    a = A_train.shape[1]
    mlp = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, a)).to(X_train.device)
    opt = AdamW(mlp.parameters(), lr=lr, weight_decay=1e-3)
    for _ in range(steps):
        loss = F.mse_loss(mlp(X_train), A_train)
        opt.zero_grad()
        loss.backward()
        opt.step()
    P = normalize(attr_t)
    return normalize(mlp(X_test)) @ P.t()


METHODS = {
    "DAP":   ("class", "linear"),
    "SAE":   ("sample", "linear"),
    "ESZSL": ("class", "linear"),
    "ALE":   ("class", "linear"),
    "SJE":   ("class", "linear"),
    "Ours":  ("sample", "nonlinear"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=str, default=None)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--backbone-epochs", type=int, default=BACKBONE_EPOCHS)
    ap.add_argument("--backbone", type=str, default="swin_lite")
    ap.add_argument("--results-json", type=str,
                    default=os.path.join(HERE, "results_baseline_compare.json"))
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
    attr_t = torch.tensor(attr_raw, dtype=torch.float32, device=device)   # (5,21)
    feat_zsl = torch.tensor(Z[zsl_pos], dtype=torch.float32)              # 逐样本属性 (N,21)

    preprocessor = SignalPreprocessor(signal_length=2048, sample_rate=15360, output_size=64)
    print("[prep] precomputing CWT images for ZSL split...", flush=True)
    images_all = torch.from_numpy(
        np.stack([preprocessor.process_single(signals[i]) for i in zsl_pos]).astype(np.float32))
    labels_zsl = torch.tensor(labels[zsl_pos], dtype=torch.long)
    print(f"[prep] zsl images={images_all.shape}", flush=True)

    per_method_rows = {m: [] for m in METHODS}
    all_rows = []

    for a_name, b_name in pairs:
        a, b = n2i[a_name], n2i[b_name]
        seen_names = [n for n in CLASS_NAMES if n not in (a_name, b_name)]
        seen_classes = [n2i[n] for n in seen_names]
        seen_pos = np.where(np.isin(labels[zsl_pos], seen_classes))[0]
        # 对 seen 类重新编号（0..2），构建 one-hot
        seen_global = sorted(set(labels_zsl[seen_pos].tolist()))
        cls_map = {int(c): i for i, c in enumerate(seen_global)}
        Y_train = np.zeros((len(seen_pos), len(seen_global)), dtype=np.float32)
        for i, sp in enumerate(seen_pos):
            Y_train[i, cls_map[int(labels_zsl[sp])]] = 1.0
        Y_train = torch.tensor(Y_train, device=device)
        # 属性原型按 seen 类顺序（ESZSL 用）
        attr_seen_ordered = attr_t[torch.tensor(seen_global, device=device)]

        sel = (labels_zsl == a) | (labels_zsl == b)
        true = torch.tensor([0 if int(x) == a else 1 for x in labels_zsl[sel].tolist()],
                            device=device)
        n_test = int(len(true))

        print(f"\n=== {a_name}/{b_name} | seen={seen_names} (n_seen={len(seen_pos)}) ===", flush=True)

        # 每 seed：训分类 backbone → 冻结提取特征 → 跑所有方法
        method_acc = {m: [] for m in METHODS}
        for s in seeds:
            print(f"  [seed {s}] training classifier backbone...", flush=True)
            backbone = train_classifier_backbone(config, device, images_all, labels_zsl,
                                                 seen_pos, s, epochs=args.backbone_epochs,
                                                 backbone=args.backbone)
            X_all = extract_features(backbone, images_all, device).to(device)   # (N,512)
            X_all = F.normalize(X_all, p=2, dim=-1)   # L2 归一化（ZSL 各方法的标准预处理，公平）
            X_train = X_all[torch.tensor(seen_pos)]
            X_test = X_all[sel]
            A_train = feat_zsl[torch.tensor(seen_pos)].to(device)   # 逐样本属性 (n_seen,21)

            # 各方法 → unseen 两类的分数 (n_test, 2)
            res = {}
            res["DAP"] = run_dap(X_train, A_train, X_test, attr_t)[:, torch.tensor([a, b])]
            res["SAE"] = run_sae(X_train, A_train, X_test, attr_t)[:, torch.tensor([a, b])]
            res["ESZSL"] = run_eszsl(X_train, Y_train, X_test, attr_seen_ordered, attr_t)[:, torch.tensor([a, b])]
            res["ALE"] = run_ale(X_train, Y_train, X_test, attr_seen_ordered, attr_t)[:, torch.tensor([a, b])]
            res["SJE"] = run_sje(X_train, Y_train, X_test, attr_seen_ordered, attr_t)[:, torch.tensor([a, b])]
            res["Ours"] = run_ours(X_train, A_train, X_test, attr_t)[:, torch.tensor([a, b])]
            for m in METHODS:
                acc = float((res[m].argmax(1) == true).float().mean().item())
                method_acc[m].append(acc)

        Pn = normalize(attr_t).cpu()
        dot = float((Pn[a] @ Pn[b]).item())
        row = {"pair": f"{a_name}/{b_name}", "seen": seen_names, "attr_cos": round(dot, 3),
               "n": n_test}
        line = f"  [{a_name}/{b_name}] cos={dot:.3f} | "
        for m in METHODS:
            mean_acc = float(np.mean(method_acc[m]))
            row[m] = round(mean_acc, 4)
            per_method_rows[m].append(mean_acc)
            line += f"{m}={mean_acc * 100:.2f}% "
        all_rows.append(row)
        print(line, flush=True)

    print("\n=== 均值（5 对平均）===", flush=True)
    summary = {}
    for m in METHODS:
        mean = float(np.mean(per_method_rows[m]))
        summary[m] = round(mean, 4)
        print(f"  {m}: {mean * 100:.2f}%", flush=True)

    out = {"note": "ESAG baseline 公平对比（共享冻结分类 backbone，严格 inductive，3 seed × 5 对）",
           "seeds": seeds, "backbone_epochs": args.backbone_epochs, "backbone": args.backbone,
           "attr_holdout": ATTR_HOLDOUT, "annot_seed": ANNOT_SEED, "attr_dim": int(attr_t.shape[1]),
           "methods": list(METHODS.keys()), "rows": all_rows, "mean": summary}
    json.dump(out, open(args.results_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n[已另存] {args.results_json}")


if __name__ == "__main__":
    main()
