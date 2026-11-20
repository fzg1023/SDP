"""
ESAG 真零样本修复 v10 —— 图像域物理特征（完全可恢复）+ 逐样本回归 + 语义点积。

根因链（v7→v9 逐步定位）：
  1) 手写属性与数据可分性错位 → 换物理特征（16维→26维）。
  2) 回归目标用「类均值原型」把 seen 类压成 3 个紧簇 → 骨干学成 3 模态分类器 → 塌缩
     （v7 38.6%、v8 30.5%）。v9 改为【逐样本回归自身物理特征】，健康/裂纹 52.9%(余弦)/
     56.2%(MSE)，不再塌缩、真超过随机。
  3) 残余差距：recoverable_features 含「原始 FFT 谱」统计，而骨干输入是 CWT 幅度图 +
     包络谱图，原始 FFT 不在输入里、只能近似恢复 → 引入 ~10% 不可恢复误差。

v10：把属性定义成【图像域可精确恢复】的量（CWT 子带能量 + CWT 尺度边缘谱形状 + 包络谱
形状，共 21 维）。直接匹配上界 68.3%（比 recoverable_features 的 65.0% 更高），且回归
目标与骨干输入信息完全一致 → 学习型骨干应能真正逼近上界。

协议（严格 inductive）：独立标注集（每类 20%）算类原型；训练只用 seen 类样本（逐样本
回归其自身图像域特征，MSE）；测试 unseen 对纯语义点积 argmax_c normalize(g)·normalize(a_c)；
不碰 unseen 标签、不早停、固定 seed。

用法：
  D:/Anaconda/envs/esag/python.exe -u train_fix_v10.py --epochs 50 --seeds 1
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset
from scipy.signal import hilbert
import pywt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "demo"))

from esag.utils import load_config, set_seed, get_device
from esag.preprocessing import SignalPreprocessor
from esag.model import ESAG
from demo.train_gear_baidu import BaiduGearDataset

CFG = os.path.join(ROOT, "configs", "light_semantic.yaml")   # swin_lite
GEAR_DATA = os.environ.get("SDP_GEAR_DATA", "D:/BaiduNetdiskDownload/GearDatasets")
CLASS_NAMES = ["健康", "剥落", "断齿", "磨损", "裂纹"]
ALL_PAIRS = [("健康", "裂纹"), ("剥落", "磨损"), ("断齿", "磨损"),
             ("剥落", "裂纹"), ("剥落", "断齿")]
MAX_FILES = 50
MAX_SEGMENTS = 50
TAU = 0.1
ALPHA = 1.0
ATTR_HOLDOUT = 0.2
ANNOT_SEED = 42


def image_features(x: np.ndarray) -> np.ndarray:
    """~21 维图像域特征：CWT 子带能量 + CWT 尺度边缘谱形状 + 包络谱形状。

    全部是骨干输入（CWT 幅度图 + 包络谱图）里直接包含的量，可精确恢复。
    """
    x = x - np.mean(x)

    # CWT 幅度（= 骨干通道1，不做 min-max 归一化以保留相对能量）
    coefs, _ = pywt.cwt(x, np.arange(1, 65), 'morl')  # (64, N)
    mag = np.abs(coefs)
    pw = mag ** 2

    # 8 个 CWT 子带能量（归一）
    bands = np.array([float(np.sum(pw[i * 8:(i + 1) * 8])) for i in range(8)])
    bands = bands / (bands.sum() + 1e-12)

    # CWT 尺度边缘谱形状（能量随尺度分布）
    marginal = pw.sum(axis=1)
    m = marginal / (marginal.sum() + 1e-12)
    scales = np.arange(64) / 63.0
    c_centroid = np.sum(scales * m)
    c_spread = np.sqrt(np.sum((scales - c_centroid) ** 2 * m) + 1e-12)
    c_skew = np.sum((scales - c_centroid) ** 3 * m) / (c_spread ** 3 + 1e-12)
    c_kurt = np.sum((scales - c_centroid) ** 4 * m) / (c_spread ** 4 + 1e-12)
    c_entropy = -np.sum(m * np.log(m + 1e-12))

    # 包络谱（= 骨干通道0）
    env = np.abs(hilbert(x))
    env = env - np.mean(env)
    env_spec = np.abs(np.fft.rfft(env))
    es = env_spec / (env_spec.sum() + 1e-12)
    freqs = np.arange(len(es)) / len(es)
    e_centroid = np.sum(freqs * es)
    e_spread = np.sqrt(np.sum((freqs - e_centroid) ** 2 * es) + 1e-12)
    e_skew = np.sum((freqs - e_centroid) ** 3 * es) / (e_spread ** 3 + 1e-12)
    e_kurt = np.sum((freqs - e_centroid) ** 4 * es) / (e_spread ** 4 + 1e-12)
    e_entropy = -np.sum(es * np.log(es + 1e-12))
    e_flatness = np.exp(np.mean(np.log(es + 1e-12))) / (np.mean(es) + 1e-12)
    e_peak_freq = float(np.argmax(es)) / len(es)
    e_peak_amp = float(np.max(es))

    return np.concatenate([
        bands, [c_centroid, c_spread, c_skew, c_kurt, c_entropy],
        [e_centroid, e_spread, e_skew, e_kurt, e_entropy, e_flatness,
         e_peak_freq, e_peak_amp]])


class CachedImageDataset(Dataset):
    def __init__(self, images, labels, feats, indices, noise_std=0.02, time_shift=8):
        self.images, self.labels, self.feats, self.indices = images, labels, feats, indices
        self.noise_std, self.time_shift = noise_std, time_shift

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        img = self.images[idx]
        if self.noise_std > 0:
            img = img + torch.randn_like(img) * self.noise_std
        if self.time_shift > 0:
            sh = int(torch.randint(-self.time_shift, self.time_shift + 1, (1,)).item())
            img = torch.roll(img, shifts=sh, dims=-1)
        return {"image": img, "label": self.labels[idx], "feat": self.feats[idx]}


def build(config, device, attr_dim, drop_path=0.3, backbone="swin_lite"):
    config["model"]["backbone_type"] = backbone
    config["model"]["use_generative"] = False
    config["model"]["semantic"]["num_attributes"] = attr_dim
    config["model"]["semantic"]["attribute_set"] = "baidu_gear"
    config["model"]["swin"]["drop_path_rate"] = drop_path
    model = ESAG(config).to(device)
    model.attr_head = nn.Linear(model.feature_dim, attr_dim).to(device)
    return model


def train_seen(model, loader, attr_t, seen_classes, device, epochs, save_dir,
               lr=5e-4, wd=1e-3, ce_weight=0.0, loss_type="mse"):
    seen_t = torch.tensor(seen_classes, device=device, dtype=torch.long)
    attr_seen = attr_t[seen_t]
    pos = {int(c): i for i, c in enumerate(seen_classes)}
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
        tr = tc = 0.0
        nb = 0
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            y_attr = batch["feat"].to(device)          # 样本自身图像域特征
            g = model.attr_head(model.backbone(images))
            if loss_type == "mse":
                loss_reg = F.mse_loss(g, y_attr)
                gn = F.normalize(g, p=2, dim=-1)
            else:
                gn = F.normalize(g, p=2, dim=-1)
                yn = F.normalize(y_attr, p=2, dim=-1)
                loss_reg = (1.0 - (gn * yn).sum(-1)).mean()
            loss = ALPHA * loss_reg
            if ce_weight > 0:
                logits = gn @ F.normalize(attr_seen, p=2, dim=-1).t() / TAU
                y = torch.tensor([pos[int(l)] for l in labels.cpu().tolist()],
                                 device=device, dtype=torch.long)
                loss_ce = F.cross_entropy(logits, y)
                loss = loss + ce_weight * loss_ce
            else:
                loss_ce = torch.zeros(())
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            tr += loss_reg.item()
            tc += loss_ce.item()
            nb += 1
        scheduler.step()
        if ep % 10 == 0 or ep == epochs:
            print(f"    [train] epoch {ep:3d}/{epochs}  reg={tr / nb:.4f} ce={tc / nb:.4f}", flush=True)


@torch.no_grad()
def eval_scores(model, images, attr_t, device, batch_size=256):
    P = F.normalize(attr_t, p=2, dim=-1)
    outs = []
    for i in range(0, len(images), batch_size):
        g = F.normalize(model.attr_head(model.backbone(images[i:i + batch_size].to(device))),
                        p=2, dim=-1)
        outs.append((g @ P.t()).cpu())
    return torch.cat(outs, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=str, default=None)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--drop-path", type=float, default=0.3)
    ap.add_argument("--ce-weight", type=float, default=0.0)
    ap.add_argument("--loss", type=str, default="mse", choices=["mse", "cosine"])
    ap.add_argument("--backbone", type=str, default="swin_lite")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--results-json", type=str,
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "results_strict_zsl_v10.json"))
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

    # 图像域物理特征
    feat = np.stack([image_features(s) for s in signals])  # (N,21)

    # 独立标注集（每类 20%，固定 seed）→ 类别属性；其余 80% 用于 ZSL
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

    # 用标注集统计做 z-score；类原型 = 标注集各类均值
    mu = feat[attr_pos].mean(0)
    sd = feat[attr_pos].std(0) + 1e-12
    Z = (feat - mu) / sd
    attr_raw = np.stack([Z[attr_pos][labels[attr_pos] == i].mean(0) for i in range(5)])  # (5,21)
    attr_t = torch.tensor(attr_raw, dtype=torch.float32, device=device)
    feat_zsl = torch.tensor(Z[zsl_pos], dtype=torch.float32)  # 样本自身特征（逐样本回归目标）
    print(f"[attr] image-domain attribute matrix from {len(attr_pos)} annot samples", flush=True)

    # CWT 图（仅 ZSL 80%）
    preprocessor = SignalPreprocessor(signal_length=2048, sample_rate=15360, output_size=64)
    print("[prep] precomputing CWT images for ZSL split...", flush=True)
    images_all = torch.from_numpy(
        np.stack([preprocessor.process_single(signals[i]) for i in zsl_pos]).astype(np.float32))
    labels_zsl = torch.tensor(labels[zsl_pos], dtype=torch.long)
    print(f"[prep] zsl images={images_all.shape}", flush=True)

    rows = []
    for a_name, b_name in pairs:
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
            save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "checkpoints", f"v10_{a_name}_{b_name}_s{s}")
            print(f"  [seed {s}] training...", flush=True)
            train_seen(model, train_loader, attr_t, seen_classes, device, args.epochs, save_dir,
                       lr=args.lr, wd=args.wd, ce_weight=args.ce_weight, loss_type=args.loss)
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
        rows.append({"pair": f"{a_name}/{b_name}", "seen": seen_names,
                     "attr_cos": round(dot, 3), "unseen_accuracy": round(acc, 4), "n": n})
        print(f"  [{a_name}/{b_name}] 属性余弦={dot:.3f}  unseen 准确率={acc * 100:.2f}% "
              f"(n={n}, seeds={len(seeds)})", flush=True)

    print("\n| 不可见对 | 属性余弦 | unseen 准确率 |")
    print("|----------|----------|---------------|")
    for r in rows:
        print(f"| {r['pair']} | {r['attr_cos']:.3f} | {r['unseen_accuracy'] * 100:.2f}% |")
    mean = float(np.mean([r["unseen_accuracy"] for r in rows]))
    print(f"\n均值准确率 = {mean * 100:.2f}%")

    out = {"note": "ESAG 真零样本修复 v10（图像域物理特征 + 逐样本回归 + 语义点积 + 严格 inductive）",
           "seeds": seeds, "epochs": args.epochs, "lr": args.lr, "wd": args.wd,
           "drop_path": args.drop_path, "ce_weight": args.ce_weight, "loss": args.loss,
           "backbone": args.backbone, "attr_holdout": ATTR_HOLDOUT, "annot_seed": ANNOT_SEED,
           "attr_dim": int(attr_t.shape[1]), "rows": rows, "mean_accuracy": round(mean, 4)}
    json.dump(out, open(args.results_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n[已另存] {args.results_json}")


if __name__ == "__main__":
    main()
