"""
P1-1：端到端强基线（DAP + SAE），与 SDP 同协议同骨干（small CNN），放开骨干一起训。

目的：回应审稿「基线与 SDP 训练条件不对等（冻结 vs 端到端）」。
协议与 baseline_compare.py / train_fix_v10.py 完全一致：
  留二类 5 对、20% 标定集算类原型、seen 训练 / unseen 测试、语义点积最近原型。
  small CNN 骨干，3 seed（42/49/56，与表 3 对齐）。

方法（都端到端，骨干可训练）：
  - DAP（端到端）：骨干 + 线性属性回归（逐样本 MSE）→ 点积。这正是 SDP 的线性投影，
    用于验证「放开骨干后 DAP 即退化为 SDP 投影」。
  - SAE（端到端）：骨干 + 语义自编码（编码到 21 维属性 + 共享权重解码重建特征，
    loss = MSE(recon) + 0.01*MSE(code,attr)）→ 点积。

结果另存 results_e2e_baselines.json。
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
LAM_ATTR = 0.01


def train_sae_e2e(model, loader, device, epochs, lr=5e-4, wd=1e-3, lam_attr=LAM_ATTR):
    """端到端语义自编码：L2 归一化特征 f，编码 code=attr_head(f)，共享权重解码，重构 + 属性一致。"""
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
        tr = tc = 0.0
        nb = 0
        for batch in loader:
            img = batch["image"].to(device)
            attr = batch["feat"].to(device)
            f = model.backbone(img)
            fn = F.normalize(f, p=2, dim=-1)
            code = model.attr_head(fn)                    # (B,21)
            W = model.attr_head.weight                    # (21, feature_dim)
            recon = code @ W                              # (B, feature_dim)
            loss_recon = F.mse_loss(recon, fn)
            loss_attr = F.mse_loss(code, attr)
            loss = loss_recon + lam_attr * loss_attr
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            tr += loss_recon.item()
            tc += loss_attr.item()
            nb += 1
        scheduler.step()
        if ep % 10 == 0 or ep == epochs:
            print(f"    [sae] epoch {ep:3d}/{epochs} recon={tr / nb:.4f} attr={tc / nb:.4f}",
                  flush=True)


@torch.no_grad()
def eval_scores_normfeat(model, images, attr_t, device, batch_size=256):
    """SAE 测试：先 L2 归一化特征再编码（与训练一致），语义点积。"""
    P = F.normalize(attr_t, p=2, dim=-1)
    outs = []
    for i in range(0, len(images), batch_size):
        f = model.backbone(images[i:i + batch_size].to(device))
        fn = F.normalize(f, p=2, dim=-1)
        g = F.normalize(model.attr_head(fn), p=2, dim=-1)
        outs.append((g @ P.t()).cpu())
    return torch.cat(outs, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--backbone", type=str, default="small")
    ap.add_argument("--results-json", type=str,
                    default=os.path.join(HERE, "results_e2e_baselines.json"))
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

    method_rows = {"DAP_e2e": [], "SAE_e2e": []}
    rows = []
    for a_name, b_name in ALL_PAIRS:
        a, b = n2i[a_name], n2i[b_name]
        seen_names = [n for n in CLASS_NAMES if n not in (a_name, b_name)]
        seen_classes = [n2i[n] for n in seen_names]
        seen_pos = np.where(np.isin(labels[zsl_pos], seen_classes))[0]
        print(f"\n=== {a_name}/{b_name} | seen={seen_names} ===", flush=True)

        dap_acc = []
        sae_acc = []
        for s in seeds:
            # DAP 端到端（= SDP 线性投影，逐样本 MSE）
            set_seed(s)
            ds = CachedImageDataset(images_all, labels_zsl, feat_zsl, seen_pos)
            train_loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=0,
                                      drop_last=True)
            model = build(config, device, attr_t.shape[1], drop_path=0.3, backbone=args.backbone)
            train_seen(model, train_loader, attr_t, seen_classes, device, args.epochs,
                       os.path.join(HERE, "checkpoints", f"e2e_dap_{a_name}_{b_name}_s{s}"),
                       lr=args.lr, wd=args.wd)
            model.eval()
            sc = eval_scores(model, images_all, attr_t, device)
            sel = (labels_zsl == a) | (labels_zsl == b)
            pred = sc[sel][:, torch.tensor([a, b])].argmax(1)
            true = torch.tensor([0 if int(x) == a else 1 for x in labels_zsl[sel].tolist()])
            dap_acc.append(float((pred == true).float().mean().item()))

            # SAE 端到端
            set_seed(s)
            model2 = build(config, device, attr_t.shape[1], drop_path=0.3, backbone=args.backbone)
            train_sae_e2e(model2, train_loader, device, args.epochs, lr=args.lr, wd=args.wd)
            model2.eval()
            sc2 = eval_scores_normfeat(model2, images_all, attr_t, device)
            pred2 = sc2[sel][:, torch.tensor([a, b])].argmax(1)
            sae_acc.append(float((pred2 == true).float().mean().item()))

            print(f"  [seed {s}] DAP_e2e={dap_acc[-1] * 100:.2f}%  SAE_e2e={sae_acc[-1] * 100:.2f}%",
                  flush=True)

        m_dap = round(float(np.mean(dap_acc)), 4)
        m_sae = round(float(np.mean(sae_acc)), 4)
        method_rows["DAP_e2e"].append(m_dap)
        method_rows["SAE_e2e"].append(m_sae)
        Pn = F.normalize(attr_t, p=2, dim=-1).cpu()
        dot = float((Pn[a] @ Pn[b]).item())
        rows.append({"pair": f"{a_name}/{b_name}", "attr_cos": round(dot, 3),
                     "DAP_e2e": m_dap, "SAE_e2e": m_sae})
        print(f"  [{a_name}/{b_name}] cos={dot:.3f} DAP_e2e={m_dap * 100:.2f}% "
              f"SAE_e2e={m_sae * 100:.2f}%", flush=True)

    mean = {m: round(float(np.mean(v)), 4) for m, v in method_rows.items()}
    print("\n=== 5 对均值 ===")
    for m, v in mean.items():
        print(f"  {m}: {v * 100:.2f}%")

    out = {"note": "端到端强基线 DAP/SAE（small CNN，放开骨干，3 seed × 5 对，协议同 SDP）",
           "seeds": seeds, "epochs": args.epochs, "backbone": args.backbone,
           "attr_holdout": ATTR_HOLDOUT, "annot_seed": ANNOT_SEED,
           "attr_dim": int(attr_t.shape[1]), "rows": rows, "mean": mean}
    json.dump(out, open(args.results_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n[已另存] {args.results_json}")


if __name__ == "__main__":
    main()
