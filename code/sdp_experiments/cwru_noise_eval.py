"""
CWRU 噪声鲁棒性 + 混淆矩阵 + 训练损失曲线（一体脚本，3 seed，不改模型）。

训练 clean seen（small CNN, 50 epoch, 逐样本 MSE，记录逐 epoch loss）；
评估：
  1) clean 3-way + 逐类 recall + 逐样本预测（供混淆矩阵）
  2) 图像域高斯噪声 SNR ∈ {20,15,10,5,0,-5} dB 下 3-way 准确率（噪声曲线）
结果另存 results_cwru_noise.json；预测 dump 到 cwru_predictions.npz；loss 存 JSON。
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
from esag.datasets import CWRUDataset
from train_fix_v10 import image_features, CachedImageDataset, build

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
SNR_LEVELS = [20, 15, 10, 5, 0, -5]


def train_seen_log(model, loader, device, epochs, lr=5e-4, wd=1e-3):
    """逐样本 MSE 回归（记录逐 epoch loss），逻辑与 train_fix_v10.train_seen 一致。"""
    params = list(model.backbone.parameters()) + list(model.attr_head.parameters())
    optimizer = AdamW(params, lr=lr, weight_decay=wd)
    warmup = 5
    scheduler = SequentialLR(
        optimizer,
        schedulers=[LinearLR(optimizer, start_factor=0.1, total_iters=warmup),
                    CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup), eta_min=1e-6)],
        milestones=[warmup])
    epoch_loss = []
    for ep in range(1, epochs + 1):
        model.train()
        tr, nb = 0.0, 0
        for batch in loader:
            images = batch["image"].to(device)
            y_attr = batch["feat"].to(device)
            g = model.attr_head(model.backbone(images))
            loss = F.mse_loss(g, y_attr)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            tr += loss.item()
            nb += 1
        scheduler.step()
        epoch_loss.append(float(tr / nb))
    return epoch_loss


@torch.no_grad()
def eval_scores(model, images, attr_t, device, batch_size=256):
    P = F.normalize(attr_t, p=2, dim=-1)
    outs = []
    for i in range(0, len(images), batch_size):
        g = F.normalize(model.attr_head(model.backbone(images[i:i + batch_size].to(device))),
                        p=2, dim=-1)
        outs.append((g @ P.t()).cpu())
    return torch.cat(outs, 0)


def add_noise(images, snr_db):
    """图像域高斯噪声：每样本按自身信号功率加噪声到指定 SNR(dB)。"""
    p = (images ** 2).mean(dim=(1, 2, 3), keepdim=True) + 1e-12
    std = torch.sqrt(p / (10 ** (snr_db / 10)))
    return images + torch.randn_like(images) * std


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
    ap.add_argument("--backbone", type=str, default="small")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--results-json", type=str,
                    default=os.path.join(HERE, "results_cwru_noise.json"))
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
    print(f"[data] n={len(signals)}", flush=True)

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
    attr_raw = np.stack([Z[attr_pos][labels[attr_pos] == i].mean(0)
                         for i in range(len(class_names))])
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

    # 结果容器
    clean_acc = []          # 每 seed clean 3-way
    clean_recall = []
    noise_acc = {s: [] for s in SNR_LEVELS}   # snr -> per-seed acc
    loss_curves = []        # 每 seed 逐 epoch loss
    preds_all = []          # 每 seed unseen 样本预测 (类别内索引 0/1/2)
    true_all = []

    for s in seeds:
        set_seed(s)
        ds = CachedImageDataset(images_all, labels_zsl, feat_zsl, seen_pos)
        train_loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=0, drop_last=True)
        model = build(config, device, attr_t.shape[1], backbone=args.backbone)
        print(f"[seed {s}] training...", flush=True)
        ep_loss = train_seen_log(model, train_loader, device, args.epochs)
        loss_curves.append(ep_loss)
        model.eval()

        # clean
        sc = eval_scores(model, images_all, attr_t, device)
        a3, n3 = multiway_acc(sc, labels_zsl, unseen_classes)
        r3 = per_class_recall(sc, labels_zsl, unseen_classes)
        clean_acc.append(a3)
        clean_recall.append(r3)
        # 逐样本预测（unseen 样本，类别内 0/1/2）
        sel = np.isin(labels_zsl.numpy(), unseen_classes)
        sub = sc[sel][:, unseen_classes]
        preds_all.append(sub.argmax(1).numpy())
        true_all.append(torch.tensor(
            [unseen_classes.index(int(x)) for x in labels_zsl[sel].tolist()]).numpy())
        print(f"[seed {s}] clean 3-way={a3 * 100:.2f}%", flush=True)

        # noisy
        for snr in SNR_LEVELS:
            noisy = add_noise(images_all, snr)
            sc_n = eval_scores(model, noisy, attr_t, device)
            a_n, _ = multiway_acc(sc_n, labels_zsl, unseen_classes)
            noise_acc[snr].append(a_n)
            print(f"    [seed {s}] SNR={snr:>3}dB 3-way={a_n * 100:.2f}%", flush=True)

    # 汇总
    clean_mean = float(np.mean(clean_acc))
    clean_std = float(np.std(clean_acc))
    rec_mean = {class_names[c]: round(float(np.mean([r.get(c, 0.0) for r in clean_recall])), 4)
                for c in unseen_classes}

    noise_mean = {str(s): round(float(np.mean(noise_acc[s])), 4) for s in SNR_LEVELS}
    noise_std = {str(s): round(float(np.std(noise_acc[s])), 4) for s in SNR_LEVELS}

    loss_mean = [float(np.mean([lc[e] for lc in loss_curves])) for e in range(args.epochs)]

    print(f"\n=== CWRU clean 3-way = {clean_mean * 100:.2f}% ± {clean_std * 100:.2f}% ===")
    for s in SNR_LEVELS:
        print(f"  SNR {s:>3}dB: {noise_mean[str(s)] * 100:.2f}% ± {noise_std[str(s)] * 100:.2f}%")

    out = {
        "note": "CWRU 噪声鲁棒性 + 混淆矩阵 + 训练损失（3 seed, small CNN, 不改模型）",
        "seeds": seeds, "epochs": args.epochs, "backbone": args.backbone,
        "snr_levels": SNR_LEVELS,
        "clean_three_way_mean": round(clean_mean, 4), "clean_three_way_std": round(clean_std, 4),
        "clean_recall_mean": rec_mean,
        "noise_three_way_mean": noise_mean, "noise_three_way_std": noise_std,
        "loss_per_epoch_mean": [round(x, 5) for x in loss_mean],
        "loss_per_epoch_per_seed": [[round(x, 5) for x in lc] for lc in loss_curves],
    }
    json.dump(out, open(args.results_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    np.savez(os.path.join(HERE, "cwru_predictions.npz"),
             preds=np.array(preds_all), true=np.array(true_all), seeds=np.array(seeds))
    print(f"\n[已另存] {args.results_json} + cwru_predictions.npz")


if __name__ == "__main__":
    main()
