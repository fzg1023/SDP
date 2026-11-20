"""
P1-2：信号域加噪鲁棒性（噪声加到原始振动信号，再走 Hilbert/CWT），3 seed。

与 cwru_noise_eval.py 的区别：原版把高斯噪声加在【时频图像】上（P1-2 审稿指出应加在信号域）。
本脚本把噪声加到原始一维信号 x(t) 上，再重新预处理（包络谱 + CWT），评估干净训练模型。

训练 clean seen（small CNN, 50 epoch, 逐样本 MSE）；评估未见类 3-way：
  干净 + 信号域 SNR ∈ {20,15,10,5,0} dB。
结果另存 results_cwru_noise_signal.json。
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
SNR_LEVELS = [20, 15, 10, 5, 0]


def train_seen_log(model, loader, device, epochs, lr=5e-4, wd=1e-3):
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
    return


@torch.no_grad()
def score_subset(model, images, attr_t, device, batch_size=256):
    P = F.normalize(attr_t, p=2, dim=-1)
    outs = []
    for i in range(0, len(images), batch_size):
        g = F.normalize(model.attr_head(model.backbone(images[i:i + batch_size].to(device))),
                        p=2, dim=-1)
        outs.append((g @ P.t()).cpu())
    return torch.cat(outs, 0)


def add_noise_signal(signals, snr_db, seed):
    """信号域高斯噪声：对一维信号按自身功率加噪到指定 SNR(dB)。"""
    rng = np.random.RandomState(seed)
    out = np.empty_like(signals)
    for i, x in enumerate(signals):
        p = float((x ** 2).mean()) + 1e-12
        std = np.sqrt(p / (10 ** (snr_db / 10)))
        out[i] = x + rng.randn(*x.shape) * std
    return out


def multiway_acc_from_scores(sc, labels_zsl, unseen_classes):
    sel = np.isin(labels_zsl.numpy(), unseen_classes)
    sub = sc[sel][:, unseen_classes]
    pred = sub.argmax(1)
    true = torch.tensor([unseen_classes.index(int(x)) for x in labels_zsl[sel].tolist()])
    return float((pred == true).float().mean().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--backbone", type=str, default="small")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--results-json", type=str,
                    default=os.path.join(HERE, "results_cwru_noise_signal.json"))
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
    print("[prep] precomputing clean CWT images...", flush=True)
    images_all = torch.from_numpy(
        np.stack([preprocessor.process_single(signals[i]) for i in zsl_pos]).astype(np.float32))
    labels_zsl = torch.tensor(labels[zsl_pos], dtype=torch.long)
    print(f"[prep] zsl images={tuple(images_all.shape)}", flush=True)

    seen_pos = np.where(np.isin(labels[zsl_pos], seen_classes))[0]
    # 未见类样本在 zsl 内的局部下标
    unseen_local = np.where(np.isin(labels_zsl.numpy(), unseen_classes))[0]
    unseen_signals = signals[zsl_pos[unseen_local]]
    print(f"[prep] unseen samples = {len(unseen_local)}", flush=True)

    clean_acc = []
    noise_acc = {s: [] for s in SNR_LEVELS}
    for s in seeds:
        set_seed(s)
        ds = CachedImageDataset(images_all, labels_zsl, feat_zsl, seen_pos)
        train_loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=0, drop_last=True)
        model = build(config, device, attr_t.shape[1], backbone=args.backbone)
        print(f"[seed {s}] training clean model...", flush=True)
        train_seen_log(model, train_loader, device, args.epochs)
        model.eval()

        # clean
        sc = score_subset(model, images_all[unseen_local], attr_t, device)
        a3 = multiway_acc_from_scores(sc, labels_zsl[unseen_local], unseen_classes)
        clean_acc.append(a3)
        print(f"[seed {s}] clean 3-way={a3 * 100:.2f}%", flush=True)

        # 信号域加噪
        for snr in SNR_LEVELS:
            noisy_sig = add_noise_signal(unseen_signals, snr, seed=1000 + snr * 10 + s)
            noisy_img = torch.from_numpy(
                np.stack([preprocessor.process_single(x) for x in noisy_sig]).astype(np.float32))
            sc_n = score_subset(model, noisy_img, attr_t, device)
            a_n = multiway_acc_from_scores(sc_n, labels_zsl[unseen_local], unseen_classes)
            noise_acc[snr].append(a_n)
            print(f"    [seed {s}] SNR={snr:>3}dB 3-way={a_n * 100:.2f}%", flush=True)

    clean_mean = float(np.mean(clean_acc))
    clean_std = float(np.std(clean_acc))
    noise_mean = {str(s): round(float(np.mean(noise_acc[s])), 4) for s in SNR_LEVELS}
    noise_std = {str(s): round(float(np.std(noise_acc[s])), 4) for s in SNR_LEVELS}

    print(f"\n=== CWRU 信号域噪声 clean = {clean_mean * 100:.2f}% ± {clean_std * 100:.2f}% ===")
    for s in SNR_LEVELS:
        print(f"  SNR {s:>3}dB: {noise_mean[str(s)] * 100:.2f}% ± {noise_std[str(s)] * 100:.2f}%")

    out = {"note": "CWRU 信号域加噪鲁棒性（噪声加到原始信号，3 seed, small CNN）",
           "seeds": seeds, "epochs": args.epochs, "backbone": args.backbone,
           "snr_levels": SNR_LEVELS,
           "clean_three_way_mean": round(clean_mean, 4), "clean_three_way_std": round(clean_std, 4),
           "clean_per_seed": [round(x, 4) for x in clean_acc],
           "noise_three_way_mean": noise_mean, "noise_three_way_std": noise_std}
    json.dump(out, open(args.results_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n[已另存] {args.results_json}")


if __name__ == "__main__":
    main()
