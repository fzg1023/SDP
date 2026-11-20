"""
t-SNE visualisation (ENGLISH): CWRU attribute-space geometry.

Left: true 21-dim physical attributes (ground truth, decides the upper bound);
Right: projection q(x) learned by the small CNN (normalised).
Colours follow the 10 classes; unseen classes (IR021/B021/OR021) are marked with stars.

Same v10 mechanism: 20% holdout for attribute calibration, 80% ZSL, per-sample MSE
regression, small CNN. Deterministic (fixed seed), reproduces the Chinese figure
with English labels only.
"""
from __future__ import annotations
import os, sys, argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "demo"))
sys.path.insert(0, HERE)

from esag.utils import load_config, set_seed, get_device
from esag.preprocessing import SignalPreprocessor
from esag.datasets import CWRUDataset
from train_fix_v10 import image_features, CachedImageDataset, build, train_seen

CFG = os.path.join(ROOT, "configs", "light_semantic.yaml")
CWRU_DATA = os.environ.get("SDP_CWRU_DATA", "D:/EsAg/data/cwru")
FAULT_TYPES = ["Normal", "IR007", "IR014", "IR021", "B007", "B014",
               "B021", "OR007", "OR014", "OR021"]
UNSEEN_NAMES = ["IR021", "B021", "OR021"]
LOADS = [0, 1, 2, 3]
SIGNAL_LENGTH = 2048
SAMPLE_RATE = 12000
ATTR_HOLDOUT = 0.2
ANNOT_SEED = 42


@torch.no_grad()
def project_all(model, images, device, batch_size=256):
    outs = []
    for i in range(0, len(images), batch_size):
        q = model.attr_head(model.backbone(images[i:i + batch_size].to(device)))
        outs.append(F.normalize(q, p=2, dim=-1).cpu())
    return torch.cat(outs, 0).numpy()


def run_tsne(X, seed=0):
    perp = float(min(30.0, max(5.0, X.shape[0] / 50.0)))
    return TSNE(n_components=2, perplexity=perp, random_state=seed, init='pca').fit_transform(X)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--backbone", type=str, default="small")
    ap.add_argument("--out", type=str, default=os.path.join(HERE, "_fix_en", "image12_tsne.png"))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    config = load_config(CFG)
    device = get_device("auto")

    loader = CWRUDataset(data_path=CWRU_DATA, signal_length=SIGNAL_LENGTH,
                         sample_rate=SAMPLE_RATE, loads=LOADS)
    signals, labels, class_indices, class_names = loader.load_data(fault_types=FAULT_TYPES)
    n2i = {n: i for i, n in enumerate(class_names)}
    n_classes = len(class_names)
    unseen_classes = [n2i[n] for n in UNSEEN_NAMES]
    seen_classes = [i for i in range(n_classes) if i not in unseen_classes]
    print(f"[data] n={len(signals)} seen={[class_names[i] for i in seen_classes]} "
          f"unseen={[class_names[i] for i in unseen_classes]}", flush=True)

    feat = np.stack([image_features(s) for s in signals])  # (N,21)

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

    seen_pos = np.where(np.isin(labels[zsl_pos], seen_classes))[0]
    set_seed(args.seed)
    ds = CachedImageDataset(images_all, labels_zsl, feat_zsl, seen_pos)
    train_loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=0, drop_last=True)
    model = build(config, device, attr_t.shape[1], backbone=args.backbone)
    train_seen(model, train_loader, attr_t, seen_classes, device, args.epochs,
               os.path.join(HERE, "checkpoints", "tsne_cwru"))
    model.eval()

    q = project_all(model, images_all, device)   # (N_zsl, 21) normalised projection
    a = feat_zsl.numpy()                          # (N_zsl, 21) true attributes

    idx = np.arange(len(zsl_pos))
    if len(idx) > 3000:
        idx = np.random.RandomState(0).choice(idx, 3000, replace=False)
    a2 = run_tsne(a[idx])
    q2 = run_tsne(q[idx])
    lab = labels_zsl.numpy()[idx]
    unseen_set = set(unseen_classes)

    cmap = plt.get_cmap('tab10')
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.6))
    for ax, X2, title in [(axes[0], a2, 'True physical attributes (ground truth)'),
                          (axes[1], q2, 'Projection q(x) learned by the small CNN')]:
        for c in range(n_classes):
            m = lab == c
            if not m.any():
                continue
            mk = '*' if c in unseen_set else 'o'
            s = 300 if c in unseen_set else 90
            lw = 1.2 if c in unseen_set else 0.8
            ax.scatter(X2[m, 0], X2[m, 1], s=s, color=cmap(c), label=class_names[c],
                       marker=mk, edgecolors='black', linewidths=lw)
        ax.set_title(title, fontsize=22)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc='lower center', bbox_to_anchor=(0.5, -0.02),
               ncol=5, fontsize=15, frameon=False, markerscale=1.3,
               handletextpad=0.5, columnspacing=1.6, labelspacing=0.5)
    fig.suptitle('CWRU attribute-space t-SNE (unseen classes = stars)', fontsize=24)
    plt.tight_layout(rect=[0, 0.12, 1, 0.94])
    plt.savefig(args.out, dpi=400, bbox_inches='tight', facecolor='white')
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
