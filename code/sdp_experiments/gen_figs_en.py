# -*- coding: utf-8 -*-
"""Regenerate figures 8/10/11/13 with ENGLISH labels from saved result JSONs (no training)."""
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "_fix_en")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'font.size': 14,
    'font.sans-serif': ['DejaVu Sans'],
    'axes.unicode_minus': False,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

def fig8_confusion():
    d = json.load(open(os.path.join(HERE, "results_cwru_confusion.json"), encoding="utf-8"))
    C = np.array(d["confusion_matrix"])
    labels = ["IR021", "B021", "OR021"]
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(C, cmap="Blues")
    ax.set_xticks(range(3)); ax.set_xticklabels(labels)
    ax.set_yticks(range(3)); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    for i in range(3):
        for j in range(3):
            v = C[i, j]
            ax.text(j, i, f"{v}", ha="center", va="center", fontsize=20,
                    color="white" if v > C.max() * 0.5 else "black")
    ax.set_title("CWRU three-way confusion matrix (unseen classes, 5 seeds)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.savefig(os.path.join(OUT, "image8_confusion.png"))
    plt.close(fig)
    print("[fig8] done")

def fig10_baseline():
    names = ["ESZSL", "SJE", "ALE", "SAE", "VAE", "GCN",
             "DAP", "SDP\n(frozen)", "DAP\n(end-to-end)", "SAE\n(end-to-end)", "SDP\n(end-to-end)"]
    acc = [50.56, 52.07, 52.05, 58.00, 64.94, 64.97,
           66.44, 65.46, 67.60, 67.43, 67.60]
    colors = (["#b0b0b0"] * 3 + ["#7fb3d5"] * 2 + ["#5dade2"] * 3 +
              ["#e67e22"] * 2 + ["#c0392b"])
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    bars = ax.bar(names, acc, color=colors, width=0.66)
    for bar, v in zip(bars, acc):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.5, f"{v:.2f}",
                ha="center", va="bottom", fontsize=12)
    ax.axhline(50.0, ls="--", color="red", lw=1.4, label="Random level 50%")
    ax.set_ylabel("5-pair mean accuracy (%)")
    ax.set_ylim(0, 78)
    ax.set_title("NEU-Gear zero-shot baseline comparison (frozen + end-to-end)")
    ax.legend(loc="upper left")
    fig.savefig(os.path.join(OUT, "image10_baseline.png"))
    plt.close(fig)
    print("[fig10] done")

def fig11_ablation():
    pairs = ["Healthy/Crack", "Spalling/Wear", "Tooth\nbreakage/Wear", "Spalling/Crack", "Spalling/Tooth\nbreakage"]
    cm = [28.27, 44.76, 71.37, 60.49, 42.96]
    ps = [62.48, 61.60, 82.55, 61.95, 69.42]
    x = np.arange(len(pairs))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    b1 = ax.bar(x - w / 2, cm, w, label="Class-mean regression (collapse)", color="#c44e52", edgecolor="black", linewidth=0.5)
    b2 = ax.bar(x + w / 2, ps, w, label="Per-sample regression (ours)", color="#4c72b0", edgecolor="black", linewidth=0.5)
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.6, f"{h:.1f}",
                    ha="center", va="bottom", fontsize=9)
    ax.axhline(50, ls="--", color="gray", lw=0.8, label="Random level 50%")
    ax.set_xticks(x)
    ax.set_xticklabels(pairs, rotation=12)
    ax.set_ylabel("Unseen accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Regression-target ablation (NEU-Gear): class-mean vs. per-sample")
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "image11_ablation.png"))
    plt.close(fig)
    print("[fig11] done")

def fig13_noise():
    r = json.load(open(os.path.join(HERE, "results_cwru_noise_signal.json"), encoding="utf-8"))
    snr = ["clean"] + [str(s) for s in r["snr_levels"]]
    x = np.arange(len(snr))
    acc = [r["clean_three_way_mean"]] + [r["noise_three_way_mean"][str(s)] for s in r["snr_levels"]]
    std = [r["clean_three_way_std"]] + [r["noise_three_way_std"][str(s)] for s in r["snr_levels"]]
    acc = [a * 100 for a in acc]; std = [s * 100 for s in std]
    labels = ["No noise"] + [f"{s} dB" for s in r["snr_levels"]]
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    ax.errorbar(x, acc, yerr=std, marker="o", ms=7, lw=2, capsize=5, color="#2874a6")
    for xi, (a, s) in enumerate(zip(acc, std)):
        ax.annotate(f"{a:.2f}", (xi, a + s + 2), ha="center", fontsize=10, color="#2874a6")
    ax.axhline(33.3, ls="--", color="red", lw=1.4, label="Random level 33.3%")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Three-way accuracy (%)")
    ax.set_xlabel("SNR of Gaussian noise on raw signal")
    ax.set_ylim(0, 100)
    ax.set_title("CWRU signal-domain noise robustness (3-seed mean \u00b1 std)")
    ax.legend(loc="upper right")
    fig.savefig(os.path.join(OUT, "image13_noise.png"))
    plt.close(fig)
    print("[fig13] done")

if __name__ == "__main__":
    fig8_confusion(); fig10_baseline(); fig11_ablation(); fig13_noise()
    print("ALL ENGLISH FIGURES ->", OUT)
