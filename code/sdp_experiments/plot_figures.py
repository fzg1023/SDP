"""生成论文图（基于各 results JSON，全英文标签，IEEE 风格）。

生成：
  fig_ablation.png      类均值(塌缩) vs 逐样本回归 的消融对比
  fig_dot_vs_acc.png    属性点积余弦 vs unseen 准确率散点（齿轮 + CWRU）
  fig_baseline_bar.png  ESZSL/SAE/ALE/DAP/SJE/Ours 对比柱状图（需 results_baseline_compare.json）
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join("D:/EsAg")

# 齿轮类名中文 -> 英文
EN = {"健康": "Healthy", "裂纹": "Crack", "剥落": "Spall", "磨损": "Wear", "断齿": "Broken tooth"}


def load(name):
    p = os.path.join(HERE, name)
    if not os.path.exists(p):
        print(f"[skip] {name} not found", flush=True)
        return None
    return json.load(open(p, encoding="utf-8"))


def fig_ablation():
    cl = load("results_classmean_ablation.json")
    v10 = load("results_strict_zsl_v10.json")
    if cl is None or v10 is None:
        return
    pairs = [EN.get(r["pair"].split("/")[0], r["pair"].split("/")[0]) + "/" +
             EN.get(r["pair"].split("/")[1], r["pair"].split("/")[1]) for r in v10["rows"]]
    cm = [r["unseen_accuracy"] * 100 for r in cl["rows"]]
    ps = [r["unseen_accuracy"] * 100 for r in v10["rows"]]

    x = np.arange(len(pairs))
    w = 0.38
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.bar(x - w / 2, cm, w, label="Class-mean regression", color="#c44e52", edgecolor="black", linewidth=0.5)
    ax.bar(x + w / 2, ps, w, label="Per-sample regression (ours)", color="#4c72b0", edgecolor="black", linewidth=0.5)
    ax.axhline(50, ls="--", color="gray", lw=0.8, label="Chance (50%)")
    ax.set_xticks(x)
    ax.set_xticklabels(pairs, fontsize=8, rotation=15)
    ax.set_ylabel("Unseen accuracy (%)")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8)
    ax.set_title("Ablation of the regression target (BaiduGear)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_ablation.png"), dpi=200)
    plt.close(fig)
    print("[fig] fig_ablation.png", flush=True)


def fig_dot_vs_acc():
    v10 = load("results_strict_zsl_v10.json")
    cw = load("results_strict_zsl_cwru.json")
    if v10 is None or cw is None:
        return
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    # 齿轮（蓝色）
    cos_g = [r["attr_cos"] for r in v10["rows"]]
    acc_g = [r["unseen_accuracy"] * 100 for r in v10["rows"]]
    ax.scatter(cos_g, acc_g, color="#4c72b0", s=70, marker="o", edgecolor="black",
               linewidth=0.6, label="BaiduGear (5 pairs)")
    # CWRU（红色）
    cos_c = [r["attr_cos"] for r in cw["pairwise"]]
    acc_c = [r["unseen_accuracy"] * 100 for r in cw["pairwise"]]
    ax.scatter(cos_c, acc_c, color="#c44e52", s=70, marker="s", edgecolor="black",
               linewidth=0.6, label="CWRU (3 pairs)")
    ax.set_xlabel("Attribute prototype cosine similarity")
    ax.set_ylabel("Unseen accuracy (%)")
    ax.set_ylim(0, 100)
    ax.axhline(50, ls="--", color="gray", lw=0.8)
    ax.legend(fontsize=8)
    ax.set_title("Accuracy vs. prototype angle")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_dot_vs_acc.png"), dpi=200)
    plt.close(fig)
    print("[fig] fig_dot_vs_acc.png", flush=True)


def fig_baseline_bar():
    b = load("results_baseline_compare.json")
    if b is None:
        return
    names = b["methods"]
    mean = [b["mean"][m] * 100 for m in names]
    colors = {"Ours": "#2e7d32", "DAP": "#4c72b0", "SAE": "#4c72b0",
              "ESZSL": "#dd8452", "ALE": "#dd8452", "SJE": "#dd8452"}
    cols = [colors.get(m, "#4c72b0") for m in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    ax.bar(x, mean, color=cols, edgecolor="black", linewidth=0.5)
    ax.axhline(50, ls="--", color="gray", lw=0.8, label="Chance (50%)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("Mean unseen accuracy (%)")
    ax.set_ylim(0, 100)
    for xi, mi in zip(x, mean):
        ax.text(xi, mi + 1.5, f"{mi:.1f}", ha="center", fontsize=8)
    ax.legend(fontsize=8)
    ax.set_title("Comparison under shared frozen backbone (BaiduGear)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_baseline_bar.png"), dpi=200)
    plt.close(fig)
    print("[fig] fig_baseline_bar.png", flush=True)


if __name__ == "__main__":
    fig_ablation()
    fig_dot_vs_acc()
    fig_baseline_bar()
    print("done", flush=True)
