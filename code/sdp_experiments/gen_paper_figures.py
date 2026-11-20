"""
生成论文全部图表（高清 dpi=300，大字体）。

产出（写到 D:/EsAg）：
  fig_baseline_bar.png   基线对比柱状图（含近年 VAE/GCN 复现）
  fig_confusion_cwru.png CWRU 三分类混淆矩阵
  fig_noise_cwru.png     CWRU 噪声鲁棒性曲线
  fig_loss_cwru.png      CWRU 训练损失曲线
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = "D:/EsAg"

# 大字体 + 高清全局设置（中文用微软雅黑/黑体，缺失时回退 DejaVu Sans）
rcParams.update({
    'font.size': 15,
    'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
    'axes.unicode_minus': False,
    'axes.titlesize': 17,
    'axes.labelsize': 15,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})


def fig_baseline_bar():
    b = json.load(open(os.path.join(HERE, "results_baseline_compare_smallcnn.json"), encoding="utf-8"))
    r = json.load(open(os.path.join(HERE, "results_baseline_recent.json"), encoding="utf-8"))
    names = ["ESZSL", "SJE", "ALE", "SAE", "DAP", "VAE", "GCN", "本文(端到端)"]
    acc = ([b["mean"]["ESZSL"], b["mean"]["SJE"], b["mean"]["ALE"], b["mean"]["SAE"],
            b["mean"]["DAP"]] +
           [r["mean"]["VAE"], r["mean"]["GCN"], 0.6778])
    acc = [a * 100 for a in acc]
    colors = ["#b0b0b0"] * 5 + ["#7fb3d5", "#85c1e9", "#e67e22"]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    bars = ax.bar(names, acc, color=colors, width=0.62)
    for bar, v in zip(bars, acc):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.6, f"{v:.2f}",
                ha="center", va="bottom", fontsize=13)
    ax.axhline(50.0, ls="--", color="red", lw=1.4, label="随机水平 50%")
    ax.set_ylabel("5 对平均准确率 (%)")
    ax.set_ylim(0, 78)
    ax.set_title("BaiduGear 共享冻结骨干下的零样本对比")
    ax.legend(loc="upper left")
    plt.savefig(os.path.join(OUT, "fig_baseline_bar.png"))
    plt.close()
    print("[fig] fig_baseline_bar.png")


def fig_confusion():
    d = np.load(os.path.join(HERE, "cwru_predictions.npz"))
    preds = d["preds"]      # (n_seeds, n_unseen)
    true = d["true"]
    labels = ["IR021", "B021", "OR021"]
    C = np.zeros((3, 3), dtype=int)
    for p, t in zip(preds, true):
        for pi, ti in zip(p, t):
            C[int(ti), int(pi)] += 1
    C = C.T  # 行=预测, 列=真值  -> 标准混淆矩阵行=真值列=预测
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    im = ax.imshow(C, cmap="Blues")
    ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=14)
    ax.set_yticks(range(3)); ax.set_yticklabels(labels, fontsize=14)
    ax.set_xlabel("预测类别", fontsize=15)
    ax.set_ylabel("真实类别", fontsize=15)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{C[i, j]}", ha="center", va="center",
                    fontsize=18, color="white" if C[i, j] > C.max() * 0.5 else "black")
    ax.set_title("CWRU 三分类混淆矩阵（未见类，3 种子汇总）")
    fig.colorbar(im, ax=ax, fraction=0.046)
    plt.savefig(os.path.join(OUT, "fig_confusion_cwru.png"))
    plt.close()
    print("[fig] fig_confusion_cwru.png")


def fig_noise():
    r = json.load(open(os.path.join(HERE, "results_cwru_noise.json"), encoding="utf-8"))
    snr = ["clean"] + [str(s) for s in r["snr_levels"]]
    x = np.arange(len(snr))
    acc = [r["clean_three_way_mean"]] + [r["noise_three_way_mean"][str(s)] for s in r["snr_levels"]]
    std = [r["clean_three_way_std"]] + [r["noise_three_way_std"][str(s)] for s in r["snr_levels"]]
    acc = [a * 100 for a in acc]; std = [s * 100 for s in std]
    fig, ax = plt.subplots(figsize=(7.6, 5))
    ax.errorbar(x, acc, yerr=std, marker="o", ms=7, lw=2, capsize=5, color="#2874a6")
    ax.axhline(33.3, ls="--", color="red", lw=1.4, label="随机水平 33.3%")
    ax.set_xticks(x)
    ax.set_xticklabels(["无噪声"] + [f"{s} dB" for s in r["snr_levels"]])
    ax.set_ylabel("三分类准确率 (%)")
    ax.set_xlabel("测试图像高斯噪声 SNR")
    ax.set_ylim(0, 100)
    ax.set_title("CWRU 噪声鲁棒性（3 种子均值 ± 标准差）")
    ax.legend(loc="upper right")
    plt.savefig(os.path.join(OUT, "fig_noise_cwru.png"))
    plt.close()
    print("[fig] fig_noise_cwru.png")


def fig_loss():
    r = json.load(open(os.path.join(HERE, "results_cwru_noise.json"), encoding="utf-8"))
    per_seed = r["loss_per_epoch_per_seed"]
    epochs = np.arange(1, len(per_seed[0]) + 1)
    arr = np.array(per_seed)
    mean = arr.mean(0)
    std = arr.std(0)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(epochs, mean, lw=2.2, color="#2874a6", label="3 种子均值")
    ax.fill_between(epochs, mean - std, mean + std, alpha=0.25, color="#2874a6")
    ax.set_xlabel("训练轮次 (epoch)")
    ax.set_ylabel("逐样本 MSE 损失")
    ax.set_title("CWRU 训练损失曲线（逐样本属性回归）")
    ax.legend()
    plt.savefig(os.path.join(OUT, "fig_loss_cwru.png"))
    plt.close()
    print("[fig] fig_loss_cwru.png")


if __name__ == "__main__":
    fig_baseline_bar()
    fig_confusion()
    fig_noise()
    fig_loss()
    print("ALL FIGURES DONE")
