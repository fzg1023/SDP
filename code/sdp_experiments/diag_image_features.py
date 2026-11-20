"""
诊断：图像域特征（CWT 幅度边缘谱 + 包络谱）——完全可从骨干输入恢复，直接匹配上界。

对比 recoverable_features（含「原始 FFT 谱」统计，骨干输入里没有、只能近似恢复）。
若图像域特征的直接匹配上界 ≈ recoverable_features（65.0%），则说明换用图像域特征能
在不损失判别性的前提下让学习型骨干真正逼近上界。仅诊断，不写模型。
"""
from __future__ import annotations
import os, sys, json
import numpy as np
from scipy.signal import hilbert
import pywt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "demo"))
from demo.train_gear_baidu import BaiduGearDataset

GEAR_DATA = os.environ.get("SDP_GEAR_DATA", "D:/BaiduNetdiskDownload/GearDatasets")
CLASS_NAMES = ["健康", "剥落", "断齿", "磨损", "裂纹"]
ALL_PAIRS = [("健康", "裂纹"), ("剥落", "磨损"), ("断齿", "磨损"),
             ("剥落", "裂纹"), ("剥落", "断齿")]
MAX_FILES = 50
MAX_SEGMENTS = 50


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


def main():
    loader = BaiduGearDataset(data_path=GEAR_DATA, signal_length=2048,
                              max_files_per_class=MAX_FILES, max_segments_per_file=MAX_SEGMENTS,
                              vibration_col=0)
    signals, labels, class_indices, class_names = loader.load_data()
    n2i = {str(n): i for i, n in enumerate(class_names)}
    print(f"[data] n={len(signals)}", flush=True)

    print("[feat] extracting 21-dim image-domain features...", flush=True)
    F = np.stack([image_features(s) for s in signals])
    print(f"[feat] dim={F.shape[1]}", flush=True)

    mu, sd = F.mean(0), F.std(0) + 1e-12
    Z = (F - mu) / sd
    attr = np.stack([Z[labels == i].mean(0) for i in range(5)])
    attr = attr / (np.linalg.norm(attr, axis=1, keepdims=True) + 1e-12)
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)

    print("\n| 不可见对 | 属性余弦 | 直接匹配准确率(图像域21维) |")
    print("|----------|----------|--------------------------|")
    rows = []
    for a_name, b_name in ALL_PAIRS:
        a, b = n2i[a_name], n2i[b_name]
        sel = (labels == a) | (labels == b)
        q = Z[sel]
        Ppair = attr[[a, b]]
        pred = np.argmax(q @ Ppair.T, axis=1)
        true = np.array([0 if int(x) == a else 1 for x in labels[sel].tolist()])
        acc = float((pred == true).mean())
        ca = float(attr[a] @ attr[b])
        rows.append({"pair": f"{a_name}/{b_name}", "cos": round(ca, 3), "acc": round(acc, 4)})
        print(f"| {a_name}/{b_name} | {ca:.3f} | {acc * 100:.1f}%", flush=True)

    print(f"\n均值 = {np.mean([r['acc'] for r in rows]) * 100:.1f}%")
    json.dump({"rows": rows, "mean": float(np.mean([r["acc"] for r in rows]))},
              open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "diag_image_features.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("[已另存] diag_image_features.json")


if __name__ == "__main__":
    main()
