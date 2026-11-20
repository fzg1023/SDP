"""
诊断：更丰富的物理特征（16→~32 维，加 CWT 子带能量 + 包络谱特征）能否提升属性可分性。

只测「直接特征匹配」的 transductive 上界（全类均值当属性），与 16 维结果对比，
判断瓶颈是否在特征维度不足。仅诊断，不写模型。

用法：D:/Anaconda/envs/esag/python.exe -u diag_rich_attrs.py
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


def rich_features(x: np.ndarray) -> np.ndarray:
    """~30 维物理特征：时域 + 频域 + 包络 + CWT 子带能量。"""
    x = x - np.mean(x)
    rms = np.sqrt(np.mean(x ** 2)) + 1e-12
    peak = np.max(np.abs(x)) + 1e-12
    mean_abs = np.mean(np.abs(x)) + 1e-12
    std = np.std(x) + 1e-12
    crest = peak / rms
    kurt = np.mean(x ** 4) / (std ** 4 + 1e-12)
    skew = np.mean(x ** 3) / (std ** 3 + 1e-12)
    shape = rms / mean_abs
    impulse = peak / mean_abs
    margin = peak / (np.mean(np.sqrt(np.abs(x))) ** 2 + 1e-12)
    zcr = float(np.sum(np.diff(np.sign(x)) != 0)) / len(x)
    pp_rms = float(np.max(x) - np.min(x)) / rms

    spec = np.abs(np.fft.rfft(x))
    spec = spec / (np.sum(spec) + 1e-12)
    freqs = np.arange(len(spec)) / len(spec)
    centroid = np.sum(freqs * spec)
    spread = np.sqrt(np.sum((freqs - centroid) ** 2 * spec) + 1e-12)
    entropy = -np.sum(spec * np.log(spec + 1e-12))
    flatness = np.exp(np.mean(np.log(spec + 1e-12))) / (np.mean(spec) + 1e-12)
    skurt = np.sum((freqs - centroid) ** 4 * spec) / (spread ** 4 + 1e-12)

    env = np.abs(hilbert(x))
    env = env - np.mean(env)
    env_kurt = np.mean(env ** 4) / (np.std(env) ** 4 + 1e-12)
    env_crest = (np.max(np.abs(env)) + 1e-12) / (np.sqrt(np.mean(env ** 2)) + 1e-12)
    env_spec = np.abs(np.fft.rfft(env))
    env_spec = env_spec / (np.sum(env_spec) + 1e-12)
    env_ent = -np.sum(env_spec * np.log(env_spec + 1e-12))
    env_peak_freq = float(np.argmax(env_spec)) / len(env_spec)
    env_peak_amp = float(np.max(env_spec))

    # CWT 子带能量（8 个频段）
    coefs, _ = pywt.cwt(x, np.arange(1, 65), 'morl')  # (64, N)
    pw = np.abs(coefs) ** 2
    bands = []
    for i in range(8):
        bands.append(float(np.sum(pw[i * 8:(i + 1) * 8])))
    bands = np.array(bands)
    bands = bands / (bands.sum() + 1e-12)

    return np.concatenate([
        [crest, kurt, skew, shape, impulse, margin, zcr, pp_rms,
         centroid, spread, skurt, entropy, flatness,
         env_kurt, env_crest, env_ent, env_peak_freq, env_peak_amp], bands])


def main():
    loader = BaiduGearDataset(data_path=GEAR_DATA, signal_length=2048,
                              max_files_per_class=MAX_FILES, max_segments_per_file=MAX_SEGMENTS,
                              vibration_col=0)
    signals, labels, class_indices, class_names = loader.load_data()
    n2i = {str(n): i for i, n in enumerate(class_names)}
    print(f"[data] n={len(signals)}", flush=True)

    print("[feat] extracting ~30-dim physical features...", flush=True)
    F = np.stack([rich_features(s) for s in signals])  # (N, 26)
    print(f"[feat] dim={F.shape[1]}", flush=True)

    # 全类均值当属性（transductive 上界），z-score 标准化
    mu, sd = F.mean(0), F.std(0) + 1e-12
    Z = (F - mu) / sd
    attr = np.stack([Z[labels == i].mean(0) for i in range(5)])
    attr = attr / (np.linalg.norm(attr, axis=1, keepdims=True) + 1e-12)
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)

    print("\n| 不可见对 | 属性余弦 | 直接匹配准确率(30维) |")
    print("|----------|----------|----------------------|")
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
                                "diag_rich_attrs.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("[已另存] diag_rich_attrs.json")


if __name__ == "__main__":
    main()
