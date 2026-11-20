"""
诊断：留一齿轮类型 + 物理统计特征属性，能否支撑严格 inductive 零样本。

协议（比 v5 更严格、真跨齿轮泛化）：
  - 属性标注集 = 人字齿轮（独立于 ZSL 训练/测试，等价于专家用参考件标定故障属性）；
  - ZSL 训练/测试 = 斜齿轮 + 直齿轮（按故障类别做 seen/unseen 划分）；
  - 类别属性 a_c = 人字齿轮该类样本的物理统计特征均值（min-max 归一化）；
  - 测试：斜+直齿轮的 unseen 对，直接对样本自身统计特征做余弦匹配（无学习映射的下界）。

仅诊断，不写模型、不覆盖现有文件。用法：
  D:/Anaconda/envs/esag/python.exe -u diag_phys_leaveone.py
"""
from __future__ import annotations
import os, sys, json
import numpy as np
from scipy.signal import hilbert

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


def physical_features(x: np.ndarray) -> np.ndarray:
    """~20 维物理统计特征（时域 + 频域 + 包络域）。"""
    x = x - np.mean(x)
    rms = np.sqrt(np.mean(x ** 2)) + 1e-12
    peak = np.max(np.abs(x)) + 1e-12
    mean_abs = np.mean(np.abs(x)) + 1e-12
    std = np.std(x) + 1e-12

    crest = peak / rms                       # 峰值因子
    kurt = np.mean(x ** 4) / (std ** 4 + 1e-12)  # 峭度
    skew = np.mean(x ** 3) / (std ** 3 + 1e-12)  # 偏度
    shape = rms / mean_abs                   # 波形因子
    impulse = peak / mean_abs                # 脉冲因子
    margin = peak / (np.mean(np.sqrt(np.abs(x))) ** 2 + 1e-12)  # 裕度因子
    zcr = float(np.sum(np.diff(np.sign(x)) != 0)) / len(x)
    pp = float(np.max(x) - np.min(x))

    # 频域
    spec = np.abs(np.fft.rfft(x))
    spec = spec / (np.sum(spec) + 1e-12)
    freqs = np.arange(len(spec))
    centroid = np.sum(freqs * spec)
    spread = np.sqrt(np.sum((freqs - centroid) ** 2 * spec) + 1e-12)
    entropy = -np.sum(spec * np.log(spec + 1e-12))
    flatness = np.exp(np.mean(np.log(spec + 1e-12))) / (np.mean(spec) + 1e-12)
    skurt = np.sum((freqs - centroid) ** 4 * spec) / (spread ** 4 + 1e-12)

    # 包络域（Hilbert）
    env = np.abs(hilbert(x))
    env = env - np.mean(env)
    env_kurt = np.mean(env ** 4) / (np.std(env) ** 4 + 1e-12)
    env_spec = np.abs(np.fft.rfft(env))
    env_spec = env_spec / (np.sum(env_spec) + 1e-12)
    env_ent = -np.sum(env_spec * np.log(env_spec + 1e-12))
    env_crest = (np.max(np.abs(env)) + 1e-12) / (np.sqrt(np.mean(env ** 2)) + 1e-12)

    return np.array([rms, peak, crest, kurt, skew, shape, impulse, margin, zcr, pp,
                     centroid, spread, entropy, flatness, skurt,
                     env_kurt, env_ent, env_crest], dtype=np.float64)


def load_with_gear(data_path, max_files=50, max_segments=50):
    loader = BaiduGearDataset(data_path=data_path, signal_length=2048,
                              max_files_per_class=max_files,
                              max_segments_per_file=max_segments, vibration_col=0)
    signals, labels, gears = [], [], []
    for gi, gt in enumerate(loader.GEAR_TYPES):
        for li, ft in enumerate(CLASS_NAMES):
            d = os.path.join(data_path, gt, ft)
            if not os.path.isdir(d):
                continue
            for cf in sorted([f for f in os.listdir(d) if f.endswith(".csv")])[:max_files]:
                segs, _ = loader._stream_segments_from_csv(os.path.join(d, cf))
                if segs.shape[0] > 0:
                    signals.append(segs)
                    labels.extend([li] * len(segs))
                    gears.extend([gi] * len(segs))
    return np.concatenate(signals, 0), np.array(labels), np.array(gears)


def main():
    signals, labels, gears = load_with_gear(GEAR_DATA, MAX_FILES, MAX_SEGMENTS)
    print(f"[data] n={len(signals)}  per-gear: "
          f"{[(g, int((gears == g).sum())) for g in np.unique(gears)]}", flush=True)

    F = np.stack([physical_features(s) for s in signals])  # (N,20)
    ref = gears == 0            # 人字齿轮 = 属性标注集
    zsl = ~ref                 # 斜 + 直齿轮 = ZSL 训练/测试

    # 类别属性 = 人字齿轮该类样本特征均值；逐维 min-max 归一化（用参考集统计）
    attr = np.stack([F[ref & (labels == i)].mean(0) for i in range(5)])
    lo, hi = attr.min(0), attr.max(0)
    attr_n = (attr - lo) / (hi - lo + 1e-12)

    n2i = {n: i for i, n in enumerate(CLASS_NAMES)}
    print("\n| 不可见对 | 属性余弦 | 直接特征匹配准确率(留一齿轮) |")
    print("|----------|----------|-------------------------------|")
    rows = []
    for a_name, b_name in ALL_PAIRS:
        a, b = n2i[a_name], n2i[b_name]
        sel = zsl & ((labels == a) | (labels == b))
        Fs = (F[sel] - lo) / (hi - lo + 1e-12)
        q = Fs / (np.linalg.norm(Fs, axis=1, keepdims=True) + 1e-12)
        P = attr_n / (np.linalg.norm(attr_n, axis=1, keepdims=True) + 1e-12)
        pred = np.argmax(q @ P[[a, b]].T, axis=1)
        true = np.array([0 if int(x) == a else 1 for x in labels[sel].tolist()])
        acc = float((pred == true).mean())
        ca = float(np.dot(attr_n[a] / (np.linalg.norm(attr_n[a]) + 1e-12),
                          attr_n[b] / (np.linalg.norm(attr_n[b]) + 1e-12)))
        rows.append({"pair": f"{a_name}/{b_name}", "cos": round(ca, 3), "acc": round(acc, 4),
                     "n": int(sel.sum())})
        print(f"| {a_name}/{b_name} | {ca:.3f} | {acc * 100:.1f}% (n={int(sel.sum())})", flush=True)

    mean = float(np.mean([r["acc"] for r in rows]))
    print(f"\n均值准确率 = {mean * 100:.1f}%")
    json.dump({"rows": rows, "mean": round(mean, 4)},
              open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "diag_phys_leaveone.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("[已另存] diag_phys_leaveone.json")


if __name__ == "__main__":
    main()
