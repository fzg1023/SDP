"""
诊断：物理可计算统计特征作为语义属性，能否真正支撑零样本（vs 现有手写属性）。

思路：现有 BAIDU_GEAR_FAULT_ATTRIBUTES 是 16 个手写数字，已被证明与数据可分性错位
（语义点积 41-66%，而数据天花板 91-99.6%）。这里测试一种「物理可计算」的属性：
直接对 1D 原始振动信号提取 16 个经典时/频域统计特征（RMS、峭度、峰值因子、波形因子、
谱重心、谱峭度、谱熵等），类别属性 = 该类样本特征均值，再按语义点积 argmax 做 2 类分类。

注意（口径透明）：
  - 本脚本用【全类样本均值】当类别属性，属 transductive 上界，仅用于证明「物理特征属性
    与数据相关、能支撑零样本」；真正方法会另做严格 inductive 的属性来源（物理先验或独立
    标注集），此处只做可行性验证。
  - 仅诊断，不写任何模型/不覆盖任何现有文件。

用法：D:/Anaconda/envs/esag/python.exe -u diag_phys_attrs.py
"""
from __future__ import annotations
import os, sys, json
import numpy as np

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


def statistical_features(x: np.ndarray) -> np.ndarray:
    """提取 16 维经典时/频域统计特征（物理可解释、跨工况通用）。"""
    x = x - np.mean(x)
    rms = np.sqrt(np.mean(x ** 2)) + 1e-12
    peak = np.max(np.abs(x)) + 1e-12
    mean_abs = np.mean(np.abs(x)) + 1e-12
    std = np.std(x) + 1e-12

    # 时域
    f_rms = rms
    f_std = std
    f_peak = peak
    f_pp = (np.max(x) - np.min(x))
    f_crest = peak / rms                      # 峰值因子
    f_shape = rms / mean_abs                  # 波形因子
    f_impulse = peak / mean_abs               # 脉冲因子
    f_margin = peak / (np.mean(np.sqrt(np.abs(x))) ** 2 + 1e-12)  # 裕度因子
    f_kurt = np.mean(x ** 4) / (std ** 4 + 1e-12)  # 峭度
    f_skew = np.mean(x ** 3) / (std ** 3 + 1e-12)  # 偏度
    f_zcr = float(np.sum(np.diff(np.sign(x)) != 0)) / len(x)  # 过零率

    # 频域（用 FFT 幅度谱）
    spec = np.abs(np.fft.rfft(x))
    spec = spec / (np.sum(spec) + 1e-12)
    freqs = np.arange(len(spec))
    f_centroid = np.sum(freqs * spec) / (np.sum(spec) + 1e-12)  # 谱重心
    f_spread = np.sqrt(np.sum(((freqs - f_centroid) ** 2) * spec) + 1e-12)  # 谱散布
    f_entropy = -np.sum(spec * np.log(spec + 1e-12))           # 谱熵
    f_flatness = np.exp(np.mean(np.log(spec + 1e-12))) / (np.mean(spec) + 1e-12)  # 谱平坦度

    return np.array([f_rms, f_std, f_peak, f_pp, f_crest, f_shape, f_impulse,
                     f_margin, f_kurt, f_skew, f_zcr, f_centroid, f_spread,
                     f_entropy, f_flatness, f_kurt / (f_crest + 1e-12)], dtype=np.float64)


def main():
    loader = BaiduGearDataset(data_path=GEAR_DATA, signal_length=2048,
                              max_files_per_class=MAX_FILES, max_segments_per_file=MAX_SEGMENTS,
                              vibration_col=0)
    signals, labels, class_indices, class_names = loader.load_data()
    n2i = {str(n): i for i, n in enumerate(class_names)}
    n_cls = len(class_names)
    print(f"[data] n_samples={len(signals)}", flush=True)

    print("[feat] extracting 16 statistical features per sample...", flush=True)
    F = np.stack([statistical_features(s) for s in signals])  # (N,16)

    # 类别属性 = 该类样本特征均值（transductive 上界），逐维 min-max 归一化到 [0,1]
    attr = np.stack([F[labels == i].mean(0) for i in range(n_cls)])  # (C,16)
    attr_n = (attr - attr.min(0)) / (attr.max(0) - attr.min(0) + 1e-12)

    # 原始属性余弦相似度（对比）
    from esag.semantic_attributes import BAIDU_GEAR_FAULT_ATTRIBUTES as BA

    def cos(a, b):
        a = a / (np.linalg.norm(a) + 1e-12)
        b = b / (np.linalg.norm(b) + 1e-12)
        return float(np.dot(a, b))

    print("\n| 不可见对 | 手写属性余弦 | 物理特征属性余弦 | 手写语义点积实测 | 物理特征语义点积准确率 |")
    print("|----------|-------------|-----------------|------------------|------------------------|")
    rows = []
    for a_name, b_name in ALL_PAIRS:
        a, b = n2i[a_name], n2i[b_name]
        sel = (labels == a) | (labels == b)
        Fs, ls = F[sel], labels[sel]
        # 逐样本特征也做同样归一化
        Fs_n = (Fs - attr.min(0)) / (attr.max(0) - attr.min(0) + 1e-12)
        q = Fs_n / (np.linalg.norm(Fs_n, axis=1, keepdims=True) + 1e-12)
        P = attr_n / (np.linalg.norm(attr_n, axis=1, keepdims=True) + 1e-12)
        Ppair = P[[a, b]]
        pred = np.argmax(q @ Ppair.T, axis=1)
        true = np.array([0 if int(x) == a else 1 for x in ls.tolist()])
        acc = float((pred == true).mean())

        ha = cos(np.array(BA[a_name]), np.array(BA[b_name]))
        fa = cos(attr_n[a], attr_n[b])
        rows.append({"pair": f"{a_name}/{b_name}", "hand_cos": round(ha, 3),
                     "phys_cos": round(fa, 3), "phys_acc": round(acc, 4)})
        print(f"| {a_name}/{b_name} | {ha:.3f} | {fa:.3f} | 见v5 | {acc * 100:.1f}%", flush=True)

    out = {"note": "物理统计特征属性(transductive上界) vs 手写属性", "rows": rows}
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "diag_phys_attrs.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\n[已另存] diag_phys_attrs.json")


if __name__ == "__main__":
    main()
