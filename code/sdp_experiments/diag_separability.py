"""
ESAG 零样本失败根因诊断：数据可分性天花板 vs 属性空间可分性。

问题：v5（swin_lite + 余弦方向回归 + 100 epoch）训练在 seen 类上完全收敛（CE~0.002、
reg~0.0003），但 unseen 准确率停在 ~41-66%。要区分两种根因：
  (A) 数据本身不可分（CWT 图对这两类故障没有可分辨信息）——那任何方法都无解；
  (B) 数据可分，但手写 16 维属性的几何与真实可分性错位——那是「语义属性设计」的锅。

诊断（纯 CPU，不训练深度网络，不涉及 ZSL 标签泄露）：
  1. 对每个不可见对，把两类 CWT 图 flatten -> 标准化 -> PCA(64) -> LogisticRegression
     （5 折分层 CV），得到「数据可分性天花板」acc_data；
  2. 同时算两类属性向量的归一化余弦相似度 cos_attr（语义点积分类器实际用的就是余弦）。
  3. 对比 acc_data 与语义点积实测 unseen 准确率：若 acc_data >> 语义准确率，
     证明瓶颈在属性几何，而非数据。

用法：D:/Anaconda/envs/esag/python.exe -u diag_separability.py
"""
from __future__ import annotations
import os, sys, json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "demo"))

from esag.preprocessing import SignalPreprocessor
from esag.semantic_attributes import BAIDU_GEAR_FAULT_ATTRIBUTES
from demo.train_gear_baidu import BaiduGearDataset

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline

GEAR_DATA = os.environ.get("SDP_GEAR_DATA", "D:/BaiduNetdiskDownload/GearDatasets")
CLASS_NAMES = ["健康", "剥落", "断齿", "磨损", "裂纹"]
ALL_PAIRS = [("健康", "裂纹"), ("剥落", "磨损"), ("断齿", "磨损"),
             ("剥落", "裂纹"), ("剥落", "断齿")]
MAX_FILES = 50
MAX_SEGMENTS = 50


def norm(v):
    n = np.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


def cos_attr(a, b):
    va = norm(BAIDU_GEAR_FAULT_ATTRIBUTES[a])
    vb = norm(BAIDU_GEAR_FAULT_ATTRIBUTES[b])
    return float(sum(x * y for x, y in zip(va, vb)))


def main():
    loader = BaiduGearDataset(data_path=GEAR_DATA, signal_length=2048,
                              max_files_per_class=MAX_FILES, max_segments_per_file=MAX_SEGMENTS,
                              vibration_col=0)
    signals, labels, class_indices, class_names = loader.load_data()
    preprocessor = SignalPreprocessor(signal_length=2048,
                                      sample_rate=int(loader.detected_sample_rate), output_size=64)
    n2i = {str(n): i for i, n in enumerate(class_names)}
    print(f"[data] n_samples={len(signals)}", flush=True)

    print("[prep] precomputing clean CWT images...", flush=True)
    images = np.stack([preprocessor.process_single(s) for s in signals])  # (N,2,64,64)
    X = images.reshape(len(images), -1).astype(np.float32)
    print(f"[prep] images={images.shape}", flush=True)

    clf = make_pipeline(StandardScaler(), PCA(n_components=64),
                        LogisticRegression(max_iter=2000, C=1.0))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\n| 不可见对 | 语义点积(原始) | 属性余弦 | 数据天花板(PCA64+LR) | 语义实测(unseen) |")
    print("|----------|---------------|----------|----------------------|------------------|")
    rows = []
    for a_name, b_name in ALL_PAIRS:
        a, b = n2i[a_name], n2i[b_name]
        sel = (labels == a) | (labels == b)
        Xs, ys = X[sel], labels[sel]
        acc = cross_val_score(clf, Xs, ys, cv=skf, n_jobs=1).mean()
        c = cos_attr(a_name, b_name)
        rows.append({"pair": f"{a_name}/{b_name}",
                     "raw_dot": round(sum(x * y for x, y in zip(
                         BAIDU_GEAR_FAULT_ATTRIBUTES[a_name], BAIDU_GEAR_FAULT_ATTRIBUTES[b_name])), 3),
                     "cos_attr": round(c, 3),
                     "data_ceiling": round(float(acc), 4),
                     "n": int(sel.sum())})
        print(f"| {a_name}/{b_name} | {rows[-1]['raw_dot']:.3f} | {c:.3f} | "
              f"{acc * 100:.1f}% | (n={rows[-1]['n']})", flush=True)

    out = {"note": "数据可分性天花板(PCA64+LR 5折CV) vs 属性几何", "rows": rows}
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "diag_separability.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\n[已另存] diag_separability.json")


if __name__ == "__main__":
    main()
