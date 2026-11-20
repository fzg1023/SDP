"""
诊断：22 维「可恢复」物理特征（谱形状+包络谱+CWT子带）的直接匹配上界。

对比 26 维 rich_features（含时域不可恢复量）的 65.6%，判断去掉非可恢复特征后
属性空间是否仍可分。仅诊断，不写模型。
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
from train_fix_v8 import recoverable_features

GEAR_DATA = os.environ.get("SDP_GEAR_DATA", "D:/BaiduNetdiskDownload/GearDatasets")
CLASS_NAMES = ["健康", "剥落", "断齿", "磨损", "裂纹"]
ALL_PAIRS = [("健康", "裂纹"), ("剥落", "磨损"), ("断齿", "磨损"),
             ("剥落", "裂纹"), ("剥落", "断齿")]
MAX_FILES = 50
MAX_SEGMENTS = 50


def main():
    loader = BaiduGearDataset(data_path=GEAR_DATA, signal_length=2048,
                              max_files_per_class=MAX_FILES, max_segments_per_file=MAX_SEGMENTS,
                              vibration_col=0)
    signals, labels, class_indices, class_names = loader.load_data()
    n2i = {str(n): i for i, n in enumerate(class_names)}
    print(f"[data] n={len(signals)}", flush=True)

    print("[feat] extracting 22-dim recoverable features...", flush=True)
    F = np.stack([recoverable_features(s) for s in signals])
    print(f"[feat] dim={F.shape[1]}", flush=True)

    mu, sd = F.mean(0), F.std(0) + 1e-12
    Z = (F - mu) / sd
    attr = np.stack([Z[labels == i].mean(0) for i in range(5)])
    attr = attr / (np.linalg.norm(attr, axis=1, keepdims=True) + 1e-12)
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)

    print("\n| 不可见对 | 属性余弦 | 直接匹配准确率(22维可恢复) |")
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
                                "diag_recoverable_attrs.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("[已另存] diag_recoverable_attrs.json")


if __name__ == "__main__":
    main()
