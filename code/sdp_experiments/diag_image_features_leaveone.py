"""
诊断：图像域特征（21维，完全可恢复）在【留一齿轮类型】协议下的直接匹配上界。

协议（最严格 inductive）：属性标注集 = 人字齿轮（参考件标定各故障类的物理统计属性）；
ZSL 训练/测试 = 斜齿轮 + 直齿轮（按故障类做 seen/unseen 划分）。属性来源与测试域完全
不同齿轮类型，杜绝 transductive 泄漏。仅诊断，不写模型。
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
from train_fix_v10 import image_features
from train_fix_v6 import load_with_gear

GEAR_DATA = os.environ.get("SDP_GEAR_DATA", "D:/BaiduNetdiskDownload/GearDatasets")
CLASS_NAMES = ["健康", "剥落", "断齿", "磨损", "裂纹"]
ALL_PAIRS = [("健康", "裂纹"), ("剥落", "磨损"), ("断齿", "磨损"),
             ("剥落", "裂纹"), ("剥落", "断齿")]
MAX_FILES = 50
MAX_SEGMENTS = 50
GEAR_REF = 0   # 人字齿轮 = 属性标注集


def main():
    signals, labels, gears = load_with_gear(GEAR_DATA, MAX_FILES, MAX_SEGMENTS)
    print(f"[data] n={len(signals)} per-gear: "
          f"{[(g, int((gears == g).sum())) for g in np.unique(gears)]}", flush=True)

    print("[feat] extracting 21-dim image-domain features...", flush=True)
    F = np.stack([image_features(s) for s in signals])
    print(f"[feat] dim={F.shape[1]}", flush=True)

    ref = gears == GEAR_REF
    zsl = ~ref
    mu, sd = F[ref].mean(0), F[ref].std(0) + 1e-12
    Z = (F - mu) / sd
    attr = np.stack([Z[ref][labels[ref] == i].mean(0) for i in range(5)])
    attr = attr / (np.linalg.norm(attr, axis=1, keepdims=True) + 1e-12)
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)

    n2i = {n: i for i, n in enumerate(CLASS_NAMES)}
    print("\n| 不可见对 | 属性余弦 | 直接匹配准确率(留一齿轮,21维) |")
    print("|----------|----------|-----------------------------|")
    rows = []
    for a_name, b_name in ALL_PAIRS:
        a, b = n2i[a_name], n2i[b_name]
        sel = zsl & ((labels == a) | (labels == b))
        q = Z[sel]
        Ppair = attr[[a, b]]
        pred = np.argmax(q @ Ppair.T, axis=1)
        true = np.array([0 if int(x) == a else 1 for x in labels[sel].tolist()])
        acc = float((pred == true).mean())
        ca = float(attr[a] @ attr[b])
        rows.append({"pair": f"{a_name}/{b_name}", "cos": round(ca, 3), "acc": round(acc, 4),
                     "n": int(sel.sum())})
        print(f"| {a_name}/{b_name} | {ca:.3f} | {acc * 100:.1f}% (n={int(sel.sum())})", flush=True)

    mean = float(np.mean([r["acc"] for r in rows]))
    print(f"\n均值 = {mean * 100:.1f}%")
    json.dump({"rows": rows, "mean": round(mean, 4)},
              open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "diag_image_features_leaveone.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("[已另存] diag_image_features_leaveone.json")


if __name__ == "__main__":
    main()
