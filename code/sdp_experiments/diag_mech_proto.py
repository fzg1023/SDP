# -*- coding: utf-8 -*-
"""诊断：看 5 类的 e_peak_freq / e_centroid 实际物理值，校准机理原型假设。"""
import os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "demo"))
sys.path.insert(0, os.path.join(ROOT, "esag_fix"))
from demo.train_gear_baidu import BaiduGearDataset
from train_fix_v10 import image_features

loader = BaiduGearDataset(data_path="D:/BaiduNetdiskDownload/GearDatasets",
                          signal_length=2048, max_files_per_class=50,
                          max_segments_per_file=50, vibration_col=0)
signals, labels, class_indices, class_names = loader.load_data()
print(f"[data] n={len(signals)} classes={class_names}", flush=True)

feat = np.stack([image_features(s) for s in signals])
print("[feat] shape", feat.shape, flush=True)

# 索引：e_centroid=13, e_spread=14, e_peak_freq=19, e_peak_amp=20
print("\n类别 | e_peak_freq(归一化) | e_centroid | e_spread | e_peak_amp")
for c in range(5):
    m = feat[labels == c]
    print(f"{class_names[c]:<6} | {m[:,19].mean():.4f} | {m[:,13].mean():.4f} | {m[:,14].mean():.4f} | {m[:,20].mean():.4f}")
print("\n奈奎斯特=7680Hz，转频(1000rpm)=16.67Hz->归一化0.0022；若啮频~917Hz->0.119")
