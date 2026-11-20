# -*- coding: utf-8 -*-
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
feat = np.stack([image_features(s) for s in signals])

names = [f"band{i}" for i in range(8)] + ["c_centroid","c_spread","c_skew","c_kurt","c_entropy",
        "e_centroid","e_spread","e_skew","e_kurt","e_entropy","e_flatness","e_peak_freq","e_peak_amp"]
lines = []
lines.append(f"classes={class_names} n={len(signals)}")
lines.append("dim | " + " | ".join(f"{c}" for c in class_names))
class_means = []
for c in range(5):
    m = feat[labels == c].mean(0)
    class_means.append(m)
class_means = np.array(class_means)
for d in range(21):
    lines.append(f"{names[d]:<12} | " + " | ".join(f"{class_means[c,d]:.4f}" for c in range(5)))
# 健康 vs 故障均值差异方向（z-score 后）
mu = feat.mean(0); sd = feat.std(0) + 1e-12
cm_z = (class_means - mu) / sd
lines.append("")
lines.append("=== z-score 类均值（健康 vs 故障均值）===")
fault_mean = cm_z[1:].mean(0)
for d in range(21):
    lines.append(f"{names[d]:<12} | 健康={cm_z[0,d]:+.3f} | 故障均值={fault_mean[d]:+.3f} | 差={cm_z[0,d]-fault_mean[d]:+.3f}")
open("esag_fix/diag_mech_proto2.txt", "w", encoding="utf-8").write("\n".join(lines))
print("done")
