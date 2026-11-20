"""
ACGAN-FG 数据准备：加载 NEU-Gear 原始振动信号 + 计算 21 维图像域属性，
按本文 SDP 完全相同的 20%/80% 标定/ZSL 划分与属性标准化（标定集统计量），
另存 acgan_fg_gear_data.npz 供 tf-env 下的 ACGAN-FG 训练脚本直接加载。

与 SDP 唯一区别：ACGAN-FG 的「样本」= 原始信号(2048)，而非双通道图像；
「语义属性」= 同一套 21 维图像域物理属性（连续值，MSE 回归）。
"""
from __future__ import annotations
import os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "demo"))
sys.path.insert(0, HERE)

from demo.train_gear_baidu import BaiduGearDataset
from train_fix_v10 import image_features

GEAR_DATA = os.environ.get("SDP_GEAR_DATA", "D:/BaiduNetdiskDownload/GearDatasets")
MAX_FILES = 50
MAX_SEGMENTS = 50
ATTR_HOLDOUT = 0.2
ANNOT_SEED = 42
OUT = os.path.join(HERE, "acgan_fg_gear_data.npz")


def main():
    loader = BaiduGearDataset(data_path=GEAR_DATA, signal_length=2048,
                              max_files_per_class=MAX_FILES,
                              max_segments_per_file=MAX_SEGMENTS,
                              vibration_col=0)
    signals, labels, class_indices, class_names = loader.load_data()
    print(f"[data] n={len(signals)} classes={class_names}", flush=True)

    signals = signals.astype(np.float64)
    attrs = np.stack([image_features(s) for s in signals])  # (N, 21)

    rng = np.random.RandomState(ANNOT_SEED)
    attr_pos, zsl_pos = [], []
    for c in range(5):
        cidx = np.where(labels == c)[0]
        rng.shuffle(cidx)
        n_attr = max(1, int(round(len(cidx) * ATTR_HOLDOUT)))
        attr_pos.extend(cidx[:n_attr])
        zsl_pos.extend(cidx[n_attr:])
    attr_pos = np.array(attr_pos, dtype=np.int64)
    zsl_pos = np.array(zsl_pos, dtype=np.int64)

    # 属性标准化（标定集统计量，与 SDP 一致）
    a_mu = attrs[attr_pos].mean(0)
    a_sd = attrs[attr_pos].std(0) + 1e-12
    attrs_z = (attrs - a_mu) / a_sd
    proto = np.stack([attrs_z[attr_pos][labels[attr_pos] == i].mean(0)
                      for i in range(5)])  # (5, 21) 类均值原型

    # 原始信号逐维标准化（标定集统计量，autoencoder 输入）
    s_mu = signals[attr_pos].mean(0)
    s_sd = signals[attr_pos].std(0) + 1e-12
    signals_z = (signals - s_mu) / s_sd

    np.savez(OUT,
             signals_z=signals_z.astype(np.float32),
             attrs_z=attrs_z.astype(np.float32),
             labels=labels.astype(np.int64),
             attr_pos=attr_pos,
             zsl_pos=zsl_pos,
             proto=proto.astype(np.float32))
    print(f"[saved] {OUT}\n  signals_z={signals_z.shape} attrs_z={attrs_z.shape} "
          f"proto={proto.shape}", flush=True)


if __name__ == "__main__":
    main()
