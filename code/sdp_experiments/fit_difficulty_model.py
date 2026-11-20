"""
P0-4：把「训练前预测难度」做实 —— 拟合 acc ≈ f(cos, σ_C) 线性模型 + 留一验证。

思路：
  难度由两个【训练前可得的几何量】决定（都来自 21 维数据派生属性的确定性计算，不依赖任何
  CNN 训练结果）：
    f1 = cos(u,v)        未见类属性原型夹角（-1..1，越接近 0 越难分）
    f2 = 1 - σ_C         类间分离度，σ_C ∈ [0,1] 为类内离散度占比
        σ_C = (r_u + r_v)/2 / ((r_u + r_v)/2 + d_uv)
        r_c = 类 c 样本到其原型 p_c 的平均欧氏距离（z-score 属性空间）
        d_uv = ||p_u - p_v|| 两原型距离
  acc = β0 + β1·cos + β2·(1 - σ_C)

留一验证：5 个齿轮对，每次留 1 对，在其余 4 对上拟合，预测留出对。
输出：全量系数、全量 R²、留一逐对预测 vs 真实、留一 MAE/RMSE、样本量。

结果另存 results_difficulty_model.json。仅 numpy，可在 esag 环境跑。
"""
from __future__ import annotations
import os, sys, json
import numpy as np
from scipy.signal import hilbert
import pywt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "demo"))

from demo.train_gear_baidu import BaiduGearDataset
from train_fix_v10 import image_features   # 21 维图像域属性（确定性）

GEAR_DATA = os.environ.get("SDP_GEAR_DATA", "D:/BaiduNetdiskDownload/GearDatasets")
CLASS_NAMES = ["健康", "剥落", "断齿", "磨损", "裂纹"]
ALL_PAIRS = [("健康", "裂纹"), ("剥落", "磨损"), ("断齿", "磨损"),
             ("剥落", "裂纹"), ("剥落", "断齿")]
MAX_FILES = 50
MAX_SEGMENTS = 50
ATTR_HOLDOUT = 0.2
ANNOT_SEED = 42

# 5-seed 逐对准确率（来自 results_gear_multiseed.json，与表 1 一致）
ACC_5SEED = {
    "健康/裂纹": 0.6258,
    "剥落/磨损": 0.6072,
    "断齿/磨损": 0.8294,
    "剥落/裂纹": 0.6239,
    "剥落/断齿": 0.7019,
}


def main():
    loader = BaiduGearDataset(data_path=GEAR_DATA, signal_length=2048,
                              max_files_per_class=MAX_FILES, max_segments_per_file=MAX_SEGMENTS,
                              vibration_col=0)
    signals, labels, class_indices, class_names = loader.load_data()
    n2i = {str(n): i for i, n in enumerate(class_names)}
    print(f"[data] n={len(signals)} classes={class_names}", flush=True)

    feat = np.stack([image_features(s) for s in signals])   # (N,21)
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

    mu = feat[attr_pos].mean(0)
    sd = feat[attr_pos].std(0) + 1e-12
    Z = (feat - mu) / sd
    proto = np.stack([Z[attr_pos][labels[attr_pos] == i].mean(0) for i in range(5)])  # (5,21)

    # 每类类内半径（全部样本到原型平均距离）
    radius = np.zeros(5)
    n_class = np.zeros(5, dtype=np.int64)
    for c in range(5):
        cidx = np.where(labels == c)[0]
        n_class[c] = len(cidx)
        radius[c] = np.mean(np.linalg.norm(Z[cidx] - proto[c], axis=1))

    rows = []
    for a_name, b_name in ALL_PAIRS:
        a, b = n2i[a_name], n2i[b_name]
        d_uv = float(np.linalg.norm(proto[a] - proto[b]))
        cos_uv = float(proto[a] @ proto[b] / (np.linalg.norm(proto[a]) * np.linalg.norm(proto[b])))
        r_mean = 0.5 * (radius[a] + radius[b])
        sigma_c = r_mean / (r_mean + d_uv)          # ∈ [0,1]，1 = 完全重叠
        sep = 1.0 - sigma_c                         # 类间分离度，越高越易分
        acc = ACC_5SEED[f"{a_name}/{b_name}"]
        n_samples = int(n_class[a] + n_class[b])
        rows.append({"pair": f"{a_name}/{b_name}", "cos": round(cos_uv, 4),
                     "sigma_c": round(sigma_c, 4), "sep_1msigma": round(sep, 4),
                     "acc": acc, "n_samples": n_samples,
                     "r_u": round(float(radius[a]), 4), "r_v": round(float(radius[b]), 4),
                     "d_uv": round(d_uv, 4)})
        print(f"[{a_name}/{b_name}] cos={cos_uv:+.3f} σ_C={sigma_c:.3f} 1-σ_C={sep:.3f} "
              f"acc={acc * 100:.2f}% n={n_samples}", flush=True)

    X = np.array([[r["cos"], r["sep_1msigma"]] for r in rows])   # (5,2)
    y = np.array([r["acc"] for r in rows])                       # (5,)
    ones = np.ones((5, 1))
    Xf = np.concatenate([ones, X], axis=1)                       # 含截距

    # 全量 OLS
    beta, _, _, _ = np.linalg.lstsq(Xf, y, rcond=None)
    yhat_full = Xf @ beta
    ss_res = float(np.sum((y - yhat_full) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2_full = 1.0 - ss_res / ss_tot

    # 留一验证
    loo_pred = np.zeros(5)
    for i in range(5):
        mask = np.ones(5, dtype=bool)
        mask[i] = False
        b, _, _, _ = np.linalg.lstsq(Xf[mask], y[mask], rcond=None)
        loo_pred[i] = float(Xf[i] @ b)
    loo_resid = loo_pred - y
    loo_mae = float(np.mean(np.abs(loo_resid)))
    loo_rmse = float(np.sqrt(np.mean(loo_resid ** 2)))
    ss_res_loo = float(np.sum(loo_resid ** 2))
    r2_loo = 1.0 - ss_res_loo / ss_tot

    print("\n=== 线性模型 acc = β0 + β1·cos + β2·(1-σ_C) ===")
    print(f"β0={beta[0]:.4f}  β1(cos)={beta[1]:.4f}  β2(1-σ_C)={beta[2]:.4f}")
    print(f"全量 R² = {r2_full:.4f}")
    print("\n留一验证（逐对 预测 vs 真实）：")
    for i, r in enumerate(rows):
        print(f"  {r['pair']}: 预测 {loo_pred[i] * 100:.2f}%  真实 {r['acc'] * 100:.2f}%  "
              f"误差 {loo_resid[i] * 100:+.2f}pp")
    print(f"留一 MAE = {loo_mae * 100:.2f}pp   RMSE = {loo_rmse * 100:.2f}pp   R²(LOO) = {r2_loo:.4f}")
    print(f"样本量：n = 5 对（留二类协议）；每对未见类样本数见 rows.n_samples")

    out = {"note": "训练前难度模型 acc=β0+β1·cos+β2·(1-σ_C)，齿轮 5 对留一验证",
           "beta": [round(float(b), 4) for b in beta],
           "r2_full": round(r2_full, 4),
           "loo": {"pred": [round(float(p), 4) for p in loo_pred],
                   "mae": round(loo_mae, 4), "rmse": round(loo_rmse, 4), "r2": round(r2_loo, 4)},
           "rows": rows}
    json.dump(out, open(os.path.join(HERE, "results_difficulty_model.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n[已另存] results_difficulty_model.json")


if __name__ == "__main__":
    main()
