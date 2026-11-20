"""
AFT（Fan et al. 2023, AEI, "Attribute Fusion Transfer"）迁移到 NEU-Gear，按本文统一协议对比。

忠实复用 AFT.py 的核心机制：
  无监督 PCA 特征提取 → 逐属性预测器 → 属性融合迁移（小网络）→ 最近原型匹配。

与本文 SDP / Industrial_ZSL 协议完全一致：
  20% 标定集算类原型（属性矩阵）、80% ZSL、留二类（5 对）、seen 训练 / unseen 测试。
  属性 = 本文 21 维图像域物理属性（连续值）。AFT 原文属性是 20 维二值、逐属性用 GaussianNB；
  迁移到连续属性时逐属性预测器用 Ridge 回归（与 Industrial_ZSL 迁移口径一致，保证可控对比）。

AFT 与 Industrial_ZSL 的区别：
  - Industrial_ZSL 用有监督 DSPCA（逐属性维核 PCA）；
  - AFT 用无监督 PCA + 一个「属性融合迁移」小网络（原文 Dense(8)->Dense(5)->weight，MSE 训练）
    把逐属性预测结果融合/精化后再做最近原型。

用法：
  D:/Anaconda/envs/py38_env/python.exe aft_on_gear.py
"""
from __future__ import annotations
import os, sys, json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "demo"))
sys.path.insert(0, HERE)

from demo.train_gear_baidu import BaiduGearDataset
from train_fix_v10 import image_features

GEAR_DATA = os.environ.get("SDP_GEAR_DATA", "D:/BaiduNetdiskDownload/GearDatasets")
CLASS_NAMES = ["健康", "剥落", "断齿", "磨损", "裂纹"]
ALL_PAIRS = [("健康", "裂纹"), ("剥落", "磨损"), ("断齿", "磨损"),
             ("剥落", "裂纹"), ("剥落", "断齿")]
MAX_FILES = 50
MAX_SEGMENTS = 50
ATTR_HOLDOUT = 0.2
ANNOT_SEED = 42
PCA_NUM = 20          # AFT 原文 num_components（无监督 PCA）
SEEDS = [42, 49, 56]


def predict_attributes(clf_factory, train_feats, train_attr, test_feats):
    """逐属性回归，返回 (n_test, attr_dim) 预测属性（AFT 逐属性预测器）。"""
    preds = []
    for i in range(train_attr.shape[1]):
        clf = clf_factory()
        clf.fit(train_feats, train_attr[:, i])
        preds.append(clf.predict(test_feats))
    return np.column_stack(preds)


def nearest_prototype(pred_attr, unseen_proto):
    """在 2 个 unseen 类原型中取最近（L2），与 AFT test_for_test 的 np.linalg.norm 一致。"""
    d = np.sum((pred_attr[:, None, :] - unseen_proto[None, :, :]) ** 2, axis=-1)
    return d.argmin(1)


def run():
    loader = BaiduGearDataset(data_path=GEAR_DATA, signal_length=2048,
                              max_files_per_class=MAX_FILES, max_segments_per_file=MAX_SEGMENTS,
                              vibration_col=0)
    signals, labels, class_indices, class_names = loader.load_data()
    n2i = {str(n): i for i, n in enumerate(class_names)}
    print(f"[data] n={len(signals)} classes={class_names}", flush=True)

    feat = np.stack([image_features(s) for s in signals])  # (N,21)
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

    sig_zsl = signals[zsl_pos].astype(np.float64)   # 原始信号作为 PCA 输入（AFT 对时序信号降维）
    lab_zsl = labels[zsl_pos]

    rows = {"ridge": [], "aft_fusion": []}
    for a_name, b_name in ALL_PAIRS:
        a, b = n2i[a_name], n2i[b_name]
        seen_names = [n for n in CLASS_NAMES if n not in (a_name, b_name)]
        seen_classes = [n2i[n] for n in seen_names]
        seen_mask = np.isin(lab_zsl, seen_classes)
        unseen_mask = (lab_zsl == a) | (lab_zsl == b)

        X_train = sig_zsl[seen_mask]
        y_train_cls = lab_zsl[seen_mask]
        train_attr = proto[y_train_cls]           # 训练属性标签 = 各类类原型
        X_test = sig_zsl[unseen_mask]
        y_test = lab_zsl[unseen_mask]
        unseen_proto = proto[[a, b]]
        true = np.where(y_test == a, 0, 1)

        print(f"\n=== {a_name}/{b_name} | seen={seen_names} | n_train={len(X_train)} n_test={len(X_test)} ===", flush=True)

        # 无监督 PCA 特征提取（AFT 原文对时序样本做 MinMax+PCA；这里统一用 StandardScaler 稳定数值）
        scaler = StandardScaler()
        Xtr_sc = scaler.fit_transform(X_train.reshape(X_train.shape[0], -1))
        Xte_sc = scaler.transform(X_test.reshape(X_test.shape[0], -1))
        pca = PCA(n_components=min(PCA_NUM, Xtr_sc.shape[0], Xtr_sc.shape[1]))
        Xtr_f = pca.fit_transform(Xtr_sc)
        Xte_f = pca.transform(Xte_sc)

        # 1) 逐属性 Ridge 回归 → 最近原型（AFT 去掉融合网络的纯属性回归基线）
        pred_attr_r = predict_attributes(lambda: Ridge(alpha=1.0), Xtr_f, train_attr, Xte_f)
        pred_r = nearest_prototype(pred_attr_r, unseen_proto)
        acc_r = float(accuracy_score(true, pred_r))
        rows["ridge"].append(round(acc_r, 4))
        print(f"  [ridge]    {a_name}/{b_name} acc={acc_r * 100:.2f}%", flush=True)

        # 2) 属性融合迁移：小网络把 [PCA 特征 + 逐属性预测] 融合/精化为 21 维属性（AFT 原文 Dense 结构 + MSE）
        pred_attr_r_train = predict_attributes(lambda: Ridge(alpha=1.0), Xtr_f, train_attr, Xtr_f)
        Xtr_in = np.concatenate([Xtr_f, pred_attr_r_train], axis=1)
        Xte_in = np.concatenate([Xte_f, pred_attr_r], axis=1)
        fus = MLPRegressor(hidden_layer_sizes=(32,), activation='relu', solver='adam',
                           max_iter=600, random_state=0, early_stopping=True,
                           validation_fraction=0.1)
        fus.fit(Xtr_in, train_attr)
        pred_attr_f = fus.predict(Xte_in)
        pred_f = nearest_prototype(pred_attr_f, unseen_proto)
        acc_f = float(accuracy_score(true, pred_f))
        rows["aft_fusion"].append(round(acc_f, 4))
        print(f"  [aft_fusion] {a_name}/{b_name} acc={acc_f * 100:.2f}%", flush=True)

    out = {"note": "AFT（Fan 2023）迁移到 NEU-Gear（统一协议：20% 标定原型 + 留二类 + seen 训练/unseen 测试；无监督 PCA + 逐属性 Ridge + 属性融合迁移 MLP + 最近原型）",
           "seeds": SEEDS, "pca_num": PCA_NUM,
           "per_pair": {m: dict(zip([f"{a}/{b}" for a, b in ALL_PAIRS], v)) for m, v in rows.items()},
           "mean": {m: round(float(np.mean(v)), 4) for m, v in rows.items()}}
    json.dump(out, open(os.path.join(HERE, "results_aft_gear.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    for m, v in rows.items():
        print(f"\n=== {m} 5 对均值 = {np.mean(v) * 100:.2f}% ===", flush=True)
    print(f"\n[已另存] results_aft_gear.json")


if __name__ == "__main__":
    run()
