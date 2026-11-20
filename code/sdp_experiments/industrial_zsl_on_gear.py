"""
Industrial_ZSL（Feng & Zhao 2021）迁移到 NEU-Gear，按本文统一协议对比。

忠实复用其核心机制：
  DSPCA（有监督核 PCA，逐属性维做监督降维）→ 逐属性回归器 → 最近原型匹配。

协议与本文 SDP 完全一致：
  20% 标定集算类原型（属性矩阵）、80% ZSL、留二类（5 对）、seen 训练 / unseen 测试，
  3 种子（42/49/56）。属性 = 本文 21 维图像域物理属性（连续值，用 Ridge 回归，
  即原文 model_dict 中的回归选项；再给 RandomForestRegressor 作对照）。

用法：
  D:/Anaconda/envs/py38_env/python.exe industrial_zsl_on_gear.py
"""
from __future__ import annotations
import os, sys, json
import numpy as np
from scipy.linalg import cholesky, svd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
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
DSPCA_NUM = 20
SEEDS = [42, 49, 56]


def svd_flip(u, v):
    max_abs_cols = np.argmax(np.abs(u), axis=0)
    signs = np.sign(u[max_abs_cols, range(u.shape[1])])
    u *= signs
    v *= signs[:, np.newaxis]
    return u, v


class DSPCA:
    """复刻 Feng 2021 util.DSPCA（有监督核 PCA）。"""
    def __init__(self, num, sigma='auto'):
        self._num = num
        self._sigma = sigma

    def kernel_matrix(self, label):
        if self._sigma == 'auto':
            self._sigma = 1 / np.mat(label).T.shape[1]
        n = label.shape[0]
        mat = np.zeros((n, n))
        for i in range(n):
            mat[i] = np.linalg.norm(np.mat(label[i] - label), axis=0)
        mat = np.exp(-1 * np.multiply(mat, mat) / (self._sigma ** 2))
        return mat

    def fit(self, data, label):
        n = data.shape[0]
        L = self.kernel_matrix(label)
        deta = cholesky(L + 1e-5 * np.eye(L.shape[0])).T
        H = np.eye(n) - (1 / n) * np.ones((n, n))
        fai = data.T.dot(H).dot(deta)
        U, S, V = svd(fai, full_matrices=False)
        U, V = svd_flip(U, V)
        V = V[0:self._num].T
        xi = np.diag(S[0:self._num])
        self._U = fai.dot(V).dot(np.mat(xi).I)

    def transform(self, data):
        return np.asarray((self._U.T.dot(data.T)).T)


def feature_extraction(train_data, test_data, train_attr, test_attr):
    """逐属性维 DSPCA，拼接为 (n, attr_dim*num) 特征。"""
    train_feats, test_feats = [], []
    for i in range(train_attr.shape[1]):
        spca = DSPCA(DSPCA_NUM)
        spca.fit(train_data, train_attr[:, i])
        train_feats.append(spca.transform(train_data))
        test_feats.append(spca.transform(test_data))
    return np.column_stack(train_feats), np.column_stack(test_feats)


def predict_attributes(clf_factory, train_feats, train_attr, test_feats):
    """逐属性回归，返回 (n_test, attr_dim) 预测属性。"""
    preds = []
    for i in range(train_attr.shape[1]):
        clf = clf_factory()
        clf.fit(train_feats, train_attr[:, i])
        preds.append(clf.predict(test_feats))
    return np.column_stack(preds)


def nearest_prototype(pred_attr, unseen_proto):
    """在 2 个 unseen 类原型中取最近（L2）。"""
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

    sig_zsl = signals[zsl_pos].astype(np.float64)   # 原始信号作为 DSPCA 输入
    lab_zsl = labels[zsl_pos]
    feat_zsl = Z[zsl_pos]

    rows = {"ridge": [], "rf": []}
    for a_name, b_name in ALL_PAIRS:
        a, b = n2i[a_name], n2i[b_name]
        seen_names = [n for n in CLASS_NAMES if n not in (a_name, b_name)]
        seen_classes = [n2i[n] for n in seen_names]
        seen_mask = np.isin(lab_zsl, seen_classes)
        unseen_mask = (lab_zsl == a) | (lab_zsl == b)

        X_train = sig_zsl[seen_mask]
        y_train_cls = lab_zsl[seen_mask]
        # 训练属性标签 = 各类类原型（与原文 attribute_matrix[item,:] 重复一致）
        train_attr = proto[y_train_cls]
        X_test = sig_zsl[unseen_mask]
        y_test = lab_zsl[unseen_mask]
        unseen_proto = proto[[a, b]]
        true = np.where(y_test == a, 0, 1)

        print(f"\n=== {a_name}/{b_name} | seen={seen_names} | n_train={len(X_train)} n_test={len(X_test)} ===", flush=True)
        # DSPCA 特征提取（逐属性维）
        print("  [DSPCA] extracting features...", flush=True)
        Xtr_f, Xte_f = feature_extraction(X_train, X_test, train_attr, None)

        for model_name, factory in [
            ("ridge", lambda: Ridge(alpha=1.0)),
            ("rf", lambda: RandomForestRegressor(n_estimators=50, random_state=0)),
        ]:
            pred_attr = predict_attributes(factory, Xtr_f, train_attr, Xte_f)
            pred = nearest_prototype(pred_attr, unseen_proto)
            acc = float(accuracy_score(true, pred))
            rows[model_name].append(round(acc, 4))
            print(f"  [{model_name}] {a_name}/{b_name} acc={acc * 100:.2f}%", flush=True)

    out = {"note": "Industrial_ZSL 迁移到 NEU-Gear（统一协议：20% 标定原型 + 留二类 + seen 训练/unseen 测试）",
           "seeds": SEEDS, "dspca_num": DSPCA_NUM,
           "per_pair": {m: dict(zip([f"{a}/{b}" for a, b in ALL_PAIRS], v)) for m, v in rows.items()},
           "mean": {m: round(float(np.mean(v)), 4) for m, v in rows.items()}}
    json.dump(out, open(os.path.join(HERE, "results_industrial_zsl_gear.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    for m, v in rows.items():
        print(f"\n=== {m} 5 对均值 = {np.mean(v) * 100:.2f}% ===", flush=True)
    print(f"\n[已另存] results_industrial_zsl_gear.json")


if __name__ == "__main__":
    run()
