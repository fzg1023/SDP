"""诊断 ESZSL/ALE/SJE 超参敏感性（1 对 1 seed），确认 baseline 用合理超参而非被调烂。"""
import os, sys
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "demo"))
sys.path.insert(0, HERE)

from esag.utils import load_config, get_device
from esag.preprocessing import SignalPreprocessor
from demo.train_gear_baidu import BaiduGearDataset
from train_fix_v10 import image_features
from baseline_compare import (train_classifier_backbone, extract_features,
                              run_eszsl, run_ale, run_sje, normalize)

CFG = os.path.join(ROOT, "configs", "light_semantic.yaml")
GEAR_DATA = os.environ.get("SDP_GEAR_DATA", "D:/BaiduNetdiskDownload/GearDatasets")
CLASS_NAMES = ["健康", "剥落", "断齿", "磨损", "裂纹"]


def main():
    config = load_config(CFG)
    device = get_device("auto")
    loader = BaiduGearDataset(data_path=GEAR_DATA, signal_length=2048,
                              max_files_per_class=50, max_segments_per_file=50, vibration_col=0)
    signals, labels, class_indices, class_names = loader.load_data()
    n2i = {str(n): i for i, n in enumerate(class_names)}
    print(f"[data] n={len(signals)}", flush=True)

    feat = np.stack([image_features(s) for s in signals])
    rng = np.random.RandomState(42)
    attr_pos, zsl_pos = [], []
    for c in range(5):
        cidx = np.where(labels == c)[0]
        rng.shuffle(cidx)
        n_attr = max(1, int(round(len(cidx) * 0.2)))
        attr_pos.extend(cidx[:n_attr])
        zsl_pos.extend(cidx[n_attr:])
    attr_pos = np.array(attr_pos, dtype=np.int64)
    zsl_pos = np.array(zsl_pos, dtype=np.int64)
    mu = feat[attr_pos].mean(0); sd = feat[attr_pos].std(0) + 1e-12
    Z = (feat - mu) / sd
    attr_raw = np.stack([Z[attr_pos][labels[attr_pos] == i].mean(0) for i in range(5)])
    attr_t = torch.tensor(attr_raw, dtype=torch.float32, device=device)

    preprocessor = SignalPreprocessor(signal_length=2048, sample_rate=15360, output_size=64)
    images_all = torch.from_numpy(
        np.stack([preprocessor.process_single(signals[i]) for i in zsl_pos]).astype(np.float32))
    labels_zsl = torch.tensor(labels[zsl_pos], dtype=torch.long)

    a_name, b_name = "健康", "裂纹"
    a, b = n2i[a_name], n2i[b_name]
    seen_names = [n for n in CLASS_NAMES if n not in (a_name, b_name)]
    seen_classes = [n2i[n] for n in seen_names]
    seen_pos = np.where(np.isin(labels[zsl_pos], seen_classes))[0]
    seen_global = sorted(set(labels_zsl[seen_pos].tolist()))
    cls_map = {int(c): i for i, c in enumerate(seen_global)}
    Y_train = np.zeros((len(seen_pos), len(seen_global)), dtype=np.float32)
    for i, sp in enumerate(seen_pos):
        Y_train[i, cls_map[int(labels_zsl[sp])]] = 1.0
    Y_train = torch.tensor(Y_train, device=device)
    attr_seen_ordered = attr_t[torch.tensor(seen_global, device=device)]

    sel = (labels_zsl == a) | (labels_zsl == b)
    true = torch.tensor([0 if int(x) == a else 1 for x in labels_zsl[sel].tolist()], device=device)
    seen_test_sel = np.isin(labels_zsl[seen_pos], seen_global)

    print("[train] classifier backbone (seed 42, 50 ep)...", flush=True)
    backbone = train_classifier_backbone(config, device, images_all, labels_zsl, seen_pos, 42, epochs=50)
    X_all = extract_features(backbone, images_all, device).to(device)
    X_all = F.normalize(X_all, p=2, dim=-1)
    X_train = X_all[torch.tensor(seen_pos)]
    X_test = X_all[sel]
    print(f"[feat] X_train={X_train.shape} X_test={X_test.shape}", flush=True)

    # --- ESZSL 扫 gamma ---
    print("\n=== ESZSL (sweep gamma, lam=1e-3) ===")
    for gamma in [1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]:
        sc = run_eszsl(X_train, Y_train, X_test, attr_seen_ordered, attr_t, gamma=gamma)
        unseen = float((sc[:, [a, b]].argmax(1) == true).float().mean().item())
        # seen 类准确率（seen 3 类，多分类）
        sc_seen = run_eszsl(X_train, Y_train, X_train, attr_seen_ordered, attr_t, gamma=gamma)
        seen_lab = torch.tensor([cls_map[int(x)] for x in labels_zsl[seen_pos].tolist()], device=device)
        seen_acc = float((sc_seen[:, torch.tensor(seen_global)].argmax(1) == seen_lab).float().mean().item())
        print(f"  gamma={gamma:>6}: unseen={unseen*100:5.2f}%  seen(3cls)={seen_acc*100:5.2f}%", flush=True)

    # --- ALE / SJE 扫 margin & steps ---
    print("\n=== ALE (sweep margin/steps) ===")
    for margin in [0.1, 0.5, 1.0]:
        for steps in [200, 1000]:
            sc = run_ale(X_train, Y_train, X_test, attr_seen_ordered, attr_t, margin=margin, steps=steps)
            unseen = float((sc[:, [a, b]].argmax(1) == true).float().mean().item())
            print(f"  margin={margin} steps={steps}: unseen={unseen*100:5.2f}%", flush=True)

    print("\n=== SJE (sweep margin/steps) ===")
    for margin in [0.1, 0.5, 1.0]:
        for steps in [200, 1000]:
            sc = run_sje(X_train, Y_train, X_test, attr_seen_ordered, attr_t, margin=margin, steps=steps)
            unseen = float((sc[:, [a, b]].argmax(1) == true).float().mean().item())
            print(f"  margin={margin} steps={steps}: unseen={unseen*100:5.2f}%", flush=True)


if __name__ == "__main__":
    main()
