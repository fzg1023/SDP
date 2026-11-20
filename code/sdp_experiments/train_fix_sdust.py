"""
ESAG 真零样本 —— SDUST 轴承数据集（LMS Test.Lab .mat 格式），复用 v10 机制。

数据集：D:/download2024/SDUST-Dataset-main/zhoucheng
  10 类：NC（正常）+ IF/OF/RF（内圈/外圈/滚动体）× 0.2/0.4/0.6 三种严重度
  每个 .mat 含 LMS Test.Lab 结构，'Signal' 内为 (1049600, 6) float64 信号
  （6 通道：上/下轴承座 × +Z/+Y/+X），fs=25600 Hz，每文件约 41 秒。

协议（留严重度 7/3，与 CWRU 口径一致）：
  seen   = NC, IF0.2, IF0.4, OF0.2, OF0.4, RF0.2, RF0.4（7 类）
  unseen = IF0.6, OF0.6, RF0.6（3 类重度）
  严格 inductive：每类 20% holdout 仅用于【计算类属性原型（侧信息）】，
  80% 做 ZSL；训练只用 seen 类样本（逐样本 MSE 回归自身图像域 21 维特征）；
  测试 unseen 纯语义点积 argmax_c normalize(g)·normalize(p_c)。
  不早停、固定种子 [42,49,56]、不碰 unseen 标签。

通道：全部 6 通道，每通道分别作为一个样本（多视图）。
分段：signal_length=4096（0.16s），每 (文件, 通道) 取 windows_per_file=24 个
  等间隔窗口，覆盖整段 41s 记录。

用法：
  D:/Anaconda/envs/esag/python.exe -u train_fix_sdust.py --epochs 50 --seeds 3
"""
from __future__ import annotations
import os, sys, json, argparse, glob, time
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "demo"))
sys.path.insert(0, HERE)

from esag.utils import load_config, set_seed, get_device
from esag.preprocessing import SignalPreprocessor
from train_fix_v10 import image_features, CachedImageDataset, build, eval_scores

CFG = os.path.join(ROOT, "configs", "light_semantic.yaml")
SDUST_DATA = os.environ.get("SDP_SDUST_DATA", "D:/download2024/SDUST-Dataset-main/zhoucheng")

CLASS_NAMES = ["NC", "IF0.2", "IF0.4", "IF0.6", "OF0.2", "OF0.4", "OF0.6",
               "RF0.2", "RF0.4", "RF0.6"]
UNSEEN_NAMES = ["IF0.6", "OF0.6", "RF0.6"]
SAMPLE_RATE = 25600
SIGNAL_LENGTH = 4096
WINDOWS_PER_FILE = 24
N_CHANNELS = 6
ATTR_HOLDOUT = 0.2
ANNOT_SEED = 42


def load_lms_signal(path: str) -> np.ndarray:
    """从 LMS Test.Lab .mat 中抽取 (N, 6) 振动信号（float32）。"""
    d = sio.loadmat(path, squeeze_me=False, struct_as_record=False)
    rec = d['Signal'][0, 0]
    sig = [None]

    def walk(o):
        if sig[0] is not None:
            return
        if isinstance(o, np.ndarray):
            if o.dtype == object or o.dtype.names is not None:
                for x in o.ravel():
                    walk(x)
            elif o.dtype in (np.float64, np.float32):
                if o.ndim == 2 and o.shape[0] > 1000 and o.shape[1] > 1:
                    sig[0] = o.astype(np.float32)
        elif hasattr(o, '_fieldnames'):
            for f in o._fieldnames:
                walk(getattr(o, f))

    walk(rec)
    if sig[0] is None:
        raise RuntimeError(f"未能从 {path} 解析出信号数组")
    return sig[0]


def segment_evenly(signal: np.ndarray, length: int, n_win: int) -> np.ndarray:
    """把一维长信号切成 n_win 个等间隔、互不重叠的窗口，每窗口 length 点。"""
    N = len(signal)
    if N < length:
        return np.zeros((0, length), dtype=np.float32)
    starts = np.linspace(0, N - length, n_win).astype(np.int64)
    starts = np.unique(starts)
    out = np.stack([signal[s:s + length] for s in starts])
    return out.astype(np.float32)


def load_sdust(data_path, signal_length, windows_per_file, max_files_per_class=None):
    """加载 SDUST，返回 (signals, labels, class_names)。每类内 6 通道×每文件若干窗。"""
    all_signals, all_labels = [], []
    for c, cname in enumerate(CLASS_NAMES):
        files = sorted(glob.glob(os.path.join(data_path, cname, "*.mat")))
        if max_files_per_class is not None:
            files = files[:int(max_files_per_class)]
        n_win_cls = 0
        t0 = time.time()
        for f in files:
            sig2d = load_lms_signal(f)          # (1049600, 6)
            for ch in range(N_CHANNELS):
                segs = segment_evenly(sig2d[:, ch], signal_length, windows_per_file)
                if segs.size == 0:
                    continue
                all_signals.append(segs)
                all_labels.extend([c] * segs.shape[0])
                n_win_cls += segs.shape[0]
        print(f"[load] {cname:6s} files={len(files):3d} windows={n_win_cls:5d} "
              f"({time.time() - t0:.1f}s)", flush=True)

    signals = np.concatenate(all_signals, axis=0)
    labels = np.asarray(all_labels, dtype=np.int64)
    return signals, labels


def multiway_acc(scores_mat, labels, classes):
    sel = np.isin(labels.numpy(), classes)
    sub = scores_mat[sel][:, classes]
    pred = sub.argmax(1)
    true = torch.tensor([classes.index(int(x)) for x in labels[sel].tolist()])
    return float((pred == true).float().mean().item()), int(sel.sum())


def per_class_recall(scores_mat, labels, classes):
    sel = np.isin(labels.numpy(), classes)
    sub = scores_mat[sel][:, classes]
    pred = sub.argmax(1)
    out = {}
    for j, c in enumerate(classes):
        mask = labels[sel] == c
        if mask.sum() > 0:
            out[int(c)] = float((pred[mask] == j).float().mean().item())
    return out


def train_seen_verbose(model, loader, attr_t, seen_classes, device, epochs,
                       lr=5e-4, wd=1e-3):
    """逐样本 MSE 回归（实时逐 epoch 日志版），逻辑与 train_fix_v10.train_seen 一致。"""
    seen_t = torch.tensor(seen_classes, device=device, dtype=torch.long)
    params = list(model.backbone.parameters()) + list(model.attr_head.parameters())
    optimizer = AdamW(params, lr=lr, weight_decay=wd)
    warmup = 5
    scheduler = SequentialLR(
        optimizer,
        schedulers=[LinearLR(optimizer, start_factor=0.1, total_iters=warmup),
                    CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup), eta_min=1e-6)],
        milestones=[warmup])
    n_batches = len(loader)
    for ep in range(1, epochs + 1):
        model.train()
        tr = 0.0
        nb = 0
        for batch in loader:
            images = batch["image"].to(device)
            y_attr = batch["feat"].to(device)
            g = model.attr_head(model.backbone(images))
            loss = F.mse_loss(g, y_attr)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            tr += loss.item()
            nb += 1
        scheduler.step()
        print(f"    [train] epoch {ep:3d}/{epochs}  mse={tr / nb:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--drop-path", type=float, default=0.3)
    ap.add_argument("--backbone", type=str, default="small")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--windows-per-file", type=int, default=WINDOWS_PER_FILE)
    ap.add_argument("--signal-length", type=int, default=SIGNAL_LENGTH)
    ap.add_argument("--max-files-per-class", type=int, default=None)
    ap.add_argument("--results-json", type=str,
                    default=os.path.join(HERE, "results_strict_zsl_sdust.json"))
    args = ap.parse_args()

    config = load_config(CFG)
    device = get_device("auto")
    seeds = [42 + i * 7 for i in range(args.seeds)]
    print(f"[device] {device}", flush=True)

    # ---- 加载 SDUST（6 通道全部作为样本）----
    t0 = time.time()
    signals, labels = load_sdust(SDUST_DATA, args.signal_length, args.windows_per_file,
                                 max_files_per_class=args.max_files_per_class)
    n_classes = len(CLASS_NAMES)
    n2i = {n: i for i, n in enumerate(CLASS_NAMES)}
    unseen_classes = [n2i[n] for n in UNSEEN_NAMES]
    seen_classes = [i for i in range(n_classes) if i not in unseen_classes]
    print(f"[data] n={len(signals)} classes={CLASS_NAMES} ({time.time() - t0:.1f}s)",
          flush=True)
    print(f"[data] seen={[CLASS_NAMES[i] for i in seen_classes]} "
          f"unseen={[CLASS_NAMES[i] for i in unseen_classes]}", flush=True)

    # ---- 图像域物理特征（21 维，逐样本）----
    print("[feat] computing image-domain physical features (CWT)...", flush=True)
    t0 = time.time()
    feat_list = []
    n_total = len(signals)
    for i, s in enumerate(signals):
        feat_list.append(image_features(s))
        if (i + 1) % 2000 == 0 or (i + 1) == n_total:
            print(f"    [feat] {i + 1}/{n_total}  ({time.time() - t0:.0f}s)", flush=True)
    feat = np.stack(feat_list)  # (N,21)
    print(f"[feat] done {feat.shape} ({time.time() - t0:.1f}s)", flush=True)

    # ---- 20% 标定 / 80% ZSL ----
    rng = np.random.RandomState(ANNOT_SEED)
    attr_pos, zsl_pos = [], []
    for c in range(n_classes):
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
    attr_raw = np.stack([Z[attr_pos][labels[attr_pos] == i].mean(0)
                         for i in range(n_classes)])
    attr_t = torch.tensor(attr_raw, dtype=torch.float32, device=device)  # (10,21)
    feat_zsl = torch.tensor(Z[zsl_pos], dtype=torch.float32)
    print(f"[attr] {n_classes} 类原型来自 {len(attr_pos)} 个标定样本", flush=True)

    # ---- CWT 图（仅 ZSL 80%）----
    preprocessor = SignalPreprocessor(signal_length=args.signal_length,
                                      sample_rate=SAMPLE_RATE, output_size=64)
    print(f"[prep] precomputing CWT images for {len(zsl_pos)} ZSL samples...", flush=True)
    t0 = time.time()
    img_list = []
    n_zsl = len(zsl_pos)
    for k, i in enumerate(zsl_pos):
        img_list.append(preprocessor.process_single(signals[i]))
        if (k + 1) % 2000 == 0 or (k + 1) == n_zsl:
            print(f"    [prep] {k + 1}/{n_zsl}  ({time.time() - t0:.0f}s)", flush=True)
    images_all = torch.from_numpy(np.stack(img_list).astype(np.float32))
    labels_zsl = torch.tensor(labels[zsl_pos], dtype=torch.long)
    print(f"[prep] zsl images={tuple(images_all.shape)} ({time.time() - t0:.1f}s)", flush=True)

    # ---- 训练（只用 seen 7 类）----
    seen_pos = np.where(np.isin(labels[zsl_pos], seen_classes))[0]
    print(f"[train] {len(seen_pos)} seen samples, {args.epochs} epochs x {len(seeds)} seeds, "
          f"backbone={args.backbone}", flush=True)
    scores_acc = None
    for s in seeds:
        set_seed(s)
        ds = CachedImageDataset(images_all, labels_zsl, feat_zsl, seen_pos)
        train_loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=0,
                                  drop_last=True)
        model = build(config, device, attr_t.shape[1], drop_path=args.drop_path,
                      backbone=args.backbone)
        print(f"[seed {s}] training...", flush=True)
        train_seen_verbose(model, train_loader, attr_t, seen_classes, device,
                           args.epochs, lr=args.lr, wd=args.wd)
        model.eval()
        sc = eval_scores(model, images_all, attr_t, device)
        scores_acc = sc if scores_acc is None else scores_acc + sc

    scores = scores_acc / len(seeds)

    # ---- 直接匹配上界（无模型）----
    scores_raw = F.normalize(feat_zsl, p=2, dim=-1).cpu() @ \
        F.normalize(attr_t, p=2, dim=-1).cpu().t()

    Pn = F.normalize(attr_t, p=2, dim=-1).cpu()

    # 3-way unseen
    acc3, n3 = multiway_acc(scores, labels_zsl, unseen_classes)
    raw3, _ = multiway_acc(scores_raw, labels_zsl, unseen_classes)
    rec3 = per_class_recall(scores, labels_zsl, unseen_classes)
    raw_rec3 = per_class_recall(scores_raw, labels_zsl, unseen_classes)

    print("\n=== 3-way unseen（IF0.6 / OF0.6 / RF0.6）===", flush=True)
    print(f"  学习模型 3-way 准确率 = {acc3*100:.2f}% (n={n3})", flush=True)
    print(f"  直接匹配上界           = {raw3*100:.2f}%", flush=True)
    for c in unseen_classes:
        print(f"    {CLASS_NAMES[c]:6s} recall: model={rec3.get(c, 0)*100:5.2f}%  "
              f"ceiling={raw_rec3.get(c, 0)*100:5.2f}%", flush=True)

    pairs = [(unseen_classes[0], unseen_classes[1]),
             (unseen_classes[0], unseen_classes[2]),
             (unseen_classes[1], unseen_classes[2])]
    rows = []
    print("\n| unseen 对 | 属性余弦 | 模型准确率 | 直接匹配上界 |", flush=True)
    print("|-----------|----------|-----------|--------------|", flush=True)
    for a, b in pairs:
        acc, n = multiway_acc(scores, labels_zsl, [a, b])
        raw, _ = multiway_acc(scores_raw, labels_zsl, [a, b])
        dot = float((Pn[a] @ Pn[b]).item())
        rows.append({"pair": f"{CLASS_NAMES[a]}/{CLASS_NAMES[b]}", "attr_cos": round(dot, 3),
                     "unseen_accuracy": round(acc, 4), "ceiling": round(raw, 4), "n": n})
        print(f"| {CLASS_NAMES[a]}/{CLASS_NAMES[b]} | {dot:.3f} | {acc*100:.2f}% | "
              f"{raw*100:.2f}% |", flush=True)

    out = {
        "note": "ESAG 真零样本 SDUST 版（留严重度 7/3：seen 7 类 / unseen 3 重度，6 通道全用）",
        "dataset": "SDUST", "seeds": seeds, "epochs": args.epochs, "lr": args.lr,
        "wd": args.wd, "drop_path": args.drop_path, "backbone": args.backbone,
        "signal_length": args.signal_length, "windows_per_file": args.windows_per_file,
        "sample_rate": SAMPLE_RATE, "n_channels": N_CHANNELS,
        "attr_holdout": ATTR_HOLDOUT, "annot_seed": ANNOT_SEED,
        "attr_dim": int(attr_t.shape[1]), "n_total": int(len(signals)),
        "seen": [CLASS_NAMES[i] for i in seen_classes],
        "unseen": [CLASS_NAMES[i] for i in unseen_classes],
        "three_way_accuracy": round(acc3, 4), "three_way_ceiling": round(raw3, 4),
        "three_way_recall": {CLASS_NAMES[k]: round(v, 4) for k, v in rec3.items()},
        "three_way_recall_ceiling": {CLASS_NAMES[k]: round(v, 4) for k, v in raw_rec3.items()},
        "pairwise": rows,
    }
    json.dump(out, open(args.results_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n[已另存] {args.results_json}", flush=True)
    print(f"\n=== SDUST 3-way unseen 准确率 = {acc3*100:.2f}%（上界 {raw3*100:.2f}%）===",
          flush=True)


if __name__ == "__main__":
    main()
