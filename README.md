# SDP Zero-Shot Fault Diagnosis — Reproducible Code Release

Paper: **Physics-based inductive zero-shot fault diagnosis with limited support samples**

Method: semantic dot-product (SDP) zero-shot fault diagnosis with a lightweight small-CNN backbone (≈1.96 M parameters).

## Directory layout

```
release/
├── README.md           this file
├── code/               source code (package + experiment scripts + reproduction entry point)
│   ├── requirements.txt
│   ├── reproduce.py    reproduction entry point
│   ├── esag/           minimal dependency package (backbones / preprocessing / datasets / attribute constants / utilities)
│   ├── demo/train_gear_baidu.py      NEU-Gear gear data loader
│   ├── configs/light_semantic.yaml   configuration
│   └── sdp_experiments/   all SDP experiment scripts
├── data/README.md      raw-data notes (data itself is not redistributed; see public sources below)
└── results/            training logs (logs/) and result files (results_json/)
```

## Environment and dependencies

```bash
pip install -r code/requirements.txt
```

Core dependencies: `torch`, `numpy`, `scipy`, `PyWavelets`, `matplotlib`, `scikit-learn`.
TensorFlow is required only for the ACGAN-FG baseline.

## Data preparation

The three datasets are **not** redistributed with this release. Download them from the public sources below and point the environment variables at your local copies.

| Environment variable | Dataset | Public download | Local example path |
|---|---|---|---|
| `SDP_GEAR_DATA` | NEU-Gear gear dataset | https://mahui-neu.github.io/lab-dataset-view/ | `D:/BaiduNetdiskDownload/GearDatasets` |
| `SDP_CWRU_DATA` | CWRU bearing dataset | https://engineering.case.edu/bearingdatacenter (data files: https://engineering.case.edu/bearingdatacenter/download-data-file) | `D:/EsAg/data/cwru` |
| `SDP_SDUST_DATA` | SDUST bearing dataset | https://github.com/JRWang-SDUST/SDUST-Dataset | `D:/download2024/SDUST-Dataset-main/zhoucheng` |

The paths in the last column are the authors' local paths and are kept only as defaults; override the variables to run elsewhere. Each dataset remains subject to its own licence and citation requirements — see `data/README.md`.

## Reproducing the main results

Run from the `code/` directory:

```bash
# NEU-Gear, 10 unseen-class pairs × 5 seeds (paper: 68.95% ± 0.37%)
python reproduce.py gear10

# CWRU three-class (paper: 85.96% ± 1.81%)
python reproduce.py cwru

# SDUST cross-dataset (paper: 58.25%)
python reproduce.py sdust
```

Equivalent direct scripts:

```bash
cd code/sdp_experiments
python train_gear_10pairs_mechproto.py --backbone small --seeds 5 --epochs 50   # NEU-Gear main result
python train_fix_cwru.py --backbone small --seeds 5 --epochs 50                 # CWRU
python train_fix_sdust.py --backbone small --epochs 50                          # SDUST
```

## Other experiments

```bash
python reproduce.py baseline      # baselines (DAP/SAE/ESZSL/ALE/SJE + VAE/GCN)
python reproduce.py difficulty    # difficulty model (leave-one-pair-out over 10 pairs)
python reproduce.py noise         # noise robustness
# ablations / recent methods / leave-location-out:
python train_fix_backbone_ablation.py       # backbone ablation (Table 5)
python train_fix_v10_classmean.py           # regression-target ablation (Table 6, class-mean vs sample-wise)
python industrial_zsl_on_gear.py            # recent method: Industrial_ZSL
python acgan_fg_on_gear.py                  # recent method: ACGAN-FG (requires TensorFlow)
python aft_on_gear.py                       # recent method: AFT
python train_cwru_leave_location.py         # leave-fault-location-out (Table 8)
```

## Method-to-code map

- **Two-channel preprocessing** (envelope spectrum + CWT) → `esag/preprocessing.py` (`SignalPreprocessor`)
- **21-dimensional image-domain physical attributes** (CWT sub-band energy 8 + scale-marginal shape 5 + envelope-spectrum shape 8) → `image_features()` in `sdp_experiments/train_fix_v10.py`
- **Lightweight convolutional backbone** (small CNN, ≈1.96 M) → `esag/small_backbone.py`
- **Sample-wise MSE regression** (linear projector 768 → 21) → `train_seen()` in `train_fix_v10.py`
- **Semantic dot-product prototype matching** → `eval_scores()` in `train_fix_v10.py`

## Notes

- The generator / discriminator / semantic-alignment / KNN modules of the earlier ESAG code have been removed from `esag/`, so the package now matches the SDP method described in the paper.
- Trained checkpoints are not included (≈915 MB). Retrain with the scripts above, or point the loader at your own checkpoint directory.
