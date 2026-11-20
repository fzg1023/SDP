# Raw Data Sources

This directory contains notes only; the raw datasets are too large to redistribute and are **not** included in this release. All three datasets are publicly available — download them from the links below and set the environment variables described in the top-level `README.md`.

> Each dataset remains subject to the licence, terms of use and citation requirements of its provider. Please cite the original dataset papers (given below) in any work that uses them.

## 1. NEU-Gear gear dataset (primary dataset)

- **Source**: gear-transmission fault dataset of Prof. Hui Ma's research group, Northeastern University, China
- **Public download**: https://mahui-neu.github.io/lab-dataset-view/
- **Environment variable**: `SDP_GEAR_DATA` (example local path: `D:/BaiduNetdiskDownload/GearDatasets`)
- **Contents**: spur, helical and herringbone gear systems; five fault conditions (healthy, spalling, broken tooth, wear, crack); sampling rate 15,360 Hz; segment length 2048; 4,321 samples in the aggregated set used here
- **Reference in paper**: [25]–[29]

## 2. CWRU bearing dataset

- **Source**: Case Western Reserve University Bearing Data Center
- **Public download**: https://engineering.case.edu/bearingdatacenter
- **Data files page**: https://engineering.case.edu/bearingdatacenter/download-data-file
- **Environment variable**: `SDP_CWRU_DATA` (example local path: `D:/EsAg/data/cwru`)
- **Contents**: ten bearing conditions — normal operation plus three fault locations (inner race IR, ball B, outer race OR) at three severities (0.007, 0.014, 0.021 in.); drive-end accelerometer data
- **Reference in paper**: [24] (Smith & Randall benchmark study)
- **Note**: the CWRU site has changed host and directory layout several times; the links above are the current official pages. A mirror of the 12 kHz drive-end subset is also available on Zenodo (record 10987113) if the original host is unreachable.

## 3. SDUST bearing dataset

- **Source**: Shandong University of Science and Technology (public GitHub repository)
- **Public download**: https://github.com/JRWang-SDUST/SDUST-Dataset
- **Environment variable**: `SDP_SDUST_DATA` (example local path: `D:/download2024/SDUST-Dataset-main/zhoucheng`)
- **Contents**: ten bearing conditions — normal operation plus three fault locations (IF/OF/RF) at three severities (0.2, 0.4, 0.6 mm); six channels; sampling rate 25.6 kHz
- **Reference in paper**: [19]

## Not used in this study

- `D:/EsAg/data/pu` — Paderborn bearing dataset (≈23 GB). Downloaded during exploration but **not used** in any experiment reported in the paper.

## Expected preprocessing

All experiments assume the following, matching Section 3.1 of the paper:

- vibration signals segmented into 2,048-point windows (CWRU and NEU-Gear);
- two-channel images of size 2 × 64 × 64 produced by the Hilbert envelope spectrum and a 64-scale Morlet CWT (see `code/esag/preprocessing.py`);
- the 21-dimensional attribute vector computed from each sample's own time–frequency image (see `image_features()` in `code/sdp_experiments/train_fix_v10.py`).
