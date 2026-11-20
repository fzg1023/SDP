# Training Logs and Results (SDP only)

Two directories:

- `logs/` — 18 training logs (stdout of the runs)
- `results_json/` — 54 result files (per-run metrics, confusion matrices, ablations, diagnostics)

## Training logs (18)

| Log | Experiment |
|---|---|
| `run_10pairs.log` | NEU-Gear 10 unseen-class pairs, sample prototypes vs mechanism prototypes (main result) |
| `v10_full_run.log`, `v10_full_run_3seed.log`, `v10_pair1.log` | NEU-Gear single-pair / single-seed debugging runs |
| `train_gear_multiseed.log` | NEU-Gear multi-seed (5-pair subset) |
| `baseline_smallcnn.log` | Classical baselines with the small CNN backbone |
| `baseline_recent.log` | Recent baselines (VAE / GCN) |
| `classmean_smallcnn.log` | Regression-target ablation (class-mean vs sample-wise) |
| `backbone_ablation_run.log` | Backbone ablation |
| `cwru_smallcnn.log` | CWRU single-seed run (debug) |
| `cwru_noise_eval.log` | CWRU noise evaluation |
| `train_cwru_multiseed.log` | CWRU multi-seed (5 seeds) |
| `train_cwru_leave_location.log`, `train_cwru_leave_location_5seed.log` | Leave-fault-location-out simulations |
| `sdust_full.log` | SDUST cross-dataset evaluation |
| `gen_tsne_cwru.log`, `gen_tsne_cwru_big.log` | t-SNE visualisation (Figure 12) |
| `ablate_similarity.log` | Similarity-measure ablation |

## Result files (54)

**Main results**

- `results_gear_multiseed.json` — NEU-Gear five-pair subset, 5 seeds
- `results_gear_10pairs_mechproto.json` — NEU-Gear ten pairs, sample vs mechanism prototypes (Table 2, Section 3.5)
- `results_cwru_multiseed.json` — CWRU three-class, 5 seeds (Table 3)
- `results_cwru_confusion.json` — CWRU aggregated confusion matrix, 5 seeds (Figure 8)
- `results_strict_zsl_cwru.json`, `results_strict_zsl_cwru_smallcnn.json` — older CWRU single-seed runs kept for traceability

**Baselines**

- `results_baseline_compare_smallcnn.json` — DAP / SAE / ESZSL / ALE / SJE / Ours with a shared frozen backbone (Table 4)
- `results_baseline_recent.json` — VAE and GCN baselines (Table 4)
- `results_e2e_baselines.json` — end-to-end trained DAP / SAE (Table 4, last three rows)
- `results_baseline_compare.json`, `results_baseline_smoke.json`, `results_baseline_smoke50.json` — pilot runs

**Ablations**

- `results_ablate_dual.json`, `results_ablate_cwt_only.json`, `results_ablate_env_only.json` — input-channel ablation (Table 7)
- `results_backbone_ablation.json`, `results_backbone_swin_lite_accavg.json`, `results_backbone_swin_accavg.json`, `results_backbone_medium_accavg.json` — backbone ablation (Table 5)
- `results_classmean_ablation.json`, `results_classmean_ablation_accavg.json`, `results_classmean_ablation_smallcnn.json` — regression-target ablation (Table 6)
- `results_ce02_smallcnn.json` — auxiliary cross-entropy weight 0.2 (Table 6)
- `results_projmlp_smallcnn.json` — deeper MLP projector (Table 6)
- `results_similarity_ablation.json` — similarity-measure ablation

**Recent methods under the unified protocol**

- `results_industrial_zsl_gear.json` (Table 10)
- `results_acgan_fg_gear.json` and `results_acgan_fg_pair0.json` … `results_acgan_fg_pair4.json` (Table 10)
- `results_aft_gear.json` (Table 10)

**Robustness and cross-dataset**

- `results_cwru_noise_signal.json` — signal-domain noise (Figure 13)
- `results_cwru_noise.json` — image-domain noise, used only as an internal comparison
- `results_strict_zsl_sdust.json` (Table 9), `results_sdust_smoke.json` (smoke test)

**Diagnostics**

- `diag_image_features*.json`, `diag_phys_attrs*.json`, `diag_recoverable_attrs.json`, `diag_rich_attrs.json`, `diag_separability.json` — attribute-construction and separability diagnostics

**Difficulty model**

- `results_difficulty_model_10pairs.json` — fitted coefficients, R², leave-one-out metrics (Section 2.7)
- `results_difficulty_model.json` — earlier five-pair fit

**Smoke / legacy runs**

- `results_strict_zsl.json`, `results_strict_zsl_v10.json`, `results_single_seed42_smallcnn.json`, `results_repro_cwru.json`, `results_repro_v10.json`

## Checkpoints

Trained weights are not included (≈915 MB). They were written to `D:/EsAg/esag_fix/checkpoints/` during the experiments; rerun the corresponding training script to regenerate them, or point the loader at your own checkpoint directory.

## Provenance notes

A few experiments were run more than once. The paper uses the files listed below; the others are kept for transparency.

1. **Regression-target ablation (Table 6).** Three files exist: `results_classmean_ablation.json` (Swin-Lite backbone, 48.09%), `results_classmean_ablation_accavg.json` (small CNN, **49.57%**, used in the paper), and `results_classmean_ablation_smallcnn.json` (small CNN, 50.12%). The latter two differ by 0.55 percentage points and come from separate runs of the same configuration.
2. **CWRU three-class accuracy.** `cwru_smallcnn.log` reports 86.91% from a single-seed debug run; the paper reports **85.96% ± 1.81%** from `results_cwru_multiseed.json` (5 seeds). Use the multi-seed file when comparing with the manuscript.
3. **Backbone ablation (Table 5).** The three Transformer rows in the paper match `results_backbone_swin_lite_accavg.json` (65.62%), `results_backbone_swin_accavg.json` (65.08%, Swin-Tiny) and `results_backbone_medium_accavg.json` (65.53%, ResNet-18). The small-CNN row in the table quotes 67.60% from the dual-channel run, whereas `results_backbone_ablation.json` reports 67.78% for the small backbone inside that same sweep.
