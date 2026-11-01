# POLARIS: calibrated vCDR intervals from a harmonic shape ensemble

Code for every number, table and figure in the paper and its supplementary material.

The datasets are not redistributed. Obtain RIGA, DRIONS-DB and Chákṣu from their original sources.

## Setup

1. Install the dependencies with `pip install -r requirements.txt`. We used one 24 GB GPU. The analysis scripts also run on a CPU, but slowly.
2. Put the scripts in one directory, the **data root**, next to the datasets. Then set `export POLARIS_ROOT=/path/to/data_root` (the default is `~/POLARIS/BTP-Datasets`). Scripts run from the data root and load each other by file name, so keep the file names unchanged.
3. Place the ImageNet ResNet-34 weights (`resnet34-b627a593.pth`, from torchvision) in the data root.

Several analysis scripts re-execute parts of earlier scripts in memory (for example `final_run.py` runs the fitting part of `calib_v4.py`) instead of importing them. This guarantees that later analyses use exactly the fitted values of the main run. Do not edit the section header comments, because they mark where each part starts.

## Files

| File | Role |
| --- | --- |
| `notebooks/prep_1_initial.ipynb` | Data preparation: disc-detector labels (`yolo_data/`) and the 512 px training crops (`preprocessed_data/`) |
| `notebooks/prep_2_secondary.ipynb` | Data preparation: rater contour extraction (`processed_data_v2/`), harmonic targets (`training_targets_v2/`), exclusion registry (`exclusion_registry.json`) |
| `make_split_v2.py` | Train / validation / calibration / test split (`project_split_v2.json`) |
| `eval_testset_all_models.py` | Shared helpers: contour-to-radius conversion, harmonic basis and decoding, U-Net model classes, polar warp |
| `rn34_unet.py` | U-Net with an ImageNet ResNet-34 encoder |

The notebooks also contain exploratory cells. Only the cells that write the folders listed above are used.

## Run order

| Step | Script | What it does | Main output |
| --- | --- | --- | --- |
| 0 | notebooks, `make_split_v2.py` | Data preparation and split | folders listed above |
| 1 | `yolo_centre_fix.py` | YOLOv8n disc detector trained on the training split; re-crops validation, calibration and test images (2.6 × disc diameter, 512 px, CLAHE on green, z-score) | `outputs/yolo_runs/`, `preprocessed_data_v3/` |
| 2 | `train_phase2_v3.py` | Harmonic regressor (ResNet-34, 50 coefficients, band-weighted loss, rotation / scale / intensity augmentation). Run once for the single model (`outputs/phase2_v3b/`), then with `SEED=k OUT_NAME=phase2_ens/seedk` for k = 1 to 5 | `outputs/phase2_v3b/`, `outputs/phase2_ens/seed{1..5}/` |
| 3 | `train_baselines_v2.py` | U-Net baselines, `MODEL=plain\|attention\|deepres\|polar`, 5 seeds each | `outputs/phase8_v2/<model>/` |
| 4 | `train_fair_polar.py` | Polar U-Net trained on the harmonic model's images and targets (matched-compute check) | `outputs/phase8_fair/polar/` |
| 5 | `train_rn34_unet.py` | U-Net with ImageNet ResNet-34 encoder, POLARIS's augmentation, 5 seeds | `outputs/phase8_rn34/` |
| 6 | `train_direct.py` | Direct-regression controls: POLARIS's training script with only the head changed, `VARIANT=radii` (360 radii per structure) or `VARIANT=vcdr` (the vCDR value), 5 seeds each | `outputs/phase9_direct/<variant>/` |
| 7 | `phase34_v3.py` | Rater-style covariance and dispersion head, fitted on training images | `outputs/phase34_v3/` |
| 8 | `chaksu_probe.py`, `chaksu_rater_check.py`, `chaksu_resolve_flags.py`, `chaksu_fix_forus.py` | Chákṣu rater masks: layout check, duplicate-rater rule, Forus Expert 1 rebuilt from the drawings | `outputs/chaksu/rater_manifest.json` |
| 9 | `calib_v4.py` | Calibration analysis, all interval configurations | `outputs/calib_v4/` |
| 10 | `chaksu_transfer.py` | Zero-shot Chákṣu evaluation | `outputs/chaksu_transfer/` |
| 11 | `block4_and_figures.py` | Rater-filter check and rater-deviation correlation | `outputs/block4/`, `outputs/figures/` |
| 12 | `final_checks.py` | Near-duplicate images between test / calibration and training / validation | `outputs/final_checks/` |
| 13 | `final_run.py` | **Main results**: de-duplication, refits, Tables 1 and 2, paired tests, width-matched controls, bias, reliability | `outputs/final_run/` |
| 14 | `extras_run.py` | Supplementary S1 to S10, matched-compute rows, further paired tests | `outputs/extras_run/` |
| 15 | `eval_rn34_and_figs.py` | ResNet-34 U-Net rows and its conformal band, distance from nominal 90%, Figure 2 | `outputs/figs_v2/` |
| 16 | `fig2_fix.py` | Confidence intervals for the U-Net comparisons, TK minus K, RIGA counts, final Figure 2(a) to (d) | `outputs/fig2_fix/` |
| 17 | `cupdisc_check.py` | Check of expert cup masks that copy the disc, coverage without them, final Figure 2(e) | `outputs/cupdisc_check/` |
| 18 | `mc_check.py`, `review_numbers.py` | Fieller vs Monte Carlo limits; centre-line vs vertical-extent vCDR; coverage per Chákṣu expert | `outputs/mc_check/`, `outputs/review_numbers/` |
| 19 | `eval_direct.py` | Direct-regression controls on both test sets, with conformal bands | `outputs/eval_direct/` |
| 20 | `fig_assets.py` | Images for the method figure | `outputs/fig_assets/` |

Steps 9 onward re-execute earlier analysis code in memory and never retrain. Run each with a log, for example:

```
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python3 final_run.py 2>&1 | tee outputs/final_run_console.log
```

## Where each reported number comes from

| Paper item | Script |
| --- | --- |
| Table 1 (segmentation), Table 2 (intervals), paired tests, width-matched controls, bias, reliability | `final_run.py` |
| ResNet-34 U-Net rows, its conformal band, distance from nominal 90% | `eval_rn34_and_figs.py` |
| Confidence intervals for U-Net vs POLARIS vCDR error, TK minus K, RIGA image counts, Figure 2(a) to (d) | `fig2_fix.py` |
| Figure 2(e), copied-mask check | `cupdisc_check.py` |
| Matched-compute rows, supplementary S1 to S10 | `extras_run.py` |
| Fieller vs Monte Carlo, centre-line vs extent vCDR, per-expert coverage | `mc_check.py`, `review_numbers.py` |
| Direct-regression controls | `eval_direct.py` |
| Rater-deviation correlation | `block4_and_figures.py` |

## Fixed settings

- Calibration on 117 RIGA calibration images; test on 94 RIGA test images and 336 Chákṣu test images, after removing near-duplicates of training images. Nothing is fitted on test images.
- Final method (TK): Σ = τ · (diag(ensemble variance) + constant rater covariance), with τ fitted per coefficient on the calibration images, and an exact Fieller interval for the ratio of vertical chords. Coverage is scored against each expert's own vertical-extent vCDR.
- Bootstrap over images with 5,000 resamples; sign-flip permutation tests with 10,000 flips; Holm correction where stated in the logs.
