# Frozen ResNet18 alternative comparison

Work branch: `resnet18-comparison`, based on `critical-assessment`. This branch contains source code and tests; real-data experiments remain unexecuted. Existing notebooks, figures, numerical results and helper modules remain unchanged. No end-to-end fine-tuning is implemented.

## Design fixed before outcomes

- Expected historical cohort: 94 PA images, 65 patients, Geographic Extent and Opacity expert means. Load the same metadata/score CSVs and preserve the reproduction's join/mask/row order. These counts are necessary checks, not proof of dataset identity.
- Outer splits come directly from `critical_methods.exact_splits`: GroupShuffleSplit, train/test group fractions 0.5/0.5, seeds 0–49. Save positional indices and patient IDs. The test filename/patient/target/seed/true-label keys of every available Critical Assessment feature set are checked against reconstructed indices. No split JSON is required. Mismatches halt execution.
- Primary alternative: torchvision `ResNet18_Weights.IMAGENET1K_V1`, frozen/eval, `fc=Identity`, 512-dimensional features. PIL luminance conversion followed by RGB replication; official weight transforms perform resize 256, center crop 224, ImageNet scaling/normalization. No learned feature normalization uses test data. The original DenseNet retains its own pretrained preprocessing, so this is a **pipeline comparison, not an isolated architecture ablation**.
- Ridge uses StandardScaler inside GridSearchCV, alpha in {0.01,0.1,1,10,100,1000}, minimizing inner validation MSE. GroupKFold uses up to five folds within outer training patients. Inner folds are equally weighted; within a fold MSE weights images equally. Refit only outer training rows. Solver is deterministic SVD, with one search worker.
- Optional secondary PCA+Ridge: disabled by default. Enable `INCLUDE_PCA=True` before observing outcomes. PCA is in the pipeline and fitted per inner training fold. Candidate components {2,5,10,20} are restricted to the smallest inner training sample capacity and feature count; fallback 1. Components and alpha are selected jointly by training-only CV. No test-outcome-driven selection is permitted.
- Primary DenseNet baseline: XRV `all` frozen checkpoint, pre-sigmoid Lung Opacity logit + OLS using existing validated evaluation code. Regenerate from real pixels or provenance-verified feature cache. Historical baseline predictions are reused only after exact split/identity validation, matching data/checkpoint/package/preprocessing provenance, and numerical revalidation against regenerated OLS predictions. A provenance mismatch uses regenerated predictions; a split mismatch stops the experiment. No predictions are inferred from figures.
- MAE, MSE, R² and Pearson are paired on the same observations for each target/seed. Report sample SD (ddof=1), not a standard error of independent replications. Export per-seed MAE differences (ResNet minus DenseNet; negative favours ResNet descriptively). Scatter seed **0** is prespecified. No formal significance claim is made from 50 overlapping splits.

## Execute exactly in Google Colab

1. Use a Colab Python runtime compatible with the pinned wheels (Python 3.11/3.12). CPU works; selecting GPU does not change this deliberately CPU-based notebook.
2. Create a Colab setup notebook and clone the comparison branch:

   ```python
   !git clone --branch resnet18-comparison https://github.com/ZIYINGLAN/nus_DSA5206_GROUPPROJECT.git /content/nus_DSA5206_GROUPPROJECT
   ```

   Open `ResNet18_Comparison.ipynb` from this branch in Colab (or download it and use **File → Upload notebook**). The cloned checkout supplies `resnet18_methods.py` and the unchanged `critical_methods.py`; no manual helper upload is required.
3. Run the pinned dependency installation cell. Restart if Colab requests it, then continue with the configuration cell (do not reinstall repeatedly).
4. Uncomment/run the Drive mount lines before evaluating the configured paths. Supply:

   ```text
   /content/drive/MyDrive/duong-data/images/
   /content/drive/MyDrive/duong-data/metadata_2020.csv
   /content/drive/MyDrive/duong-data/covid-severity-scores.csv
   ```

   The scores CSV uses the reproduction format: five preamble lines followed by filename/geographic_mean/opacity_mean headers. These historical files are not bundled. Preserve filenames, patient identities and annotations.
5. If the previous Critical Assessment output folder is in Drive, set `DUONG_CRITICAL_RESULTS_DIR` to that folder (containing `heldout_predictions.csv` and `provenance.json`) before the configuration cell. To reuse an existing XRV checkpoint set `DUONG_REPRO_RESULTS_DIR` to the earlier reproduction results folder. All such folders are read-only sources; checkpoints are copied into the new output folder before loading.
6. The same overrides as earlier notebooks work: `DUONG_REPO_DIR`, `DUONG_DATA_DIR`, `DUONG_IMAGE_DIR`, `DUONG_METADATA_CSV`, `DUONG_SCORES_CSV`. Edit these variables directly if easier. Leave `INCLUDE_PCA=False` for the primary experiment; optional PCA must be prespecified before examining comparisons.
7. Run remaining cells in order. Official weights download if needed. All new caches, numerical files, weights and plots are written beneath **`/content/duong-resnet18-results/`**. Subsequent runs reuse the feature cache only if provenance and checksum match. Runtime resets remove `/content`; download or copy results to Drive after completion.

## Outputs after successful real-data execution

Required outputs:

- `model_comparison_summary.csv`: mean, sample SD and defined-count per target/model/metric, with mean ± SD displayed in the notebook.
- `model_comparison_by_seed.csv`: each model/target/seed metric result.
- `resnet18_heldout_predictions.csv`: actual ResNet held-out predictions (both alternative models if PCA enabled).
- `paired_model_comparison.png`: metric distributions and per-seed paired MAE differences.
- `resnet18_prediction_scatter.png`: common seed-0 test observations, shared axis limits per target.
- `reproducibility_manifest.json`: ordered filenames, patient IDs, labels, image/data/checkpoint hashes, preprocessing/versions, experiment settings, provenance checks and baseline source.

Additional audit files: `ordered_sample_manifest.csv`, `outer_split_audit.json`, `inner_split_audit.json`, `all_model_heldout_predictions.csv`, `verified_densenet_baseline_predictions.csv`, `paired_MAE_differences.csv`, feature arrays plus integrity manifest, and timing/selected-parameter files. Extraction timings include decoding/preprocessing; regression search timings include all inner fits. Baseline regeneration timing also includes metrics/audits. They are separately labelled and **not equivalent benchmark workloads**. Feature-cache hits do not manufacture inference timings. GPU latency/memory and pretrained training cost are not measured.

## Validation and remaining limitations

**Executed:** 26 tests passed, including the 15 prior tests and 11 comparison tests. These cover 50-split record alignment, corrupted identity/membership rejection, ordered/finite features, inner patient separation, fold-local scaling, optional training-only PCA capacity/fit, test-label/test-feature invariance of selection, reference metrics, paired alignment and an end-to-end synthetic 50-split primary comparison. Shape tests exercise a frozen ResNet with finite (2,512) output without download. A separate smoke check downloads official pretrained weights and runs the real preprocessing/backbone on one synthetic grayscale image; this is functional validation, not a severity experiment.

**Static/code checks:** notebook schema and every transformed code cell's syntax; empty saved experimental outputs; preservation of inherited tracked files byte-for-byte. Review resource/provenance paths and all requested exports. No datasets or pretrained checkpoints are stored in the repository.

**Unexecuted:** real 94-image/65-patient extraction, historical held-out CSV alignment, real nested-CV model comparison, real plots and clinical conclusions. Historical images, metadata, expert scores and prior predictions are absent from this workspace. The GPU is not the blocker: CPU inference and synthetic CV work. Google Colab itself has not been run from this environment. If the historical CSV remains absent in Colab, regenerated split consistency is audited, but precise alignment with the unavailable previous execution remains unverified. No superiority, clinical utility or external validity claim is supported before appropriate experiments; even internal performance differences alone would not establish those claims.

Run tests from the repository root with the notebook's scientific dependencies installed:

```bash
python -m unittest discover -s tests -v
```
