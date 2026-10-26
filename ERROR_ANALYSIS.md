# DSA5206 — prediction-only error analysis

Branch: `error-analysis`, based on `eva-x-comparison`. This branch contains the prediction-only implementation and tests; real-data findings remain unexecuted. All existing notebooks, helpers, figures, weights and model results remain unchanged. This analysis loads **predictions only**: it never loads a backbone/checkpoint, extracts features, refits coefficients, trains a model or selects hyperparameters. It adds no fourth model.

## Verified real-data inputs

The reviewed code in `EVA_X_Comparison.ipynb` exports:

```text
predictions.to_csv(OUTPUT_DIR/'three_model_heldout_predictions.csv', index=False)
```

Thus the correct filename is **`three_model_heldout_predictions.csv`**, not `all_three_model_heldout_predictions.csv`. The analysis reads the actual source export expression again at runtime and requires an unambiguous match; it never guesses a similarly named input or substitutes summary means.

Default input folder: `/content/duong-eva-x-results/`. Required files:

1. `three_model_heldout_predictions.csv`, containing `model,target,seed,row_index,filename,patientid,y_true,y_pred`.
2. `patient_split_audit.json`, the original 50-split audit exported by the EVA-X comparison. This is required to verify previously audited membership; the analysis does not invent an audit from the observed rows alone.

Also supply the original `reproducibility_manifest.json` when available. Its ordered filenames, patient IDs and target labels must match, with 94 images and 65 patients. If the manifest is absent, the analysis explicitly records incomplete source-provenance verification and uses the required audit's identity/membership information. It does not claim that observed row counts alone prove historical cohort identity. Without either audit patient-ID lists or a full ordered source manifest, patient-separation validation cannot proceed.

No historical images, metadata CSVs, pretrained weights, feature arrays or GPU are needed for this prediction-only workflow. **Real prediction/audit files are absent from the development workspace**, so no actual patient ranking, improvement proportion, clinical error conclusion or study figure has been generated here.

## Strict validation

Require exactly these identifiers:

- `DenseNet Lung Opacity + OLS`
- `ResNet18 + Ridge`
- `EVA-X-Tiny + Ridge`

Require exactly `Geographic Extent` and `Opacity`, seeds 0–49 for every model/target, finite predictions and reference scores, valid score ranges, nonnegative integer sample indices/seeds, and no duplicates for `(model,target,seed,row_index)`. Validate exact true-label equality across model predictions **and across repeat appearances**, stable row/filename/patient mappings, and complete three-model pairing. Source IDs retain their exact string representation; identity mismatches are not repaired by truncation, recoding or dropping cases.

Compare every observed target/seed test-index set with the original audit. Require the same full audited cohort across seeds, disjoint train/test rows and patients, and consistent audited row-to-patient mapping. Check filename lists too when supplied. Full manifest identity/labels are checked whenever present. An image is not assumed to occur in every split. Images/patients never observed in a test set are reported as uncovered, not assigned zero errors or fabricated predictions. A partial paired observation, including an image missing from all three models for an audited test set, raises an error.

Input hashes and source-manifest consistency support reproducibility but do not independently authenticate model training or rule out upstream pretraining overlap. This notebook reports provenance from the supplied artifacts rather than reloading or retraining models to certify them.

## Estimands and weighting

For each actual held-out prediction:

```text
signed_error   = y_pred - y_true
absolute_error = abs(signed_error)
squared_error  = signed_error ** 2
```

Match on target/seed/sample identity before computing `AE_EVA − AE_DenseNet` and `AE_EVA − AE_ResNet18`. Negative means lower EVA error on that particular held-out observation. Never pair different seeds or different images.

**Image unit:** for each unique target/image/model, average only its actual held-out appearances. Export number of appearances, mean signed error, mean and median absolute error, sample SD of signed error (ddof=1; N/A with one appearance), mean prediction and true score. Image paired differences are means of matched per-seed AE differences. `mean(abs(error))` is not `abs(mean(error))`.

**Patient unit:** first compute each distinct image's mean held-out errors; then average those image-level values with equal image weights within each patient. More held-out appearances do not give an image more patient weight. Patient paired differences use the same equal-image aggregation. Report total scored images, analyzed images, per-image appearance counts and their min/max/mean. Patients without held-out observations do not enter a falsely complete error table. The improvement denominator is the number of analyzed patients, with lower/equal-error counts reported separately.

**Bias rates:** calculate each image's fraction of actual held-out errors >0 (over), <0 (under) or ==0. Image-weighted rates average these fractions equally over images. Patient-weighted rates first average image fractions within patient, then weight patients equally. Separately report the proportions of unit-mean signed errors that are positive/negative/zero. This distinction prevents a zero mean error from hiding alternating over/under predictions. Mean and median signed-error summaries operate on the image or patient mean signed errors; MAE summaries average the corresponding unit MAEs.

Removing repeat appearances from the plotted unit count does **not** create independent patients: overlapping training sets still couple summaries. No naive p-values, independent-sample confidence intervals or 50-independent-replication claims are produced.

## Prespecified strata, rankings and cases

Severity groups are **descriptive score strata**, not established clinical categories:

| Target | S1 | S2 | S3 |
|---|---|---|---|
| Geographic Extent | [0,8/3] | (8/3,16/3] | (16/3,8] |
| Opacity | [0,2] | (2,4] | (4,6] |

Use each image once after aggregation. Each model/target/stratum reports distinct images, distinct patients, MAE and mean signed error. Groups with fewer than five images **or** patients are flagged small; empty strata retain N/A, never zero-error imputation. A patient with several images may span strata, so stratum patient counts are not disjoint. Trend lines connect fixed-stratum descriptive means and fit no regression model.

Image rankings are target-specific: largest strict EVA improvement/deterioration relative to DenseNet, lowest EVA image MAE, highest EVA image MAE. These are descriptive rankings of repeatedly held-out images.

Select one illustrative image per target/criterion before considering identity, tie-breaking by smallest original row index:

- A: most negative strict mean matched EVA−DenseNet AE difference.
- B: most positive strict mean matched EVA−DenseNet AE difference.
- C: largest positive EVA mean signed error.
- D: most negative EVA mean signed error.
- E: all three image mean AEs ≥1 score unit; maximize the smallest model MAE. The one-unit threshold is descriptive, not clinically validated.

Cases can recur across criteria. No eligible case yields an explicit unavailable entry; no substitute or in-sample prediction is used. Report all three models' mean predictions, mean signed and absolute errors, true score, appearance counts and both paired differences. Cases are deliberately selected extrema, not representative clinical evidence. Numerical panels are produced without inventing X-rays or making lesion-localization claims.

## Outputs and privacy

Default new folder: **`/content/duong-error-analysis-results/`**. It must not already exist and must be outside the input/results folder and repository. Reruns require an explicit new folder, e.g. `/content/duong-error-analysis-results/run2`; no existing result folder is overwritten.

Required exports:

- `aligned_prediction_errors.csv` (long paired prediction/error records with safe study IDs)
- `image_level_errors.csv`
- `patient_level_errors.csv`
- `image_level_paired_differences.csv`
- `patient_level_paired_differences.csv`
- `signed_error_summary.csv`
- `severity_stratified_summary.csv`
- `representative_cases.csv`
- `error_analysis_manifest.json`

Additional tables: `patient_improvement_summary.csv` and `image_rankings.csv`.

All six requested figures are implemented: `image_level_paired_error.png`, `patient_level_error_comparison.png`, `signed_error_bias.png`, `severity_stratified_error.png`, `failure_case_comparison.png`, `error_analysis_summary.png`. Their units and dependence limits are stated in captions. They never display pooled repeat predictions as independent images/patients.

Public tables and figures use `C###` image and `P###` study-patient IDs. Filenames and original patient IDs are excluded from presentation exports. `private_audit/identity_mapping.csv`, `private_audit/representative_cases.csv` and a private source-manifest copy retain the internal audit link. The private directory is created with owner-only directory permissions; **do not publish it or commit it**. Pseudonyms are not a guarantee of anonymization. Public provenance records input hashes, a cohort-identity digest, selected source-model provenance, code/package versions, precise estimands, selection rules and dependence limits without dumping raw source identities.

## Exact Google Colab execution

1. Clone the published analysis branch from a Colab setup notebook:

   ```python
   !git clone --branch error-analysis https://github.com/ZIYINGLAN/nus_DSA5206_GROUPPROJECT.git /content/nus_DSA5206_GROUPPROJECT
   ```

2. Open `Error_Analysis.ipynb` from this branch in Colab, or download it and use File → Upload notebook. The cloned checkout supplies `error_analysis_methods.py` and the unchanged source notebook; no manual helper upload is required.
3. Run the lightweight install cell (NumPy 1.26.4, pandas 2.2.3, matplotlib 3.9.2); restart if requested. No checkpoint or model dependency is required for the new analysis.
4. Mount Drive with the optional mount lines if previous predictions are stored there. Set `DUONG_EVA_RESULTS_DIR` to the real result folder containing the prediction CSV, audit and optional source manifest. Default is `/content/duong-eva-x-results/`. Use `DUONG_REPO_DIR` for a different checkout location.
5. Leave `DUONG_ERROR_RESULTS_DIR` unset for the default new output folder, or set a new nonexisting location. Do not direct it into the previous model results folder. Run the configuration and validation cells. Missing input files, inconsistent labels/pairing/membership or manifest mismatches stop execution.
6. Run the aggregation, presentation, and export cells in order. Inspect coverage, stratum counts and explicit unavailable cases before interpreting the six figures. No synthetic clinical results are used as a fallback.
7. Download/copy the presentation files to Drive if needed; `/content` is ephemeral. Keep the private audit folder secure and separate from presentation materials. Never commit prior prediction files, clinical images, checkpoints or generated results.

## Lecture 9 assessment

- **Appropriate metrics:** signed error for direction, MAE for magnitude without cancellation, squared residuals for outliers, exact paired comparisons and explicit image/patient estimands. Zero reference scores do not create percentage-error denominators.
- **Interpretability:** target-specific failure/bias patterns, severity plots and deterministic illustrative cases describe observable predictions. They do not establish causation, fairness, model attention or pathological localization.
- **Reproducibility:** actual-export inspection, strict externally audited membership/identity checks, source hashes, private mappings, deterministic rules and recorded versions. Previous experimental artifacts remain read-only.
- **Generalizability:** 65 patients, annotation uncertainty, retrospective sampling, potential upstream overlap and repeated shared training sets limit inference. Descriptive improvement across patients does not establish independent-patient significance, external validity or clinical readiness. A separate external/prospective design would be needed.

## Validation status

Real predictions and audit are absent locally. Only synthetic unit/integration checks are feasible; no real clinical figures, values, patient rankings or superiority claims are delivered. Synthetic figure exports are watermarked and generated only in temporary test directories, which are removed.

The complete suite includes earlier model tests (which use isolated test fixtures, not this notebook's runtime). Enable the optional official-source tests to avoid skips:

```bash
DUONG_EVA_OFFICIAL_SOURCE=/path/to/reviewed/eva_x.py python -m unittest discover -s tests -v
```

The prediction-only tests can also run separately with only their scientific dependencies plus the existing split helper's scikit-learn dependency:

```bash
python -m unittest discover -s tests -p test_error_analysis_methods.py -v
```

Notebook schema/syntax, source export selectors and byte-for-byte preservation of all inherited files are checked separately. **Executed result: 69/69 tests passed with no skips** on Python 3.11 (48 inherited tests and 21 new analysis tests). This includes the complete synthetic export path and all six watermarked plots, unequal-appearance and equal-image aggregation checks, never-held-out coverage, deterministic ties, strict identity/membership validation, empty strata/case categories and privacy-safe public exports. Notebook schema/code syntax and byte-for-byte preservation of every inherited file also passed. These results validate code paths, not clinical findings from missing real predictions.
