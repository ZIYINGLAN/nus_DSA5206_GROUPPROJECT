# Critical assessment — execution and validation

Branch: `critical-assessment`, based on `improve-duong-reproduction`. No reproduction artifacts are changed. No ResNet18 is included.

## Google Colab

1. Obtain `Critical_Assessment.ipynb`, `critical_methods.py` and `duong_methods.py`. Once the branch is pushed, clone it in a Colab setup notebook:

   ```python
   !git clone --branch critical-assessment https://github.com/ZIYINGLAN/nus_DSA5206_GROUPPROJECT.git /content/nus_DSA5206_GROUPPROJECT
   ```

   Until publication, upload those three files manually, placing both `.py` files in `/content/nus_DSA5206_GROUPPROJECT/`, and open the notebook in Colab. This local task does not itself imply a GitHub push.
2. Run the notebook's pinned dependency-installation cell. Restart the runtime if Colab requests it, then continue with configuration. CPU is supported; a GPU is unnecessary.
3. Mount Drive using the optional cell. Use the same data as the improved reproduction: `/content/drive/MyDrive/duong-data/images/`, `metadata_2020.csv` and `covid-severity-scores.csv` (five preamble lines before its header). These files are not bundled. Set `DUONG_REPO_DIR`, `DUONG_DATA_DIR`, `DUONG_IMAGE_DIR`, `DUONG_METADATA_CSV` or `DUONG_SCORES_CSV`, or edit the equivalent path variables, if your locations differ.
4. Run all remaining cells in order. All assessment outputs go to **`/content/duong-critical-assessment-results/`**. Download that directory or copy it to Drive after execution; `/content` is ephemeral.
5. To reuse an earlier assessment run's verified arrays, set `DUONG_CRITICAL_CACHE_DIR` to its directory. To reuse a downloaded reproduction checkpoint, set `DUONG_REPRO_RESULTS_DIR` (default `/content/duong-improved-results`). The checkpoint is copied into the new output directory; no writes go into the reproduction results. Internet access is needed if the checkpoint is absent.

## What it produces, after real-data execution

- Ordered cohort manifest, image/data/checkpoint/package fingerprints and a checksummed numerical feature cache.
- Exact train/test row indices, filenames and patient IDs for all Table 1 seeds 0–49, plus an overlap audit. Fits use training rows only; no seed/feature selection is performed using test performance.
- Individual held-out predictions and split metrics for both targets and all five Table 1 feature sets (500 fits).
- One-sided versus absolute tolerance comparison tables/plots. At error ≤−0.7 the one-sided metric falsely counts success. Strict boundaries are preserved.
- MAE/MSE/R²/Pearson distributions, seed traces and descriptive statistics. Dependent split distributions are not population confidence intervals. Undefined baseline Pearson stays missing/N/A; count=0 makes this explicit.
- Paired saliency panels on the seed-0 reproduction's deliberately selected examples, raw A/B arrays, coefficients, checkpoint hash, case filenames and scale limits. Both methods use the same score gradient; A/B use separate global scales because their units differ.
- A source-based discrepancy table and discussion within the notebook. The 199-versus-208 cause remains unresolved; no nine images are removed.

Legacy reproduction `.npy` files lack array-integrity manifests. Their presence is reported, but their values are not assumed to match the current data or checkpoint. The notebook regenerates features from real pixels when it cannot verify a compatible cache, then regenerates inexpensive OLS predictions rather than inferring them from summary tables or plots. Cached assessment arrays are reused only if their full provenance fingerprint and checksum match.

## Validation status

**Successfully executed:** 15 unit tests across both test modules. Tests cover signed/absolute threshold behaviour (including equality), misaligned/nonfinite inputs, all 50 exact grouped splits, missing patient IDs/duplicate filenames, prediction and metric equivalence to direct training-only OLS, undefined baseline Pearson, descriptive-statistic counts, and invariance of a split's predictions to changes in its held-out labels. Existing tests cover A/B saliency transformations and boundaries. All fixtures are explicitly synthetic test data, not experimental results.

**Code checks / source inspection:** notebook schema and transformed Python syntax for every code cell; no stored experimental outputs; all existing tracked notebooks, figures and numerical files compared byte-for-byte to branch HEAD. Checkpoint URL lookup was checked against installed TorchXRayVision. The four-feature representation, Table 1 split parameters, author saved 199 count and paper's 208 description were inspected. This does not resolve historical dataset identity or author loader versions.

**Not executed:** actual 94-image metric/stability experiments or real-image saliency panels, because source images, historical metadata, expert scores and cached predictions are absent from this workspace. A pretrained checkpoint is available in the onboarding environment; GPU resources are not the blocker. Colab execution has not been directly tested. The notebook contains no fabricated figures, metrics or experimental conclusions. Assertions that concern real patients become findings only when the corresponding data cells successfully execute.

Run tests from the repository root:

```bash
python -m unittest discover -s tests -v
```
