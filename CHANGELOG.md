# Verified corrections

- Added `Duong_Reproduction_Improved.ipynb`; original notebooks and figures remain byte-identical to HEAD.
- Corrected tolerance accuracy to strict absolute error <0.7. Pearson, R², MAE and MSE calculations are unchanged.
- Added tested `duong_methods.py` helpers for baseline parameter reporting, undefined correlation formatting and two documented saliency methods.
- Preserved the 208-image Figure 3 workflow; reuse one four-logit embedding for regression seeds 0 and 49. Documented the unresolved author-output count of 199, exact feature order and t-SNE settings; export ordered filename/patient manifests.
- Preserved seed-0 deliberate case-selection rules for Figure 4. A matches released code; B explicitly documents an assumed signed-gradient, sigma-1, 5×5 reflected-boundary interpretation. Shared inputs/regression and global per-method scaling avoid silently normalizing each image independently.
- Added configurable Colab paths/seeds, dependency instructions, separate output directory and package/checkpoint/data provenance recording. Original quantitative plots are recreated only in the separate output directory.

## Validation executed in the onboarding environment

- **7/7 unit tests passed:** overprediction, underprediction, strict threshold, baseline reporting, method A equality with skimage, method B support/sign and method B constant-boundary preservation.
- **Synthetic integration passed:** executed the improved notebook's actual evaluate function on a patient-grouped split and independently checked Pearson, R², MAE and MSE equality.
- **Code/schema checks passed:** nbformat validation and transformed syntax parsing of every code cell; all saved outputs are empty.
- **Preservation passed:** original two notebooks, all tracked PNG figures and paper PDF compared byte-for-byte against HEAD.
- **Static inspection:** matched Figure 3's four pre-sigmoid outputs to author cells 22–23; traced author qualitative seed 49 through its 50-trial loops; verified the saved author embedding says 199 versus paper Figure 3's 208. No evidence determines which images differ or why.

## Not executed / remaining limitations

Full 94-image quantitative evaluation, the 208-image t-SNE and saliency panels could not run because images, historical metadata and expert scores are absent. The onboarding environment has a downloaded pretrained checkpoint; unavailable GPU resources are not the blocker. Colab installation/execution has not been directly tested. Historical checkpoint and dataset identity, exact selected examples and historical t-SNE coordinates cannot be certified. No numerical study results were fabricated. No seed was selected for improved performance. No ResNet18 was added.
