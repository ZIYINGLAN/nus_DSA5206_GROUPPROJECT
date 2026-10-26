# EVA-X-Tiny three-pipeline comparison

Branch: `eva-x-comparison`, based on `resnet18-comparison`. Source implementation with the pretrained smoke test and real-data evaluation still unexecuted. All prior files/results are preserved. No pretrained weight binaries, medical images or generated study results are placed in the repository.

## Actual compatibility results and current blocker

- Created a **separate** environment at `/workspace/.onboarding/eva-x-env`; the earlier validated environment was not changed. Tested Python 3.12.14, torch 2.5.1+cpu and torchvision 0.20.1+cpu.
- Official README recommends timm 0.9.0. Imports succeeded, but an actual architecture forward test failed: `AttributeError: 'EVA_X' object has no attribute '_pos_embed'`. The legacy training requirements also list a conflicting timm 0.4.12. Neither is silently assumed compatible.
- **timm 0.9.16** passed the official architecture constructor/forward-pooling test, producing finite 192D features and bitwise-stable repeated outputs in frozen eval mode. The official code/architecture were not patched. This test uses an **untrained architecture fixture**, not pretrained evidence; the notebook pins 0.9.16.
- **Official pretrained weights have NOT been loaded.** The attempted HTTPS request to the official Hugging Face metadata/download service was rejected by this cloud's proxy (`Tunnel connection failed: 403 Forbidden`). There is no checkpoint hash or successful checkpoint-coverage report to claim from this environment. The complete pretrained smoke test and real-data experiment remain unexecuted.
- A network configuration draft was saved adding `huggingface.co`, `cdn-lfs.huggingface.co`, and `cas-bridge.xethub.hf.co` while preserving existing network presets. Saving the draft does not activate it. Review/save it in environment settings and publish the environment to enable the requested access; further redirects, if any, must be diagnosed rather than bypassing controls. Alternatively, run the gated smoke test in Colab with public Hugging Face access, or supply the official checkpoint locally with a trusted independently obtained SHA-256.

The notebook **stops at the pretrained smoke test** unless it successfully loads the official checkpoint, audits the backbone state, processes one synthetic grayscale image and obtains a stable finite (1,192) pooled output. Downstream workflow is implemented but not validated on real data. This is not a claim that the requested pretrained smoke test passed.

## Official implementation, pooling and preprocessing

Reviewed source: https://github.com/hustvl/EVA-X at commit `eedd642c85339b25682ceb574b644cb9afe88e62`.

- Root `eva_x.py` SHA-256: `5c21eb02cd977c35d4b8cf2984081b88b73db49953c1b4c2235f30df29df8219`.
- Calls the unchanged official `eva_x_tiny_patch16(pretrained=...)` factory: patch16, 224 input, embed192, depth12, 3 heads, SwiGLU, rotary embeddings. Source is downloaded to the output assets directory at the pinned commit, hash-verified, then imported.
- Uses `forward_features` followed by `forward_head(..., pre_logits=True)`: average patch tokens excluding prefix/CLS tokens, then official fc_norm. No logits or arbitrary flattening. All parameters frozen, eval mode, inference_mode.
- Official filter handles MIM wrappers/heads and fixed rotary buffers. After factory loading, require every non-head/non-fixed-RoPE backbone tensor to have matching checkpoint key, shape **and loaded value**; reject missing/unexpected backbone keys. The unused classification head may be absent. Report its absence and all caught warnings; no random-backbone fallback exists. Actual official checkpoint coverage remains untested because download is blocked.
- Official checkpoint: https://huggingface.co/MapleF/eva_x/blob/main/eva_x_tiny_patch16_merged520k_mim.pt . Downloader queries official metadata for an immutable revision and LFS SHA-256, verifies the downloaded payload, and records both. Local checkpoint override requires a matching trusted expected hash.
- `classification/utils/datasets.py` uses **CXR mean=0.49185243, std=0.28509309**, repeated across channels when model name includes eva. For the non-CheXpert CXR14 evaluation branch: bicubic resize shorter edge256, center crop224, ToTensor and this normalization. The official CXR14 loader calls PIL `.convert('RGB')`; we do the same (grayscale inputs are naturally replicated). This preserves any source RGB channels, unlike the earlier ResNet pipeline's explicit luminance conversion. Antialias=True is explicit. This is the prespecified transfer policy here.
- Official CheXpert-Tiny instead resizes the shorter edge to224 before center cropping. There is no official COVID-severity preprocessing branch. The CXR14 choice is documented before testing, not tuned on outcomes. It is not ImageNet normalization and not a claim of a unique universal EVA-X evaluation transform.

## Shared evaluation and baseline reuse

To avoid a second inconsistent loader, the new notebook reads the unchanged `ResNet18_Comparison.ipynb`, selects its exact reviewed data/feature/baseline cells by entry-point prefixes, and executes them in a separate namespace. Its OUTPUT_DIR is redirected to `/content/duong-eva-x-results/baseline_revalidation/`. No previous notebook/result is modified. Cell selectors fail if source structure changes; the shared notebook and helper hashes are recorded.

This reuses exactly the XRV PA loader, filename join/mask/row order, 94-image/65-patient checks, source/image hashes, original transforms, checkpoints, verified feature cache logic and DenseNet OLS code. Previous ResNet source manifests, when available, must match ordered names/patient IDs/labels and all image/CSV hashes. Mismatch halts; equal cohort counts alone are insufficient. Every available historical Critical Assessment CSV feature/target/seed record is audited by the original code without requiring split JSON.

Earlier baseline OLS/Ridge predictions are regenerated using validated code and verified cached features where compatible. Available exported predictions are reused only after identity and exact split checks, matching feature/checkpoint/environment/preprocessing/regression provenance and per-observation numerical agreement with recomputed predictions. Unverified baseline provenance triggers regeneration; identity/split mismatches stop the run. This conservative numerical revalidation can require re-running both earlier regressions. User-provided MAE/R² reference means are displayed only as diagnostics after actual predictions exist; they never populate study result files.

EVA features use a **separate** checksummed `.npz` cache and feature integrity manifest containing source order, patient IDs, labels, image/CSV/checkpoint/source hashes, preprocessing, pooling and versions. Expected shape (94,192); nonfinite or incorrect dimensions fail. Cache hit timings are marked explicitly; extraction duration is null on hits.

Exactly the same `critical_methods.exact_splits` seeds 0–49 and `resnet18_methods.fit_nested_ridge` are used. Alpha grid {0.01,0.1,1,10,100,1000}, fold-local StandardScaler, up to five patient GroupKFold inner folds, MSE selection, refit on outer training only; no PCA or backbone fine-tuning in this three-model experiment. Training-only inputs to the shared fitter are tested explicitly. Export all outer and inner memberships with patient overlap assertions. Scatter split stays at seed0. Per-seed MAE differences are EVA-X minus DenseNet and EVA-X minus ResNet18; negative favours EVA-X descriptively.

## Exact Colab execution

1. Use a fresh Colab Python 3.11/3.12 runtime. The executed local compatibility tests used **3.12**, not Colab itself. CPU is the default; GPU is unnecessary for the small frozen-backbone experiment.
2. Clone this branch from a Colab setup notebook:

   ```python
   !git clone --branch eva-x-comparison https://github.com/ZIYINGLAN/nus_DSA5206_GROUPPROJECT.git /content/nus_DSA5206_GROUPPROJECT
   ```

   Open `EVA_X_Comparison.ipynb` from this branch in Colab, or download it and use File → Upload notebook. The cloned checkout includes `eva_x_methods.py` and all unchanged shared helpers/notebooks; manual helper uploads are unnecessary.
3. Run the pinned dependency cell (includes **timm==0.9.16**, torch2.5.1, torchvision0.20.1). Restart if requested. Continue with configuration; mount Drive using the optional mount lines before data access.
4. Default data paths remain:

   ```text
   /content/drive/MyDrive/duong-data/images/
   /content/drive/MyDrive/duong-data/metadata_2020.csv
   /content/drive/MyDrive/duong-data/covid-severity-scores.csv
   ```

   Score CSV format: five preamble rows before filename/geographic_mean/opacity_mean header. Data are not included. Use existing `DUONG_REPO_DIR`, `DUONG_DATA_DIR`, `DUONG_IMAGE_DIR`, `DUONG_METADATA_CSV`, `DUONG_SCORES_CSV` overrides as needed.
5. Set `DUONG_RESNET_RESULTS_DIR` to your prior ResNet output folder (containing `reproducibility_manifest.json`, `all_model_heldout_predictions.csv` and optional feature caches). Set `DUONG_CRITICAL_RESULTS_DIR` and `DUONG_REPRO_RESULTS_DIR` for prior assessment predictions and existing XRV weights. These are read-only sources. If predictions/provenance are unavailable, the code regenerates them from real data; no historical equivalence is claimed solely from the supplied mean metrics.
6. Run the **smoke cell first**. It downloads pinned official source and the official checkpoint, verifies hashes/coverage and prints versions, device, warnings, input/output shapes and stability. If Hugging Face is blocked in your runtime, fix service access or upload the official checkpoint and set `DUONG_EVA_CHECKPOINT` plus `DUONG_EVA_CHECKPOINT_SHA256` to a trusted matching value. Do not set the expected hash from an untrusted substitute. Never bypass the smoke assertion.
7. Only after `status: passed`, run the remaining cells in order. All new outputs, including assets and caches, stay under `/content/duong-eva-x-results/`. Download/copy that directory to Drive after completion because `/content` is ephemeral. No plots or result tables are generated until actual data execution.

## Required exports (created only after successful experiment)

`three_model_comparison_summary.csv`, `three_model_comparison_by_seed.csv`, `eva_x_heldout_predictions.csv`, `paired_eva_x_differences.csv`, `three_model_comparison.png`, `eva_x_prediction_scatter.png`, `reproducibility_manifest.json`, `patient_split_audit.json`, `feature_integrity_manifest.json`.

Additional outputs include actual three-model predictions, the ordered cohort manifest, verified earlier predictions, raw EVA features, pretrained smoke report and separate resource records. Report mean ± sample SD over overlapping splits, counts of defined metrics and per-seed paired differences. Timing separates fresh extraction (decode/preprocessing included), cache hits and nested-CV workloads; it is not a controlled model-training benchmark.

## Tests and scientific limits

The complete suite is run with the official-architecture-only test enabled using `DUONG_EVA_OFFICIAL_SOURCE`; **38/38 tests passed with no skips** (26 inherited and 12 new). This includes the synthetic 50-split three-model integration, checkpoint-coverage rejection cases, official CXR normalization/RGB handling, identity/pairing checks, training-only fitter inputs, and the architecture-only forward/pooling/stability test. Notebook schema/code syntax, shared-cell selectors and byte-for-byte preservation checks also passed. The isolated environment passed its package compatibility check. Synthetic fixtures are unit/integration tests only; no synthetic experimental results or figures are delivered. Checkpoint coverage validators are exercised on controlled tensor fixtures; that cannot establish actual pretrained checkpoint compatibility. Full pretrained smoke, real 94-image features, previous-run identity and numerical checks, clinical metrics and plots are still **unexecuted**.

To run the general suite after installation:

```bash
python -m unittest discover -s tests -v
```

The optional official-architecture test is skipped unless its source path is configured. After the notebook fetches source, enable it explicitly:

```bash
DUONG_EVA_OFFICIAL_SOURCE=/content/duong-eva-x-results/eva_assets/official_eva_x.py python -m unittest discover -s tests -v
```

Interpretation is limited to frozen **pipeline transfer**. Chest-X-ray versus ImageNet pretraining, self-supervised objectives, single-logit versus 192D/512D representations and preprocessing differ. Sixty-five labelled patients and uncertain expert-mean scores limit precision and external validity. Repeated overlapping outer splits are dependent; sample SD is descriptive, not an independent-replication SE, and no naive paired t-test is offered as proof of significance. Upstream pretraining images could overlap the evaluation cohort; supervised split disjointness does not rule that out. Upstream manifests/deduplication are unavailable, so pretraining overlap remains unverified. No architecture-only causal effect, external generalization, clinical readiness or EVA-X superiority is claimed.

## Manually downloaded checkpoint fallback

When Hugging Face is unavailable, upload the official `.pt` file to Drive or Colab and run this before the smoke cell:

```python
import os
os.environ['DUONG_EVA_CHECKPOINT'] = '/content/drive/MyDrive/duong-data/eva_x_tiny_patch16_merged520k_mim.pt'
os.environ['DUONG_EVA_CHECKPOINT_SHA256'] = 'f82d6fc450a83ac7c4d32fa9f0d6459ea7a53fd08e47fdef0cb12ce35c97174d'
```

This digest was supplied by the user and is the notebook's default expected checksum for the local-file path. It has **not** been locally verified against the official pretrained payload. The loader hashes the actual file and stops on mismatch. With a local checkpoint, no Hugging Face request is made; fetching pinned official Python source still requires GitHub access unless already cached. The hash alone is not a successful loading/coverage/finite-feature smoke test. All those gates remain mandatory, and neither pretrained loading nor real-data evaluation has been executed here.
