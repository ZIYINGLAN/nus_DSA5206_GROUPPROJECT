# EVA-X restricted checkpoint-loading fix

## Diagnosis and evidence

The Colab traceback identifies a NumPy scalar pickle global, `numpy.core.multiarray.scalar`, rejected by torch 2.5.1's weights-only unpickler. A tensor checkpoint mapping with `best_auc=np.float64(...)` reproduces that exact failure. Loading it also requires NumPy's dtype constructor and its concrete float64 dtype class, which can be created dynamically during unpickling.

The pinned official factory contains its own `torch.load(pretrained, map_location='cpu')` without `weights_only`. Under torch 2.5.1 this is unrestricted pickle loading. Previously this happened **before** the explicit restricted second load failed. Fixing only that second call would leave the earlier unsafe load in place.

The official Hugging Face metadata endpoint was inspected successfully:

- Repository revision: `35ddcd6dab6ca99bbdb6cb45c8d1b093aefbd0ee`.
- Checkpoint size: 79,513,863 bytes.
- Published LFS SHA-256: `f82d6fc450a83ac7c4d32fa9f0d6459ea7a53fd08e47fdef0cb12ce35c97174d` (matches the supplied digest).
- The payload redirects to `us.aws.cdn.hf.co`, which the execution proxy blocks with HTTP 403. Its actual pickle metadata and full pretrained payload therefore could not be inspected locally. Float64 metadata compatibility is established with regression fixtures; other NumPy dtype classes are deliberately not speculatively allowed.

A draft adding that exact CDN hostname to the existing network allowlist was saved. It is not a runtime access change: review/save and publish the environment configuration before retrying here. The file can also be tested directly in Colab using the existing local-checkpoint path.

## Proposed implementation

Only `eva_x_methods.py` is changed among existing project files. New regression tests and this document are added; all previous notebooks, results, documentation, evaluation helpers and tests remain untouched.

1. Open the checkpoint once, calculate SHA-256 from that handle, compare to the expected digest, seek back, then deserialize that same handle. A mismatch stops before any deserialization or factory execution.
2. Use `torch.load(..., map_location='cpu', weights_only=True)` inside `torch.serialization.safe_globals()` containing only:
   - `numpy.core.multiarray.scalar`;
   - `numpy.dtype`;
   - `type(numpy.dtype(numpy.float64))` (NumPy 1.26: `numpy.dtypes.Float64DType`).
3. Add no permanent globals and preserve any caller-preexisting entries on scope exit. Unknown classes/dtypes fail closed. There is **no unrestricted-pickle fallback**, automatic or opt-in, because current regression evidence does not establish a need for one. Never change to `weights_only=False` merely to suppress the error; it can execute arbitrary checkpoint code.
4. Execute the same official factory bytecode/architecture with a **function-local** replacement for its `torch.load` reference, returning the already validated in-memory mapping. The official source, module globals and process-wide `torch.load` are not patched. No second checkpoint deserialization occurs. Official construction and checkpoint filtering still run.
5. Keep the existing complete backbone key/shape/loaded-value coverage checks unchanged. Architecture, frozen/eval state, finite 192D pooled features, and repeated-output checks remain mandatory. No missing/random-backbone fallback is introduced.
6. Include deserialization mode, scoped allowlist, top-level checkpoint keys and NumPy scalar metadata types (not values) in the smoke audit. This lets an actual Colab run report the real metadata without dumping checkpoint tensors or optimizer values.

## Tests actually executed

**48/48 tests passed with no skips** in each isolated environment:

| Python | torch | torchvision | timm | NumPy | Result |
|---|---|---|---|---|---|
| 3.11.17 | 2.5.1+cpu | 0.20.1+cpu | 0.9.16 | 1.26.4 | 48 passed |
| 3.12.14 | 2.5.1+cpu | 0.20.1+cpu | 0.9.16 | 1.26.4 | 48 passed |

The ten new tests cover exact error reproduction and float64 recovery, checksum-before-load enforcement, one restricted load only, no unrestricted retry, rejection of unknown globals/unobserved dtype classes, scope restoration, non-mapping rejection, and official factory execution/coverage with a synthetic state mapping containing NumPy metadata. The pinned official factory integration fixture produces stable finite 192D features but is **not the official pretrained checkpoint smoke test**. The prior 38 tests also pass, including split, CV, metric and pairing checks.

Run with the reviewed official source downloaded outside the repository:

```bash
DUONG_EVA_OFFICIAL_SOURCE=/path/to/official_eva_x.py python -m unittest discover -s tests -v
```

Without that source variable, the two official-source architecture/factory tests are explicitly skipped. No fixtures or model binaries are committed. Actual Google Colab execution itself has not been performed from this workspace.

## Retry in the existing Colab runtime

No dependency upgrade is required: retain Python 3.11, torch 2.5.1, torchvision 0.20.1, timm 0.9.16 and the notebook's NumPy 1.26.4 pin.

1. Upload the locally fixed `eva_x_methods.py` to `/content/nus_DSA5206_GROUPPROJECT/eva_x_methods.py`. Use the updated helper from branch `eva-x-comparison`, or upload it directly.
2. Restart the runtime and rerun the notebook installation/configuration imports as needed, or explicitly reload the helper before rerunning its smoke cell:

   ```python
   import importlib, eva_x_methods
   importlib.reload(eva_x_methods)
   from eva_x_methods import pretrained_smoke, pooled_features
   ```

3. Keep `DUONG_EVA_CHECKPOINT` pointing at the manually downloaded official `.pt` file, with the expected SHA above. Run the original smoke cell. It now performs the restricted single load and records the metadata audit.
4. Confirm the current run reports `status: passed`, backbone coverage 1.0, the expected hash, finite `(1,192)` features and repeat stability before continuing. Unsupported additional metadata must be inspected; do not disable the restrictions or fake a smoke report.

**Unexecuted:** loading the actual official pretrained payload with this fix, the real-data smoke test in Colab, and any historical-cohort evaluation. The payload download remains externally blocked here, and the real dataset is absent. This patch fixes and tests the reported deserialization mechanism; it does not claim an official pretrained smoke pass without that run.
