# Porting TAB to Python 3.12

This branch makes TAB run on Python 3.12 with current numpy, pandas, scikit-learn and
torch, **without changing what the benchmark computes**. Evaluation code, metrics,
thresholds, data loading, splits, seeds and the hyperparameters of the benchmark scripts
are unchanged. Wherever a newer library changed a default a baseline relied on, the old
behaviour is restored explicitly in the model code.

The approach is based on [marcpinet/TAB@python312-support](https://github.com/marcpinet/TAB/tree/python312-support).
It differs from that port in:
- the TODS defaults (see the PCA note below);
- the scikit-learn compatibility fixes;
- the parity checks against the original code.

## Environments

| | Original (legacy) | This branch |
|---|---|---|
| Requirements | `requirements-legacy-py38.txt` (upstream `5a7e6f8` + `pyod==2.1.0`) | `requirements.txt` |
| Python | 3.8 | 3.12 |
| numpy | 1.21.0 | 1.26.x |
| pandas | 1.3.4 | 2.x |
| scikit-learn | 0.24.2 | ≥ 1.3 (1.9.1 when checked) |
| pyod | 2.1.0 | 2.1.0 |
| stumpy | 1.4.0 | ≥ 1.12 (1.14.1 when checked) |
| torch | ≥ 1.11 (2.4.1 when checked) | ≥ 2.1 (2.14.1 when checked) |
| transformers | unpinned (4.46.3 when checked) | ≥ 4.40, < 4.50 |
| d3m / TODS stack | tamu-d3m 2022.5.23, tensorflow | not installed |

**pyod is pinned to 2.1.0 in both environments.** The original requirements did not pin
it, and pyod ≥ 3 requires Python ≥ 3.9. With the same pyod on both sides, a parity
check isolates the effect of the port.

The legacy file adds `more_itertools`. The vendored TODS code imports it, but the
original requirements did not list it. On a GPU machine, run the legacy environment with
`NUMBA_DISABLE_CUDA=1`. stumpy 1.4 compiles a CUDA kernel on import, which numba 0.55
cannot build for current NVIDIA drivers. No TAB model uses stumpy's GPU functions.

`requirements.txt` keeps `numpy<2` for salesforce-merlion, and `transformers<4.50` because
the GPT-2 code vendored by the LLM baselines targets the 4.x API.

## Changes that affect code paths

| Change | Why | Effect on results |
|---|---|---|
| `requirements.txt`: dropped `tamu_d3m`, `nimfa`, `combo`, `tensorflow`; raised floors | The d3m stack caps numpy, pandas and scikit-learn at 2021 versions and only installs on Python 3.8. Only `tods/third_party` used it. | none |
| `self_impl/TFAD`: `np.complex` → `np.complex128` | Removed in numpy 1.24. It was an alias of `complex`, and the product with an int array was already complex128. | none |
| `torch.load(..., weights_only=False)` in UniTS, Timer and CALF | torch 2.6 changed the default to `True`, which rejects these pickled checkpoints | none (restores the old default) |
| TODS models call pyod directly (`tods/tods_models.py`, `tods/pyod_core/`) | d3m does not install on Python 3.12 | none; bit-exact on the local check below |
| `tods/pyod_core/PCA.py`: re-orient principal components | scikit-learn 1.5 changed PCA's `svd_flip` sign convention. pyod's PCA score is a distance to the components, so it depends on their sign. | none; without the fix, PCA scores changed by up to 5% and AUC-PR by up to 0.011 on test series |
| `sklearn_compat.kmeans_sklearn024()` used by `self_impl.KMeans` and CBLOF | scikit-learn changed `KMeans` defaults: `n_init` 10 → `'auto'` (1.4), `algorithm` elkan → lloyd (1.1), and how k-means++ draws its first center (1.3). The same seed gave a different clustering. | none; without the fix, CBLOF AUC-PR changed from 0.29 to 0.64 on one test series |
| `tods.autoencoderski`, `tods.lstmodetectorski` raise `NotImplementedError` | They used the TensorFlow/Keras models of the d3m stack | Not used by any benchmark script. Run them in the legacy environment. |

### TODS on pyod

With TAB's hyperparameters (`--model-hyper-params '{}'`, `use_semantic_types=False`,
`return_subseq_inds=False`), the d3m wrappers (`BaseSKI` → `UODBasePrimitive`) only:
- convert the numpy array to a d3m DataFrame and back;
- call `self._clf.fit(X)` / `decision_function(X)` / `predict(X)` on all columns.

The new adapters build the same estimators with the same **effective** constructor
arguments. These were dumped from the original primitives with
`scripts/parity/dump_tods_defaults.py` and are stored in
[`tods_legacy_params.json`](tods_legacy_params.json). `scripts/parity/check_tods_params.py`
checks `get_params()` of every new model against the dump.

- **PCA (`pcaodetectorski`):** `n_components=None` and `n_selected_components=None`, so all
  components are kept. The reference port uses `n_components=1`, which would change the
  PCA baseline. PCA also uses `whiten=True`, `standardization=True`, `window_size=10` and
  `step_size=1`.
- **HBOS:** `tol=0.1` (the current pyod default is 0.5).
- **LOF:** `novelty=True` (pyod's default; the primitive did not pass it).
- **CBLOF:** gets an explicit `clustering_estimator` (the 0.24-compatible KMeans). This is
  the only intended difference in `get_params()`.
- **Model names:** `tods.<name>` are unchanged. `isolationforestski` and `cofski` are
  ported too and, as before, are reachable as `tods.tods_models.<name>`.

### Known residual differences

- **Floating-point noise** from different numpy, scipy and BLAS builds, around 1e-9
  relative.
- **Series2Graph** uses scikit-learn's `PCA(n_components=3)` directly. The `'auto'` solver
  and the sign convention changed after 0.24, and 0.24 picked a randomized solver there,
  which cannot be reproduced exactly. It is to be measured on the host.
- **Deep-learning models** run on different torch and CUDA versions and are not expected to
  be bit-identical.

## Checks done so far

Both runs use the same harness, `scripts/parity/dump_scores.py`, on the original code
(upstream `5a7e6f8`, legacy Python 3.8 environment, CPU) and on this branch (Python 3.12,
CPU). The harness:
- runs a benchmark script's model through the strategy's own `execute()`, so the config,
  model loader, hyperparameters, split and seed 2021 are the pipeline's own;
- saves the raw, unpadded output of `detect_score` / `detect_label`.

The data was synthetic series in TAB's on-disk format: 3 univariate series and 2
multivariate series of 900–1,600 points, with point and subsequence anomalies.

- **Model resolution:** every `--model-name` in `scripts/` resolves and instantiates on
  Python 3.12 (`check_models.py --instantiate`: 248 invocations, 0 failures). Pre-trained
  checkpoints were not present locally.
- **TODS parameters:** `check_tods_params.py` passes for all 9 TODS models.
- **TODS score and label parity,** across every TODS benchmark script (univariate and
  multivariate, score and label):

| Model | Scores (max \|Δ\|) | Labels (agreement) |
|---|---|---|
| hbosski, knnski, lofski, lodaski, cofski, isolationforestski | 0 | 100% |
| ocsvmski | ≤ 3e-14 | 100% |
| pcaodetectorski | ≤ 7e-9 (scores ~1e4) | 100% |
| cblofski | ≤ 2e-16 | 100% |

`cblofski` raises "Could not form valid cluster separation" on one synthetic series in
**both** environments. That is pyod's behaviour on that data, not a regression.

Not yet checked: everything else, including KMeans, MatrixProfile, Merlion, the deep
models, and the pre-trained and LLM models on real data. See below.

## Verification on the GPU host

See the commands in the hand-off message, or:
- `scripts/parity/run_parity.sh` dumps the scores of every benchmark script on a set of
  series.
- `scripts/parity/compare_scores.py` compares two dump directories.

**Expectations:**
- **Classical models** (TODS, KMeans, MatrixProfile, Merlion statistical detectors) match
  up to floating-point noise.
- **Deep models** should land in the same range.
- **Zero-shot pre-trained models** should give Spearman ≥ 0.99.
- **Regression signal:** a script that is `OK` in the reference log but `FAIL` in the new
  log is a regression.
