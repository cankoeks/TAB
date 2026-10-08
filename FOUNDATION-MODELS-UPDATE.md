# TODO: integrate new foundation models (Chronos, TimesFM, TabPFN)

Add three foundation models to TAB as anomaly-detection baselines, each on its latest
stable release: **Chronos**, **TimesFM** and **TabPFN**. They must run in the same
Python 3.12 environment as the existing baselines, without changing any existing
baseline's results.

## Versions

Latest releases on PyPI when this file was written (2026-10-08). Re-check them before
starting and pin the exact versions used.

| Model | Package | Version | Requires Python |
|---|---|---|---|
| Chronos (Chronos-2) | `chronos-forecasting` | 2.3.2 | ≥ 3.10 |
| TimesFM | `timesfm` | 3.0.2 | ≥ 3.10 |
| TabPFN | `tabpfn` (via `tabpfn-time-series`) | 9.1.0 (`tabpfn-time-series` 1.3.0) | ≥ 3.10 |

- Record which TabPFN model generation and checkpoint the pinned `tabpfn` release uses by
  default, so the exact model version can be stated in the thesis.
- Check whether the TabPFN weights need a licence acceptance or a Hugging Face login on the
  GPU host, and whether `tabpfn-time-series` sends telemetry or calls the TabPFN client API.
  Disable both for offline benchmark runs.

### Co-install status

A dry-run resolve on Python 3.12 of `requirements.txt` plus all three packages succeeded,
with the latest release of each:
- chronos-forecasting 2.3.2;
- timesfm 3.0.2;
- tabpfn-time-series 1.3.0 (tabpfn 9.1.0).

That resolve already respected TAB's constraints: `numpy<2` for salesforce-merlion and
`transformers<4.50` for the vendored LLM code. It ran with pyod unpinned; repeat it with
the current `pyod==2.1.0`. Re-run the resolve before adding the packages and read the
resolved versions, not just the exit code: pip can silently backtrack a package to an old
release.

## Constraints

- **Do not change existing baselines.** TAB already contains `pre_train.Chronos` (a
  vendored Chronos-Bolt) and `pre_train.TimesFM` (a vendored TimesFM 1.0 PyTorch port).
  Leave both untouched and register the new models under new names, e.g.
  `pre_train.Chronos2`, `pre_train.TimesFM3` and `pre_train.TabPFN`.
- **Watch for import shadowing.** Several TAB adapters insert cwd-relative paths at
  `sys.path[0]`, for example `pre_train/submodules/chronos`, `.../tsfm` and `.../moment`.
  These can shadow top-level modules (`utils`, `base`, `models`, `layers`, `tsfm_public`,
  `momentfm`). Add a one-process import test: import `ts_benchmark.baselines.pre_train` and
  `ts_benchmark.baselines.LLM`, then `chronos`, `timesfm` and `tabpfn_time_series`.
- **Avoid eager imports.** `pre_train/__init__.py` imports every pre-trained model eagerly.
  Import the new packages lazily, inside the adapters, so a missing optional package
  doesn't break the other pre-trained baselines.
- **Weights:** download them into `ts_benchmark/baselines/pre_train/checkpoints/<model>/`,
  which is git-ignored, or document the Hugging Face cache location. Never commit them.

## Implementation

1. Add the three packages with exact pins, either to `requirements.txt` or to a separate
   `requirements-foundation.txt` installed on top of it. Run `pip check` and the import test.
2. Write one adapter per model that follows TAB's model interface:
   - `detect_fit(train_data, train_label)`, where zero-shot models may just store context;
   - `detect_score(test)` returning a 2-tuple `(scores, scores)`, one score per test
     timestamp (shorter outputs are zero-padded at the end by the pipeline);
   - `detect_label(test)` returning `({ratio: labels}, scores)` for the label strategies,
     following the existing models that use `anomaly_ratio`.
3. **Scoring.** All three models are forecasters, not detectors. Score anomalies the same
   way the existing `PreTrain_adapter` does
   (`ts_benchmark/baselines/pre_train/adapters_for_model.py`): the per-timestamp error
   between the model's prediction and the observed value, averaged over channels.
   - Use the same context length (`seq_len`) and horizon as the existing TimesFM zero-shot
     scripts, so the comparison is like for like.
   - Chronos-2 and TimesFM 3 predict quantiles. Use the median (point forecast) for the
     error. Optionally add a quantile-based score as a separate, clearly named variant.
   - TabPFN-TS treats forecasting as tabular regression. Document how the rolling
     context/target windows are built.
   - Handle multivariate series. Chronos-2 supports them natively; for the others, use
     per-channel univariate forecasts and average the errors, and document the choice.
4. **Seeds and determinism.** The strategy seeds `random`, `numpy` and `torch` with 2021
   before each series. Check that the new models are deterministic under that seed (TabPFN
   ensembling, any sampling) and set their own seed parameters explicitly where they exist.
5. **Scripts.** Add zero-shot scripts under `scripts/{univariate,multivariate}/{score,label}/`,
   mirroring `TimesFMzero.sh` (same config files, `--adapter` if applicable, `--gpus`).
   Add few-shot or fine-tuned variants only if needed for the thesis.
6. **Devices.** The GPU host has RTX 3090 (24 GB) and RTX 2080 Ti (11 GB) cards. Check
   memory per model at the chosen context length and batch size. The 2080 Ti has no
   bfloat16 support, so pick a dtype that works on both cards.

## Verification

- [ ] `pip check` is clean, and the import test passes in one process.
- [ ] Every existing script model still resolves and instantiates, and the existing
      baselines still match their pre-integration scores on a few series.
- [ ] Each new model runs end to end through `scripts/run_tab.sh` (`run_benchmark.py`) on a
      small univariate and multivariate subset:
      - output length equals the test length;
      - no traceback in the result CSV's `log_info`. The pipeline catches exceptions and
        still exits 0, so the exit code proves nothing.
- [ ] A second run with the same seed reproduces the scores (or the nondeterminism is
      documented).
- [ ] The versions, weights, scoring method and any deviations are written down for the
      thesis.
