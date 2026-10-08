"""Check that the pyod-based TODS models use the original effective parameters.

Run from the TAB root in the Python 3.12 environment:

    python scripts/parity/check_tods_params.py [docs/tods_legacy_params.json]

The JSON file is the output of ``dump_tods_defaults.py`` in the legacy Python 3.8
environment. For every model, ``get_params()`` of the estimator built by
``tods_models.TodsModelAdapter`` with TAB's hyperparameters ('{}') must equal the
parameters of the estimator built by the original d3m primitive. The only
expected difference is CBLOF's explicit ``clustering_estimator``.
"""
import json
import os
import sys

sys.path.insert(0, os.getcwd())

from ts_benchmark.baselines.tods import tods_models  # noqa: E402

EXPECTED_DIFFERENCES = {"CBLOFSKI": {"clustering_estimator"}}


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join("docs", "tods_legacy_params.json")
    with open(path) as f:
        legacy = json.load(f)

    failed = False
    for model_name, model_class, _ in tods_models.TODS_MODELS:
        model = tods_models.TodsModelAdapter(model_name, model_class, {}).build_model()
        params = model.get_params(deep=False)
        expected = legacy[model_name]["pyod"]
        allowed = EXPECTED_DIFFERENCES.get(model_name, set())
        diffs = {
            key: (expected.get(key), params.get(key))
            for key in set(expected) | set(params)
            if key not in allowed and expected.get(key) != params.get(key)
        }
        print("%-20s %s" % (model_name, "OK" if not diffs else "DIFF %s" % diffs))
        failed = failed or bool(diffs)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
