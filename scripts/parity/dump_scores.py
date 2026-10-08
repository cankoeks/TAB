"""Dump raw anomaly scores / labels of TAB models for parity checks.

Runs a model exactly as one of TAB's benchmark scripts does (same config,
strategy, model loader, hyperparameters, data split and seeding, via the
strategy's own ``execute``) and saves the unpadded output of ``detect_score`` /
``detect_label`` per series to ``.npz`` files. Compatible with Python 3.8 and
3.12, so the same harness can run against the original code in the legacy
environment and against the ported code:

    python scripts/parity/dump_scores.py --tab-root ~/TAB-ref \\
        --script scripts/multivariate/score/hbosski.sh \\
        --series <name.csv> [<name.csv> ...] --out-dir ~/parity/ref

``--tab-root`` selects the code that is imported; the script paths are read
from that tree. Compare two output directories with ``compare_scores.py``.
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from script_args import parse_script  # noqa: E402

VERSION_PACKAGES = [
    "numpy", "pandas", "sklearn", "scipy", "pyod", "stumpy", "numba", "torch",
    "transformers", "merlion",
]


def _versions():
    versions = {"python": platform.python_version()}
    for name in VERSION_PACKAGES:
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "?")
        except Exception:
            versions[name] = None
    return versions


def _git_sha(root):
    try:
        return subprocess.check_output(
            ["git", "-C", root, "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


class _SourcePool:
    """Minimal data pool over a LocalDataSource (the pipeline uses a global storage pool)."""

    def __init__(self, source):
        self._dataset = source.dataset

    def get_series(self, name):
        return self._dataset.get_series(name)

    def get_series_meta_info(self, name):
        return self._dataset.get_series_meta_info(name)


class _RecordingFactory:
    """Wraps a ModelFactory and records what the model's detect_* methods return."""

    def __init__(self, factory):
        self.factory = factory
        self.model_name = factory.model_name
        self.model_hyper_params = factory.model_hyper_params
        self.records = {}

    def __call__(self):
        model = self.factory()
        for method in ("detect_score", "detect_label"):
            original = getattr(model, method, None)
            if original is not None:
                setattr(model, method, self._wrap(method, original))
        return model

    def _wrap(self, method, original):
        def wrapper(*args, **kwargs):
            result = original(*args, **kwargs)
            self.records[method] = result
            return result

        return wrapper


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--tab-root", required=True, help="TAB tree whose code is run")
    parser.add_argument("--script", required=True, help="benchmark script, relative to --tab-root")
    parser.add_argument("--series", nargs="+", required=True, help="series file names")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dataset-root", default=None,
                        help="anomaly_detect dataset folder (default: <tab-root>/dataset/anomaly_detect)")
    parser.add_argument("--model-name", default=None,
                        help="run this model with the script's config instead of the script's model")
    parser.add_argument("--hyper-params", default=None,
                        help="JSON overriding the script's --model-hyper-params")
    parser.add_argument("--seed", type=int, default=None,
                        help="replace the strategy's fixed seed (2021); only to study seed variance")
    opts = parser.parse_args()

    root = os.path.realpath(os.path.expanduser(opts.tab_root))
    out_dir = os.path.abspath(os.path.expanduser(opts.out_dir))
    dataset_root = os.path.abspath(os.path.expanduser(opts.dataset_root)) if opts.dataset_root else None
    os.makedirs(out_dir, exist_ok=True)
    # several adapters insert cwd-relative paths into sys.path, so run from the tree root
    os.chdir(root)
    sys.path.insert(0, root)

    import ts_benchmark
    imported = os.path.realpath(ts_benchmark.__file__)
    if not imported.startswith(root + os.sep):
        raise RuntimeError("ts_benchmark imported from %s, not from %s" % (imported, root))

    from ts_benchmark.common.constant import ANOMALY_DETECT_DATASET_PATH
    from ts_benchmark.data.data_pool import DataPool
    from ts_benchmark.data.data_source import LocalDataSource
    from ts_benchmark.evaluation.evaluator import Evaluator
    from ts_benchmark.evaluation.strategy import STRATEGY
    from ts_benchmark.evaluation.strategy import anomaly_detect
    from ts_benchmark.models.model_loader import get_models

    entries = list(parse_script(opts.script))
    if len(entries) != 1:
        raise RuntimeError("expected one run_benchmark.py call in %s" % opts.script)
    entry = entries[0]
    if opts.model_name is not None:
        entry["model_name"] = opts.model_name
        entry["adapter"] = None
    if opts.hyper_params is not None:
        entry["hyper_params"] = json.loads(opts.hyper_params)
    with open(os.path.join("config", entry["config_path"])) as f:
        config = json.load(f)

    if opts.seed is not None:
        fix_random_seed = anomaly_detect.fix_random_seed
        anomaly_detect.fix_random_seed = lambda seed=None: fix_random_seed(opts.seed)

    source = LocalDataSource(dataset_root or ANOMALY_DETECT_DATASET_PATH, "DETECT_META.csv")
    source.load_series_list(opts.series)
    DataPool().set_pool(_SourcePool(source))

    strategy_args = config["evaluation_config"]["strategy_args"]
    strategy_class = STRATEGY[strategy_args["strategy_name"]]
    evaluator = Evaluator([{"name": m} for m in strategy_class.accepted_metrics()])
    strategy = strategy_class(strategy_args, evaluator)

    model_config = {
        "models": [{
            "adapter": entry["adapter"],
            "model_name": entry["model_name"],
            "model_hyper_params": entry["hyper_params"],
        }],
        "recommend_model_hyper_params": config["model_config"].get(
            "recommend_model_hyper_params", {}
        ),
    }
    factory = get_models(model_config)[0]

    meta = {
        "model_name": entry["model_name"],
        "script": opts.script,
        "config_path": entry["config_path"],
        "strategy": strategy_args["strategy_name"],
        "hyper_params": factory.model_hyper_params,
        "seed": opts.seed if opts.seed is not None else 2021,
        "tab_root": root,
        "ts_benchmark_file": imported,
        "git_sha": _git_sha(root),
        "versions": _versions(),
    }

    failures = []
    script_stem = os.path.splitext(opts.script.replace(os.sep, "_"))[0]
    for series_name in opts.series:
        recorder = _RecordingFactory(factory)
        rows = strategy.execute(series_name, recorder)
        log_info = rows[0][strategy.field_names.index("log_info")] if rows else ""
        if not recorder.records or "Traceback" in str(log_info):
            failures.append(series_name)
            print("FAIL  %s %s\n%s" % (entry["model_name"], series_name, log_info))
            continue

        _, _, _, test_label = strategy.split_data(series_name)
        arrays = {"test_label": test_label.to_numpy().flatten()}
        for method, result in recorder.records.items():
            first, second = result
            if isinstance(first, dict):
                # some label models return {ratio: labels}
                for key, value in first.items():
                    arrays["%s_0_%s" % (method, key)] = np.asarray(value)
            else:
                arrays["%s_0" % method] = np.asarray(first)
            arrays["%s_1" % method] = np.asarray(second)
        metrics = dict(zip(strategy.field_names, rows[0]))
        series_meta = dict(meta, series=series_name, metrics={
            k: v for k, v in metrics.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        })
        path = os.path.join(
            out_dir,
            "%s__%s__%s.npz" % (entry["model_name"], script_stem, os.path.splitext(series_name)[0]),
        )
        np.savez(path, meta=json.dumps(series_meta, default=str), **arrays)
        print("OK    %s %s -> %s" % (entry["model_name"], series_name, path))

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(2)
