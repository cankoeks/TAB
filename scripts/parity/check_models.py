"""Resolve (and optionally instantiate) every model used by TAB's benchmark scripts.

Run from the TAB root:

    python scripts/parity/check_models.py                # import / resolve only
    python scripts/parity/check_models.py --instantiate  # also build each model

Every ``run_benchmark.py`` invocation under ``scripts/`` is parsed and passed
through the same model loader (``ts_benchmark.models.get_models``) that the
pipeline uses. Exit status is non-zero if any model fails.
"""
import argparse
import json
import os
import sys
import traceback

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from script_args import parse_scripts  # noqa: E402
from ts_benchmark.models.model_loader import get_models  # noqa: E402


def build_factory(entry):
    with open(os.path.join("config", entry["config_path"])) as f:
        config = json.load(f)
    # same structure as build_model_config() in scripts/run_benchmark.py
    model_config = {
        "adapter": entry["adapter"],
        "model_name": entry["model_name"],
        "model_hyper_params": entry["hyper_params"],
    }
    all_model_config = {
        "models": [model_config],
        "recommend_model_hyper_params": config["model_config"].get(
            "recommend_model_hyper_params", {}
        ),
    }
    return get_models(all_model_config)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instantiate", action="store_true")
    parser.add_argument("--only", nargs="*", help="restrict to these model names")
    opts = parser.parse_args()

    failures = []
    entries = list(parse_scripts())
    for entry in entries:
        if opts.only and entry["model_name"] not in opts.only:
            continue
        label = "%s (%s)" % (entry["model_name"], entry["script"])
        try:
            factory = build_factory(entry)
            if opts.instantiate:
                factory()
            print("OK    " + label)
        except Exception:
            print("FAIL  " + label)
            traceback.print_exc()
            failures.append(label)

    print("\n%d invocations checked, %d failed" % (len(entries), len(failures)))
    for label in failures:
        print("  " + label)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
