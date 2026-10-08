"""Compare two directories of ``dump_scores.py`` outputs (reference vs. candidate).

    python scripts/parity/compare_scores.py ~/parity/ref ~/parity/new [--markdown]

For every file present in both directories it reports the output length,
Pearson / Spearman correlation and maximum absolute difference of the scores,
and AUC-PR / VUS-PR of both runs computed with TAB's own metric functions
(run from the TAB root so ``ts_benchmark`` is importable). Label outputs are
compared by their agreement rate.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, os.getcwd())


def _metrics(label, score):
    from ts_benchmark.evaluation.metrics.classification_metrics_score import VUS_PR, auc_pr

    n = min(len(label), len(score))
    # same zero padding as the pipeline for shorter (windowed) outputs
    padded = np.pad(score.astype(float), (0, len(label) - n)) if n < len(label) else score[: len(label)]
    return auc_pr(label, padded), VUS_PR(label, padded)


def _corr(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan"), float("nan")
    return pearsonr(a, b)[0], spearmanr(a, b)[0]


def compare(ref_path, new_path, with_metrics=True):
    ref, new = np.load(ref_path), np.load(new_path)
    meta = json.loads(str(new["meta"]))
    label = ref["test_label"].astype(float)
    rows = []
    for key in sorted(set(ref.files) & set(new.files)):
        if key in ("meta", "test_label") or key.endswith("_1"):
            continue
        a, b = ref[key].astype(float).ravel(), new[key].astype(float).ravel()
        row = {
            "model": meta["model_name"],
            "series": meta["series"],
            "output": key,
            "len_ref": len(a),
            "len_new": len(b),
        }
        if len(a) == len(b):
            row["max_abs_diff"] = float(np.max(np.abs(a - b))) if len(a) else 0.0
            if key.startswith("detect_label"):
                row["agreement"] = float(np.mean(a == b))
            else:
                row["pearson"], row["spearman"] = _corr(a, b)
                if with_metrics:
                    row["auc_pr_ref"], row["vus_pr_ref"] = _metrics(label, a)
                    row["auc_pr_new"], row["vus_pr_new"] = _metrics(label, b)
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("ref_dir")
    parser.add_argument("new_dir")
    parser.add_argument("--no-metrics", action="store_true", help="skip AUC-PR / VUS-PR")
    parser.add_argument("--markdown", action="store_true")
    opts = parser.parse_args()

    names = sorted(
        set(map(os.path.basename, glob.glob(os.path.join(opts.ref_dir, "*.npz"))))
        & set(map(os.path.basename, glob.glob(os.path.join(opts.new_dir, "*.npz"))))
    )
    columns = ["model", "series", "output", "len_ref", "len_new", "pearson", "spearman",
               "max_abs_diff", "agreement", "auc_pr_ref", "auc_pr_new", "vus_pr_ref", "vus_pr_new"]
    rows = []
    for name in names:
        rows.extend(compare(os.path.join(opts.ref_dir, name), os.path.join(opts.new_dir, name),
                            not opts.no_metrics))

    def fmt(value):
        if isinstance(value, float):
            return "%.4g" % value
        return "" if value is None else str(value)

    if opts.markdown:
        print("| " + " | ".join(columns) + " |")
        print("|" + "---|" * len(columns))
        for row in rows:
            print("| " + " | ".join(fmt(row.get(c)) for c in columns) + " |")
    else:
        for row in rows:
            print("  ".join("%s=%s" % (c, fmt(row.get(c))) for c in columns if c in row))
    print("\n%d files compared" % len(names), file=sys.stderr)


if __name__ == "__main__":
    main()
