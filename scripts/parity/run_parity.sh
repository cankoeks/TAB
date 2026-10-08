#!/usr/bin/env bash
# Dump raw scores of the benchmark scripts' models for a parity check.
#
# usage: scripts/parity/run_parity.sh PYTHON TAB_ROOT OUT_DIR UNI_SERIES MULTI_SERIES [PATTERN]
#
#   PYTHON        interpreter of the environment to test (e.g. the legacy 3.8 one)
#   TAB_ROOT      TAB tree whose code is run (e.g. a worktree of upstream 5a7e6f8)
#   OUT_DIR       where the .npz files go (keep it outside both trees)
#   UNI_SERIES    space-separated univariate series names, e.g. "a.csv b.csv"
#   MULTI_SERIES  space-separated multivariate series names
#   PATTERN       optional extended regex on the script path, e.g. 'tods|KMeans'
#
# Every scripts/{univariate,multivariate}/{score,label}/*.sh matching PATTERN is
# run through dump_scores.py on the matching series. One line per script is
# written to OUT_DIR/run.log. Run it once with the reference environment and
# once with the new one, then compare with compare_scores.py.
set -u
PY=$1; ROOT=$2; OUT=$3; UNI=$4; MULTI=$5; PATTERN=${6:-.}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
cd "$ROOT"
for f in scripts/univariate/*/*.sh scripts/multivariate/*/*.sh; do
  [[ $f =~ $PATTERN ]] || continue
  case $f in scripts/univariate/*) series=$UNI ;; *) series=$MULTI ;; esac
  start=$(date +%s)
  # shellcheck disable=SC2086
  result=$("$PY" -W ignore "$HERE/dump_scores.py" --tab-root "$ROOT" --script "$f" \
    --series $series --out-dir "$OUT" 2>&1 | grep -E "^(OK|FAIL)" | awk '{print $1}' | sort | uniq -c | tr '\n' ' ')
  echo "$f $(( $(date +%s) - start ))s ${result:-ERROR}" | tee -a "$OUT/run.log"
done
