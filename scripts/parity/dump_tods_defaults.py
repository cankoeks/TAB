"""Dump the effective hyperparameters of the original d3m-based TODS wrappers.

Run with the legacy Python 3.8 environment (requirements-legacy-py38.txt) from the
TAB root:

    python scripts/parity/dump_tods_defaults.py > tods_defaults.json

For every TODS model it prints
  * ``d3m``: the primitive's ``Hyperparams.defaults()`` -- exactly what ``BaseSKI``
    used when TAB's scripts passed ``--model-hyper-params '{}'``;
  * ``pyod``: ``get_params()`` of the pyod estimator the primitive constructed, i.e.
    the effective arguments including pyod defaults the primitive did not pass.

These dumps are the source of ``_LEGACY_DEFAULTS`` in
``ts_benchmark/baselines/tods/tods_models.py``.
"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join("ts_benchmark", "baselines", "tods", "third_party"))

from tods.detection_algorithm.PyodHBOS import HBOSPrimitive
from tods.detection_algorithm.PyodKNN import KNNPrimitive
from tods.detection_algorithm.PyodLOF import LOFPrimitive
from tods.detection_algorithm.PyodOCSVM import OCSVMPrimitive
from tods.detection_algorithm.PyodLODA import LODAPrimitive
from tods.detection_algorithm.PCAODetect import PCAODetectorPrimitive
from tods.detection_algorithm.PyodCBLOF import CBLOFPrimitive
from tods.detection_algorithm.PyodIsolationForest import IsolationForestPrimitive
from tods.detection_algorithm.PyodCOF import COFPrimitive

PRIMITIVES = {
    "HBOSSKI": HBOSPrimitive,
    "KNNSKI": KNNPrimitive,
    "LOFSKI": LOFPrimitive,
    "OCSVMSKI": OCSVMPrimitive,
    "LODASKI": LODAPrimitive,
    "PCAODetectorSKI": PCAODetectorPrimitive,
    "CBLOFSKI": CBLOFPrimitive,
    "IsolationForestSKI": IsolationForestPrimitive,
    "COFSKI": COFPrimitive,
}


def _plain(value):
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_plain(v) for v in value]
    return repr(value)


def main():
    out = {}
    for name, primitive in PRIMITIVES.items():
        defaults = primitive.metadata.get_hyperparams().defaults()
        instance = primitive(hyperparams=defaults)
        out[name] = {
            "d3m": {k: _plain(v) for k, v in defaults.items()},
            "pyod_class": type(instance._clf).__module__ + "." + type(instance._clf).__name__,
            "pyod": {k: _plain(v) for k, v in instance._clf.get_params(deep=False).items()},
        }
    json.dump(out, sys.stdout, indent=2, sort_keys=True)
    print()


if __name__ == "__main__":
    main()
