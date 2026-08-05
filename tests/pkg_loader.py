"""Load the runner package under the stable import name ``h3rt`` (idempotent).

The custom-node folder name contains dashes so it cannot be imported with the
usual ``import`` statement; ComfyUI loads it via ``spec_from_file_location``.
Tests register the same package under ``h3rt`` so relative imports inside
``models/`` / ``utils/`` resolve through a proper package hierarchy.
"""

import importlib.util
import os
import sys

_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_NAME = "h3rt"


def load():
    if _NAME in sys.modules:
        return sys.modules[_NAME]
    sys.path.insert(0, _PKG)
    spec = importlib.util.spec_from_file_location(_NAME, os.path.join(_PKG, "__init__.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_NAME] = mod
    spec.loader.exec_module(mod)
    return mod
