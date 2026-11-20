"""Shared result-saving utilities used by every experiment script in ``exp/``.

Keeps JSON serialization of numpy scalars/arrays and the results directory
path in one place so all four experiments write artifacts the same way.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that converts numpy scalars/arrays to native Python types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def save_json(filename: str, payload: dict) -> Path:
    """Write ``payload`` as pretty-printed JSON to ``results/<filename>``."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / filename
    path.write_text(json.dumps(payload, indent=2, cls=NumpyEncoder))
    return path


def results_path(filename: str) -> Path:
    """Return ``results/<filename>``, creating the directory if needed."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR / filename
