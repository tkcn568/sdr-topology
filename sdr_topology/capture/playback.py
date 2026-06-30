from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .rtlsdr import CaptureMetadata


def load(npy_path: Path) -> tuple[np.ndarray, CaptureMetadata]:
    """
    Load a previously captured IQ sample file and its sidecar metadata.

    Expects:
        npy_path       — path to .npy file (complex64)
        npy_path.json  — sidecar metadata file
    """
    npy_path = Path(npy_path)
    samples = np.load(npy_path)

    json_path = npy_path.with_suffix(".json")
    with open(json_path) as f:
        meta_dict = json.load(f)

    metadata = CaptureMetadata(**meta_dict)
    return samples, metadata
