from __future__ import annotations

import json
import subprocess  # in macOS, LIBUSB_ERROR_OVERFLOW errors occur with a sync process, but the CLI works
import tempfile
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from ..logging import logger


@dataclass
class CaptureMetadata:
    center_freq_hz: int
    sample_rate_hz: int
    gain: str | float  # values: ["auto", dB value]
    n_samples: int
    timestamp_utc: str
    device: str
    tuner: str
    environment_notes: str


def capture(
    center_freq_hz: int,
    sample_rate_hz: int,
    n_samples: int,
    output_path: Path,
    gain: str | float = "auto",
    environment_notes: str = "",
    device_index: int = 0,
    chunk_size: int = 2**14,
) -> tuple[np.ndarray, CaptureMetadata]:
    """
    Capture IQ samples from the RTL-SDR and save to disk.

    Saves two files:
        output_path.npy   — complex64 IQ samples
        output_path.json  — CaptureMetadata sidecar

    Returns the samples array and metadata.
    """
    output_path = Path(output_path)
    tmp_bin = Path(tempfile.mktemp(suffix=".bin"))

    gain_args = [] if gain == "auto" else ["-g", str(gain)]

    cmd = [
        "rtl_sdr",
        "-f",
        str(center_freq_hz),
        "-s",
        str(sample_rate_hz),
        "-n",
        str(n_samples),
        "-d",
        str(device_index),
        *gain_args,
        str(tmp_bin),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if not tmp_bin.exists() or tmp_bin.stat().st_size == 0:
        error_msg = f"rtl_sdr capture failed:\n{result.stderr}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    raw = np.fromfile(tmp_bin, dtype=np.uint8).astype(np.float32)
    raw -= 127.5
    samples = (raw[0::2] + 1j * raw[1::2]).astype(np.complex64)
    tmp_bin.unlink()

    if environment_notes == "":
        logger.warn(
            "environment_notes is empty. Notes should be provided in a production setting."
        )

    metadata = CaptureMetadata(
        center_freq_hz=center_freq_hz,
        sample_rate_hz=sample_rate_hz,
        gain=gain,
        n_samples=len(samples),
        timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        device=f"RTL-SDR index {device_index}",
        tuner="R820T2",
        environment_notes=environment_notes,
    )

    np.save(output_path.with_suffix(".npy"), samples)
    with open(output_path.with_suffix(".json"), "w") as f:
        json.dump(asdict(metadata), f, indent=2)

    return samples, metadata
