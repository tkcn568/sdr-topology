# Integration test for capture functionality
# Requires hardware: RTL-SDR device
# Run from project root with `uv run pytest tests/functiona/test_capture_hardware.py`

from pathlib import Path
from sdr_topology.capture.rtlsdr import capture


def test_capture_kiss_fm():
    """Integration test: capture from WBAI 99.5 MHz."""
    samples, metadata = capture(
        center_freq_hz=99_500_000,
        sample_rate_hz=250_000,
        n_samples=250_000,
        output_path=Path("captures/wbai_fm_test"),
        gain="auto",
        environment_notes="WBAI 99.5 MHz New York, indoor antenna, integration test",
    )

    assert len(samples) > 0
    assert samples.dtype.name == "complex64"
    assert samples.real.min() < samples.real.max()
    assert samples.imag.min() < samples.imag.max()
    assert metadata.center_freq_hz == 99_500_000
    assert metadata.sample_rate_hz == 250_000
