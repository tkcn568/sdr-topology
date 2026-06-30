"""Unit tests for capture module."""
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from sdr_topology.capture.rtlsdr import CaptureMetadata
from sdr_topology.capture.playback import load


class TestCaptureMetadata:
    """Tests for CaptureMetadata dataclass."""

    def test_init(self):
        """Test basic initialization."""
        meta = CaptureMetadata(
            center_freq_hz=99_500_000,
            sample_rate_hz=250_000,
            gain="auto",
            n_samples=250_000,
            timestamp_utc="2026-06-30T12:34:56Z",
            device="RTL-SDR index 0",
            tuner="R820T2",
            environment_notes="Test capture",
        )
        assert meta.center_freq_hz == 99_500_000
        assert meta.sample_rate_hz == 250_000
        assert meta.gain == "auto"
        assert meta.n_samples == 250_000

    def test_gain_as_float(self):
        """Test that gain accepts numeric values."""
        meta = CaptureMetadata(
            center_freq_hz=100_000_000,
            sample_rate_hz=250_000,
            gain=20.0,
            n_samples=1000,
            timestamp_utc="2026-06-30T12:34:56Z",
            device="RTL-SDR index 0",
            tuner="R820T2",
            environment_notes="",
        )
        assert meta.gain == 20.0


class TestLoadPlayback:
    """Tests for playback.load() function."""

    def test_load_capture(self):
        """Test loading a saved capture with metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create test data
            samples = np.array([1+2j, 3+4j, 5+6j], dtype=np.complex64)
            np.save(tmpdir / "test.npy", samples)

            # Create metadata sidecar
            metadata_dict = {
                "center_freq_hz": 100_000_000,
                "sample_rate_hz": 250_000,
                "gain": "auto",
                "n_samples": 3,
                "timestamp_utc": "2026-06-30T12:34:56Z",
                "device": "RTL-SDR index 0",
                "tuner": "R820T2",
                "environment_notes": "Test",
            }
            with open(tmpdir / "test.json", "w") as f:
                json.dump(metadata_dict, f)

            # Load and verify
            loaded_samples, loaded_meta = load(tmpdir / "test.npy")

            assert np.allclose(loaded_samples, samples)
            assert loaded_meta.center_freq_hz == 100_000_000
            assert loaded_meta.sample_rate_hz == 250_000
            assert loaded_meta.n_samples == 3
            assert loaded_meta.environment_notes == "Test"

    def test_load_preserves_dtype(self):
        """Test that loading preserves complex64 dtype."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            samples = np.random.random(100).astype(np.complex64) + \
                      1j * np.random.random(100).astype(np.complex64)
            np.save(tmpdir / "test.npy", samples)

            metadata_dict = {
                "center_freq_hz": 100_000_000,
                "sample_rate_hz": 250_000,
                "gain": "auto",
                "n_samples": 100,
                "timestamp_utc": "2026-06-30T12:34:56Z",
                "device": "RTL-SDR index 0",
                "tuner": "R820T2",
                "environment_notes": "",
            }
            with open(tmpdir / "test.json", "w") as f:
                json.dump(metadata_dict, f)

            loaded_samples, _ = load(tmpdir / "test.npy")
            assert loaded_samples.dtype == np.complex64

    def test_load_missing_metadata_file(self):
        """Test that loading fails gracefully without metadata sidecar."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            samples = np.array([1+2j], dtype=np.complex64)
            np.save(tmpdir / "test.npy", samples)

            with pytest.raises(FileNotFoundError):
                load(tmpdir / "test.npy")
