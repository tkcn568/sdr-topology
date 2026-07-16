import numpy as np
import pytest
from sdr_topology.capture.rtlsdr import CaptureMetadata
from sdr_topology.profiles.library import ProfileEntry
from sdr_topology.pipeline import run_iq, run_delay

# Synthetic capture fixture — saved to tmp_path in each test
N = 1_000
RNG = np.random.default_rng(42)
# Simulate a noisy rotating IQ signal
THETA = np.linspace(0, 40 * np.pi, N)
SYNTHETIC = (
    (np.cos(THETA) + 0.1 * RNG.standard_normal(N))
    + 1j * (np.sin(THETA) + 0.1 * RNG.standard_normal(N))
).astype(np.complex64)

METADATA = CaptureMetadata(
    center_freq_hz=99_500_000,
    sample_rate_hz=250_000,
    gain="auto",
    n_samples=N,
    timestamp_utc="2026-06-30T21:45:27Z",
    device="RTL-SDR index 0",
    tuner="R820T2",
    environment_notes="pipeline unit test fixture",
)


@pytest.fixture
def capture_file(tmp_path):
    """Save synthetic IQ samples and metadata to tmp_path."""
    import json
    from dataclasses import asdict

    npy_path = tmp_path / "synthetic.npy"
    json_path = tmp_path / "synthetic.json"
    np.save(npy_path, SYNTHETIC)
    with open(json_path, "w") as f:
        json.dump(asdict(METADATA), f)
    return npy_path


class TestRunIQ:
    def test_returns_profile_entry(self, capture_file, tmp_path):
        entry = run_iq(
            label="test_iq",
            capture_path=capture_file,
            library_dir=tmp_path / "lib",
            n_points=500,
        )
        assert isinstance(entry, ProfileEntry)

    def test_label_preserved(self, capture_file, tmp_path):
        entry = run_iq(
            label="fm_broadcast_strong",
            capture_path=capture_file,
            library_dir=tmp_path / "lib",
            n_points=500,
        )
        assert entry.label == "fm_broadcast_strong"

    def test_embedding_method_recorded(self, capture_file, tmp_path):
        entry = run_iq(
            label="test_iq",
            capture_path=capture_file,
            library_dir=tmp_path / "lib",
            n_points=500,
        )
        assert entry.embedding_params.method == "iq"

    def test_saves_to_library(self, capture_file, tmp_path):
        lib = tmp_path / "lib"
        entry = run_iq(
            label="test_iq",
            capture_path=capture_file,
            library_dir=lib,
            n_points=500,
        )
        assert (lib / f"{entry.key}.npz").exists()
        assert (lib / f"{entry.key}.json").exists()

    def test_save_false_does_not_write(self, capture_file, tmp_path):
        lib = tmp_path / "lib"
        entry = run_iq(
            label="test_iq",
            capture_path=capture_file,
            library_dir=lib,
            n_points=500,
            save_entry=False,
        )
        assert not (lib / f"{entry.key}.npz").exists()

    def test_raises_without_capture_source(self, tmp_path):
        with pytest.raises(ValueError, match="capture_path or capture_params"):
            run_iq(label="test", library_dir=tmp_path / "lib")


class TestRunDelay:
    def test_returns_profile_entry(self, capture_file, tmp_path):
        entry = run_delay(
            label="test_delay",
            capture_path=capture_file,
            library_dir=tmp_path / "lib",
            n_points=500,
            max_tau=10,
            max_dim=2,
        )
        assert isinstance(entry, ProfileEntry)

    def test_embedding_method_recorded(self, capture_file, tmp_path):
        entry = run_delay(
            label="test_delay",
            capture_path=capture_file,
            library_dir=tmp_path / "lib",
            n_points=500,
            max_tau=10,
            max_dim=2,
        )
        assert entry.embedding_params.method == "delay"

    def test_explicit_tau_and_dim_used(self, capture_file, tmp_path):
        entry = run_delay(
            label="test_delay",
            capture_path=capture_file,
            library_dir=tmp_path / "lib",
            n_points=500,
            dim=2,
            tau=10,
        )
        assert entry.embedding_params.tau == 10
        assert entry.embedding_params.dim == 2

    def test_auto_tau_and_dim_selected(self, capture_file, tmp_path):
        entry = run_delay(
            label="test_delay",
            capture_path=capture_file,
            library_dir=tmp_path / "lib",
            n_points=500,
            max_tau=10,
            max_dim=3,
        )
        assert entry.embedding_params.tau is not None
        assert entry.embedding_params.dim is not None

    def test_saves_to_library(self, capture_file, tmp_path):
        lib = tmp_path / "lib"
        entry = run_delay(
            label="test_delay",
            capture_path=capture_file,
            library_dir=lib,
            n_points=2000,
            max_tau=20,
            max_dim=3,
        )
        assert (lib / f"{entry.key}.npz").exists()

    def test_raises_without_capture_source(self, tmp_path):
        with pytest.raises(ValueError, match="capture_path or capture_params"):
            run_delay(label="test", library_dir=tmp_path / "lib")
