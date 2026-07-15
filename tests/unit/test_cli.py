import json
import numpy as np
import pytest
from dataclasses import asdict
from click.testing import CliRunner
from sdr_topology.cli import main
from sdr_topology.capture.rtlsdr import CaptureMetadata

# Synthetic capture fixture
N = 1000
RNG = np.random.default_rng(42)
THETA = np.linspace(0, 20 * np.pi, N)
SYNTHETIC = (
    (np.cos(THETA) + 0.1 * RNG.standard_normal(N)) +
    1j * (np.sin(THETA) + 0.1 * RNG.standard_normal(N))
).astype(np.complex64)

METADATA = CaptureMetadata(
    center_freq_hz=99_500_000,
    sample_rate_hz=250_000,
    gain="auto",
    n_samples=N,
    timestamp_utc="2026-06-30T21:45:27Z",
    device="RTL-SDR index 0",
    tuner="R820T2",
    environment_notes="cli unit test fixture",
)


@pytest.fixture
def capture_file(tmp_path):
    npy_path = tmp_path / "synthetic.npy"
    json_path = tmp_path / "synthetic.json"
    np.save(npy_path, SYNTHETIC)
    with open(json_path, "w") as f:
        json.dump(asdict(METADATA), f)
    return npy_path


@pytest.fixture
def runner():
    return CliRunner()


class TestMain:
    def test_help(self, runner):
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "SDR Topology" in result.output

    def test_subcommands_registered(self, runner):
        result = runner.invoke(main, ["--help"])
        assert "capture" in result.output
        assert "analyze" in result.output
        assert "profile" in result.output


class TestCapture:
    def test_help(self, runner):
        result = runner.invoke(main, ["capture", "--help"])
        assert result.exit_code == 0
        assert "--freq" in result.output

    def test_missing_required_freq(self, runner, tmp_path):
        result = runner.invoke(main, [
            "capture",
            "--output", str(tmp_path / "out"),
        ])
        assert result.exit_code != 0


class TestAnalyze:
    def test_help(self, runner):
        result = runner.invoke(main, ["analyze", "--help"])
        assert result.exit_code == 0

    def test_iq_method(self, runner, capture_file, tmp_path):
        result = runner.invoke(main, [
            "analyze",
            "--capture-path", str(capture_file),
            "--method", "iq",
            "--label", "test_iq",
            "--n-points", "200",
            "--library", str(tmp_path / "lib"),
        ])
        assert result.exit_code == 0
        assert "Analysis complete" in result.output

    def test_delay_method_explicit_params(self, runner, capture_file, tmp_path):
        result = runner.invoke(main, [
            "analyze",
            "--capture-path", str(capture_file),
            "--method", "delay",
            "--label", "test_delay",
            "--n-points", "300",
            "--dim", "2",
            "--tau", "5",
            "--library", str(tmp_path / "lib"),
        ])
        assert result.exit_code == 0
        assert "Analysis complete" in result.output

    def test_no_save_flag(self, runner, capture_file, tmp_path):
        lib = tmp_path / "lib"
        result = runner.invoke(main, [
            "analyze",
            "--capture-path", str(capture_file),
            "--method", "iq",
            "--label", "test_iq",
            "--n-points", "200",
            "--library", str(lib),
            "--no-save",
        ])
        assert result.exit_code == 0
        assert not lib.exists() or len(list(lib.glob("*.npz"))) == 0

    def test_missing_required_label(self, runner, capture_file, tmp_path):
        result = runner.invoke(main, [
            "analyze",
            "--capture-path", str(capture_file),
            "--method", "iq",
            "--n-points", "200",
            "--library", str(tmp_path / "lib"),
        ])
        assert result.exit_code != 0


class TestProfileList:
    def test_empty_library(self, runner, tmp_path):
        result = runner.invoke(main, [
            "profile", "list",
            "--library", str(tmp_path / "lib"),
        ])
        assert result.exit_code == 0
        assert "No entries" in result.output

    def test_lists_saved_entry(self, runner, capture_file, tmp_path):
        lib = tmp_path / "lib"
        runner.invoke(main, [
            "analyze",
            "--capture-path", str(capture_file),
            "--method", "iq",
            "--label", "fm_broadcast_strong",
            "--n-points", "200",
            "--library", str(lib),
        ])
        result = runner.invoke(main, [
            "profile", "list",
            "--library", str(lib),
        ])
        assert result.exit_code == 0
        assert "fm_broadcast_strong" in result.output


class TestProfileShow:
    def test_show_existing_entry(self, runner, capture_file, tmp_path):
        lib = tmp_path / "lib"
        analyze_result = runner.invoke(main, [
            "analyze",
            "--capture-path", str(capture_file),
            "--method", "iq",
            "--label", "test_show",
            "--n-points", "200",
            "--library", str(lib),
        ])
        assert analyze_result.exit_code == 0

        # Extract key from analyze output
        key = None
        for line in analyze_result.output.splitlines():
            if "Entry key:" in line:
                raw = line.split("Entry key:")[-1].strip()
                # Rich panel border appears after whitespace — take first token only
                key = raw.split()[0]
                break

        assert key is not None
        result = runner.invoke(main, [
            "profile", "show", key,
            "--library", str(lib),
        ])
        assert result.exit_code == 0
        assert "test_show" in result.output

    def test_show_missing_key(self, runner, tmp_path):
        result = runner.invoke(main, [
            "profile", "show", "nonexistent_key",
            "--library", str(tmp_path / "lib"),
        ])
        assert result.exit_code != 0


class TestProfileCompare:
    def test_help(self, runner):
        result = runner.invoke(main, ["profile", "compare", "--help"])
        assert result.exit_code == 0

    def test_compare_by_key(self, runner, capture_file, tmp_path):
        lib = tmp_path / "lib"

        # Save two entries
        r1 = runner.invoke(main, [
            "analyze",
            "--capture-path", str(capture_file),
            "--method", "iq",
            "--label", "test_a",
            "--n-points", "200",
            "--library", str(lib),
        ])
        r2 = runner.invoke(main, [
            "analyze",
            "--capture-path", str(capture_file),
            "--method", "iq",
            "--label", "test_b",
            "--n-points", "200",
            "--library", str(lib),
        ])

        def extract_key(output):
            for line in output.splitlines():
                if "Entry key:" in line:
                    # Strip Rich panel border characters and whitespace
                    return line.split("Entry key:")[-1].strip().rstrip("│").strip()
            return None

        key_a = extract_key(r1.output)
        key_b = extract_key(r2.output)
        assert key_a and key_b

        result = runner.invoke(main, [
            "profile", "compare",
            "--key-a", key_a,
            "--key-b", key_b,
            "--library", str(lib),
        ])
        assert result.exit_code == 0
        assert "Wasserstein" in result.output

    def test_compare_by_label(self, runner, capture_file, tmp_path):
        lib = tmp_path / "lib"

        # Save two entries with same label but different timestamps
        # Force different keys by adjusting notes
        for i in range(2):
            runner.invoke(main, [
                "analyze",
                "--capture-path", str(capture_file),
                "--method", "iq",
                "--label", "fm_broadcast_strong",
                "--n-points", "200",
                "--library", str(lib),
                "--notes", f"run {i}",
            ])

        result = runner.invoke(main, [
            "profile", "compare",
            "--label", "fm_broadcast_strong",
            "--library", str(lib),
        ])
        assert result.exit_code == 0

    def test_compare_no_options(self, runner, tmp_path):
        result = runner.invoke(main, [
            "profile", "compare",
            "--library", str(tmp_path / "lib"),
        ])
        assert result.exit_code != 0

    def test_compare_missing_key(self, runner, tmp_path):
        result = runner.invoke(main, [
            "profile", "compare",
            "--key-a", "missing_a",
            "--key-b", "missing_b",
            "--library", str(tmp_path / "lib"),
        ])
        assert result.exit_code != 0