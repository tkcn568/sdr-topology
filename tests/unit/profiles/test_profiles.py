import numpy as np
import pytest
from sdr_topology.topology.persistence import compute
from sdr_topology.capture.rtlsdr import CaptureMetadata
from sdr_topology.profiles.library import (
    EmbeddingParams,
    ProfileEntry,
    make_entry,
    save,
    load,
    list_entries,
    query,
    _make_key,
)

# Synthetic circle diagram
N = 200
THETA = np.linspace(0, 2 * np.pi, N, endpoint=False)
CIRCLE = np.stack([np.cos(THETA), np.sin(THETA)], axis=1).astype(np.float32)
CIRCLE_DIAG = compute(CIRCLE, maxdim=1)

CAPTURE_META = CaptureMetadata(
    center_freq_hz=99_500_000,
    sample_rate_hz=250_000,
    gain="auto",
    n_samples=250_000,
    timestamp_utc="2026-06-30T21:45:27Z",
    device="RTL-SDR index 0",
    tuner="R820T2",
    environment_notes="unit test fixture",
)

EMBEDDING_PARAMS = EmbeddingParams(
    method="iq",
    dim=None,
    tau=None,
    n_points=200,
    start=0,
    stride=1,
)


class TestMakeKey:
    def test_filesystem_safe(self):
        key = _make_key("FM Broadcast / Strong Signal!", "2026-06-30T21:45:27Z")
        assert all(c.isalnum() or c == "_" for c in key)

    def test_includes_label(self):
        key = _make_key("fm_broadcast", "2026-06-30T21:45:27Z")
        assert "fm_broadcast" in key

    def test_includes_timestamp(self):
        key = _make_key("fm_broadcast", "2026-06-30T21:45:27Z")
        assert "20260630" in key


class TestMakeEntry:
    def test_returns_profile_entry(self):
        entry = make_entry(
            label="fm_broadcast_strong",
            capture_metadata=CAPTURE_META,
            embedding_params=EMBEDDING_PARAMS,
            diagram=CIRCLE_DIAG,
        )
        assert isinstance(entry, ProfileEntry)

    def test_key_auto_generated(self):
        entry = make_entry(
            label="fm_broadcast_strong",
            capture_metadata=CAPTURE_META,
            embedding_params=EMBEDDING_PARAMS,
            diagram=CIRCLE_DIAG,
        )
        assert len(entry.key) > 0

    def test_label_preserved(self):
        entry = make_entry(
            label="noise_floor",
            capture_metadata=CAPTURE_META,
            embedding_params=EMBEDDING_PARAMS,
            diagram=CIRCLE_DIAG,
        )
        assert entry.label == "noise_floor"


class TestSaveLoad:
    def test_roundtrip(self, tmp_path):
        entry = make_entry(
            label="fm_broadcast_strong",
            capture_metadata=CAPTURE_META,
            embedding_params=EMBEDDING_PARAMS,
            diagram=CIRCLE_DIAG,
        )
        save(entry, tmp_path)
        loaded = load(entry.key, tmp_path)

        assert loaded.key == entry.key
        assert loaded.label == entry.label
        assert loaded.notes == entry.notes

    def test_diagram_roundtrip(self, tmp_path):
        entry = make_entry(
            label="fm_broadcast_strong",
            capture_metadata=CAPTURE_META,
            embedding_params=EMBEDDING_PARAMS,
            diagram=CIRCLE_DIAG,
        )
        save(entry, tmp_path)
        loaded = load(entry.key, tmp_path)

        np.testing.assert_array_equal(loaded.diagram.h0, entry.diagram.h0)
        np.testing.assert_array_equal(loaded.diagram.h1, entry.diagram.h1)

    def test_capture_metadata_roundtrip(self, tmp_path):
        entry = make_entry(
            label="fm_broadcast_strong",
            capture_metadata=CAPTURE_META,
            embedding_params=EMBEDDING_PARAMS,
            diagram=CIRCLE_DIAG,
        )
        save(entry, tmp_path)
        loaded = load(entry.key, tmp_path)

        assert loaded.capture_metadata.center_freq_hz == 99_500_000
        assert loaded.capture_metadata.sample_rate_hz == 250_000

    def test_load_missing_key_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load("nonexistent_key", tmp_path)

    def test_creates_library_dir(self, tmp_path):
        new_dir = tmp_path / "new_library"
        entry = make_entry(
            label="test",
            capture_metadata=CAPTURE_META,
            embedding_params=EMBEDDING_PARAMS,
            diagram=CIRCLE_DIAG,
        )
        save(entry, new_dir)
        assert new_dir.exists()

    def test_npz_and_json_created(self, tmp_path):
        entry = make_entry(
            label="test",
            capture_metadata=CAPTURE_META,
            embedding_params=EMBEDDING_PARAMS,
            diagram=CIRCLE_DIAG,
        )
        save(entry, tmp_path)
        assert (tmp_path / f"{entry.key}.npz").exists()
        assert (tmp_path / f"{entry.key}.json").exists()


class TestListEntries:
    def test_empty_library(self, tmp_path):
        assert list_entries(tmp_path) == []

    def test_nonexistent_dir(self, tmp_path):
        assert list_entries(tmp_path / "missing") == []

    def test_lists_saved_entries(self, tmp_path):
        entry = make_entry(
            label="test",
            capture_metadata=CAPTURE_META,
            embedding_params=EMBEDDING_PARAMS,
            diagram=CIRCLE_DIAG,
        )
        save(entry, tmp_path)
        keys = list_entries(tmp_path)
        assert entry.key in keys


class TestQuery:
    def test_query_by_label(self, tmp_path):
        entry = make_entry(
            label="fm_broadcast_strong",
            capture_metadata=CAPTURE_META,
            embedding_params=EMBEDDING_PARAMS,
            diagram=CIRCLE_DIAG,
        )
        save(entry, tmp_path)
        results = query(tmp_path, label="fm_broadcast_strong")
        assert len(results) == 1
        assert results[0].label == "fm_broadcast_strong"

    def test_query_no_match(self, tmp_path):
        entry = make_entry(
            label="fm_broadcast_strong",
            capture_metadata=CAPTURE_META,
            embedding_params=EMBEDDING_PARAMS,
            diagram=CIRCLE_DIAG,
        )
        save(entry, tmp_path)
        results = query(tmp_path, label="noise_floor")
        assert results == []

    def test_query_by_frequency(self, tmp_path):
        entry = make_entry(
            label="fm_broadcast_strong",
            capture_metadata=CAPTURE_META,
            embedding_params=EMBEDDING_PARAMS,
            diagram=CIRCLE_DIAG,
        )
        save(entry, tmp_path)
        results = query(tmp_path, center_freq_hz=99_500_000)
        assert len(results) == 1

    def test_query_empty_library(self, tmp_path):
        results = query(tmp_path, label="anything")
        assert results == []
