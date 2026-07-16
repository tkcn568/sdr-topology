from __future__ import annotations
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from ..capture.rtlsdr import CaptureMetadata
from ..logging import logger
from ..topology.persistence import PersistenceDiagram
from ..topology.features import (
    lifetimes,
    max_persistence,
    betti_curve,
    betti_numbers,
)


@dataclass
class EmbeddingParams:
    """
    Parameters used to produce the point cloud for this profile entry.

    Stores enough information to reproduce the embedding from the raw
    capture without re-running parameter selection.
    """

    method: str  # "delay" or "iq"
    dim: int | None  # embedding dimension (delay only)
    tau: int | None  # time delay in samples (delay only)
    n_points: int  # number of points in the point cloud
    start: int = 0  # start index into capture (iq only)
    stride: int = 1  # stride (iq only)


@dataclass
class ProfileEntry:
    """
    A single profile library entry: one signal type, one capture, one embedding.

    Attributes
    ----------
    key : str
        Unique identifier for this entry. Derived from label + timestamp.
    label : str
        Signal type label (e.g. "fm_broadcast_strong", "noise_floor").
        Free string — no enforced vocabulary.
    capture_metadata : CaptureMetadata
        Hardware and capture parameters from capture/rtlsdr.py.
    embedding_params : EmbeddingParams
        Parameters used to produce the point cloud.
    diagram : PersistenceDiagram
        Persistence diagram computed from this entry's point cloud.
    notes : str
        Free-form notes about this specific entry — signal conditions,
        anomalies, or anything not captured by structured metadata.
    """

    key: str
    label: str
    capture_metadata: CaptureMetadata
    embedding_params: EmbeddingParams
    diagram: PersistenceDiagram
    notes: str = ""


def _make_key(label: str, timestamp: str) -> str:
    """Stable, filesystem-safe key from label and timestamp."""
    slug = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    ts = re.sub(r"[^0-9T]", "", timestamp)
    return f"{slug}_{ts}"


def _entry_paths(library_dir: Path, key: str) -> tuple[Path, Path]:
    """Return (.npz path, .json path) for a given key."""
    return (
        library_dir / f"{key}.npz",
        library_dir / f"{key}.json",
    )


def save(entry: ProfileEntry, library_dir: Path) -> None:
    """
    Save a profile entry to the library directory.

    Saves two files:
        {key}.npz  — diagram arrays and extracted feature arrays
        {key}.json — all non-array fields as JSON

    Parameters
    ----------
    entry : ProfileEntry
    library_dir : Path
        Directory to save into. Created if it does not exist.
    """
    library_dir = Path(library_dir)
    library_dir.mkdir(parents=True, exist_ok=True)

    npz_path, json_path = _entry_paths(library_dir=library_dir, key=entry.key)

    # Extract features for storage alongside diagram
    lt_h0 = lifetimes(entry.diagram, dim=0)
    lt_h1 = lifetimes(entry.diagram, dim=1)
    eps_h1, betti_h1 = betti_curve(entry.diagram, dim=1)

    np.savez(
        npz_path,
        h0=entry.diagram.h0,
        h1=entry.diagram.h1,
        h2=entry.diagram.h2,
        lifetimes_h0=lt_h0,
        lifetimes_h1=lt_h1,
        betti_curve_eps=eps_h1,
        betti_curve_counts=betti_h1,
    )

    sidecar = {
        "key": entry.key,
        "label": entry.label,
        "notes": entry.notes,
        "capture_metadata": asdict(entry.capture_metadata),
        "embedding_params": asdict(entry.embedding_params),
        "diagram_meta": {
            "maxdim": entry.diagram.maxdim,
            "metric": entry.diagram.metric,
        },
        "features": {
            "max_persistence_h0": float(max_persistence(entry.diagram, dim=0)),
            "max_persistence_h1": float(max_persistence(entry.diagram, dim=1)),
            "betti_number_h1_default": int(betti_numbers(entry.diagram, dim=1)),
        },
    }

    with open(json_path, "w") as f:
        json.dump(sidecar, f, indent=2)


def load(key: str, library_dir: Path) -> ProfileEntry:
    """
    Load a profile entry by key.

    Parameters
    ----------
    key : str
        Entry key as returned by _make_key or stored in the JSON sidecar.
    library_dir : Path

    Returns
    -------
    ProfileEntry

    Raises
    ------
    FileNotFoundError
        If no entry with this key exists in library_dir.
    """
    library_dir = Path(library_dir)
    npz_path, json_path = _entry_paths(library_dir=library_dir, key=key)

    if not npz_path.exists() or not json_path.exists():
        error_msg = f"No profile entry with key '{key}' in {library_dir}."
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    with open(json_path) as f:
        sidecar = json.load(f)

    arrays = np.load(npz_path)

    diagram = PersistenceDiagram(
        h0=arrays["h0"],
        h1=arrays["h1"],
        h2=arrays["h2"],
        maxdim=sidecar["diagram_meta"]["maxdim"],
        metric=sidecar["diagram_meta"]["metric"],
    )

    return ProfileEntry(
        key=sidecar["key"],
        label=sidecar["label"],
        notes=sidecar["notes"],
        capture_metadata=CaptureMetadata(**sidecar["capture_metadata"]),
        embedding_params=EmbeddingParams(**sidecar["embedding_params"]),
        diagram=diagram,
    )


def list_entries(library_dir: Path) -> list[str]:
    """
    List all entry keys in the library directory.

    Parameters
    ----------
    library_dir : Path

    Returns
    -------
    list[str]
        Sorted list of entry keys. Empty if directory does not exist
        or contains no profile entries.
    """
    library_dir = Path(library_dir)
    if not library_dir.exists():
        return []
    return sorted(p.stem for p in library_dir.glob("*.json"))


def query(
    library_dir: Path,
    label: str | None = None,
    center_freq_hz: int | None = None,
    sample_rate_hz: int | None = None,
) -> list[ProfileEntry]:
    """
    Query the profile library by signal type and/or capture parameters.

    Loads all matching entries. For small libraries (tens of entries)
    this is a linear scan — no index required.

    Parameters
    ----------
    library_dir : Path
    label : str, optional
        Filter by exact signal type label.
    center_freq_hz : int, optional
        Filter by center frequency.
    sample_rate_hz : int, optional
        Filter by sample rate.

    Returns
    -------
    list[ProfileEntry]
        All entries matching all provided filters.
        Empty list if no matches or library is empty.
    """
    keys = list_entries(library_dir=library_dir)
    results = []

    for key in keys:
        entry = load(key=key, library_dir=library_dir)

        if label is not None and entry.label != label:
            continue
        if (
            center_freq_hz is not None
            and entry.capture_metadata.center_freq_hz != center_freq_hz
        ):
            continue
        if (
            sample_rate_hz is not None
            and entry.capture_metadata.sample_rate_hz != sample_rate_hz
        ):
            continue

        results.append(entry)

    return results


def make_entry(
    label: str,
    capture_metadata: CaptureMetadata,
    embedding_params: EmbeddingParams,
    diagram: PersistenceDiagram,
    notes: str = "",
) -> ProfileEntry:
    """
    Construct a ProfileEntry with an auto-generated key.

    Parameters
    ----------
    label : str
        Signal type label.
    capture_metadata : CaptureMetadata
    embedding_params : EmbeddingParams
    diagram : PersistenceDiagram
    notes : str

    Returns
    -------
    ProfileEntry
    """
    key = _make_key(label=label, timestamp=capture_metadata.timestamp_utc)
    return ProfileEntry(
        key=key,
        label=label,
        capture_metadata=capture_metadata,
        embedding_params=embedding_params,
        diagram=diagram,
        notes=notes,
    )
