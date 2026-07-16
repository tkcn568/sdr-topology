from __future__ import annotations
from pathlib import Path

import numpy as np

from .capture.rtlsdr import CaptureMetadata, capture
from .capture.playback import load as load_capture
from .embedding.utils import get_magnitude
from .embedding.delay import embed as delay_embed, optimal_dim, optimal_tau
from .embedding.iq import embed as iq_embed
from .logging import logger
from .topology.persistence import compute
from .profiles.library import (
    EmbeddingParams,
    ProfileEntry,
    make_entry,
    save,
)


def run_iq(
    label: str,
    capture_path: Path | None = None,
    capture_params: dict | None = None,
    library_dir: Path = Path("profiles/data"),
    n_points: int = 2000,
    start: int = 0,
    stride: int = 1,
    maxdim: int = 1,
    notes: str = "",
    save_entry: bool = True,
) -> ProfileEntry:
    """
    End-to-end pipeline: IQ-plane embedding path.

    Either loads an existing capture from disk or runs a new capture,
    embeds via the IQ plane, computes persistence, and saves a profile entry.

    Parameters
    ----------
    label : str
        Signal type label for the profile entry.
    capture_path : Path, optional
        Path to an existing .npy capture file. If provided, skips capture.
        capture_params is ignored when capture_path is provided.
    capture_params : dict, optional
        Keyword arguments passed to capture.rtlsdr.capture() if no
        capture_path is provided. Required keys: center_freq_hz,
        sample_rate_hz, n_samples, output_path.
    library_dir : Path
        Profile library directory. Default "profiles/data".
    n_points : int
        Number of contiguous IQ samples to embed. Default 2000.
    start : int
        Start index into the capture. Default 0.
    stride : int
        Stride between samples. Default 1 (contiguous).
        See embedding.iq docstring for stride warnings.
    maxdim : int
        Maximum homology dimension for persistence computation. Default 1.
    notes : str
        Free-form notes for the profile entry.
    save_entry : bool
        If True, saves the entry to library_dir. Default True.

    Returns
    -------
    ProfileEntry
    """
    samples, metadata = _get_samples(capture_path, capture_params)

    point_cloud = iq_embed(
        samples=samples, n_points=n_points, start=start, stride=stride
    )
    diagram = compute(point_cloud, maxdim=maxdim)

    embedding_params = EmbeddingParams(
        method="iq",
        dim=None,
        tau=None,
        n_points=len(point_cloud),
        start=start,
        stride=stride,
    )

    entry = make_entry(
        label=label,
        capture_metadata=metadata,
        embedding_params=embedding_params,
        diagram=diagram,
        notes=notes,
    )

    if save_entry:
        save(entry=entry, library_dir=library_dir)

    return entry


def run_delay(
    label: str,
    capture_path: Path | None = None,
    capture_params: dict | None = None,
    library_dir: Path = Path("profiles/data"),
    n_points: int = 5000,
    dim: int | None = None,
    tau: int | None = None,
    max_dim: int = 5,
    max_tau: int = 50,
    maxdim: int = 1,
    notes: str = "",
    save_entry: bool = True,
) -> ProfileEntry:
    """
    End-to-end pipeline: time-delay embedding path.

    Either loads an existing capture from disk or runs a new capture,
    extracts magnitude, embeds via time-delay embedding (with automatic
    parameter selection if dim/tau not provided), computes persistence,
    and saves a profile entry.

    Parameters
    ----------
    label : str
        Signal type label for the profile entry.
    capture_path : Path, optional
        Path to an existing .npy capture file. If provided, skips capture.
    capture_params : dict, optional
        Keyword arguments passed to capture.rtlsdr.capture() if no
        capture_path is provided.
    library_dir : Path
        Profile library directory. Default "profiles/data".
    n_points : int
        Number of points to use from the magnitude signal. Default 5000.
    dim : int, optional
        Embedding dimension. If None, selected automatically via FNN.
    tau : int, optional
        Time delay in samples. If None, selected automatically via AMI.
    max_tau : int
        Maximum lag for automatic tau selection. Default 50.
    max_dim : int
        Maximum dimension for automatic dim selection. Default 5.
    maxdim : int
        Maximum homology dimension for persistence. Default 1.
    notes : str
        Free-form notes for the profile entry.
    save_entry : bool
        If True, saves the entry to library_dir. Default True.

    Returns
    -------
    ProfileEntry

    Notes
    -----
    The use of max_dim and maxdim can cause confusion and lead to collisions
    and very expensive calculations. max_dim should not be used, for example,
    in diagram computations &mdash; maxdim is the right variable for the compute()
    maxdim parameter.
    """
    samples, metadata = _get_samples(
        capture_path=capture_path, capture_params=capture_params
    )

    magnitude = get_magnitude(samples=samples)[:n_points]
    selected_tau = tau if tau is not None else optimal_tau(magnitude, max_tau=max_tau)
    # max_dim: ceiling for FNN embedding dimension search
    # maxdim: maximum homology dimension for Ripser — independent of embedding dim
    selected_dim = (
        dim
        if dim is not None
        else optimal_dim(magnitude, tau=selected_tau, max_dim=max_dim)
    )
    point_cloud = delay_embed(magnitude, dim=selected_dim, tau=selected_tau)
    diagram = compute(point_cloud, maxdim=maxdim)

    embedding_params = EmbeddingParams(
        method="delay",
        dim=selected_dim,
        tau=selected_tau,
        n_points=len(point_cloud),
        start=0,
        stride=1,
    )

    entry = make_entry(
        label=label,
        capture_metadata=metadata,
        embedding_params=embedding_params,
        diagram=diagram,
        notes=notes,
    )

    if save_entry:
        save(entry=entry, library_dir=library_dir)

    return entry


def _get_samples(
    capture_path: Path | None, capture_params: dict | None
) -> tuple[np.ndarray, CaptureMetadata]:
    """
    Load samples from disk or run a new capture.

    Parameters
    ----------
    capture_path : Path, optional
        Path to existing .npy capture file.
    capture_params : dict, optional
        Parameters for a new capture. Required if capture_path is None.

    Returns
    -------
    tuple[np.ndarray, CaptureMetadata]

    Raises
    ------
    ValueError
        If neither capture_path nor capture_params is provided.
    """
    if capture_path is not None:
        return load_capture(Path(capture_path))
    elif capture_params is not None:
        return capture(**capture_params)
    else:
        error_msg = "Either capture_path or capture_params must be provided."
        logger.error("Invalid capture data. " + error_msg)
        raise ValueError(error_msg)
