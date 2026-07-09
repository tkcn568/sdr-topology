from __future__ import annotations
from ..logging import logger
import numpy as np


def embed(
    samples: np.ndarray,
    n_points: int | None = None,
    start: int = 0,
    stride: int = 1,
) -> np.ndarray:
    """
    IQ-plane embedding: treat complex IQ samples as a 2D point cloud.

    Maps each sample I + jQ to a point (I, Q) in R^2. No parameter
    selection required — the embedding dimension is fixed at 2 by the
    IQ plane geometry.

    Parameters
    ----------
    samples : np.ndarray
        Complex64 IQ samples as returned by capture modules.
    n_points : int, optional
        Number of points to include. If None, uses all available samples
        after applying start and stride.
    start : int
        Index of first sample to include. Default 0.
    stride : int
        Step between samples. stride=1 gives contiguous samples (recommended
        for preserving trajectory structure). stride>1 subsamples the
        trajectory — use with caution, as striding breaks temporal adjacency
        and distorts the point cloud geometry. See writeup null result.

    Returns
    -------
    np.ndarray
        Shape (n_points, 2), float32. Column 0 is I, column 1 is Q.

    Notes
    -----
    Contiguous sampling (stride=1) is strongly preferred over strided
    subsampling. Experiments on FM broadcast IQ at 250 kS/s showed that
    strided point clouds produce distorted persistence diagrams relative
    to contiguous windows of equivalent size. See null result documentation.
    """
    if not np.iscomplexobj(samples):
        error_msg = f"""
        samples must be complex, got dtype={samples.dtype}.
        Pass complex64 IQ samples directly -- use sdr_topology.embedding.delay
        for real-valued time series.
        """
        logger.error(error_msg)
        raise ValueError(error_msg)

    if stride < 1:
        error_msg = f"stride must be an integer > 0. Got stride={stride}."
        logger.error("stride has an invalid value. " + error_msg)
        raise ValueError(error_msg)

    if start < 0 or start >= len(samples):
        error_msg = f"""
        start value out of range. Got start={start},
        but start must be an integer in the range [0..||samples|| - 1].
        """
        logger.error("Invalid start value. " + error_msg)
        raise ValueError(error_msg)

    windowed = samples[start::stride]

    if n_points is not None:
        if n_points < 1:
            error_msg = f"n_points must be >= 1. Got n_points={n_points}."
            logger.error("Invalid n_points value. " + error_msg)
            raise ValueError(error_msg)

        windowed = windowed[:n_points]

    return np.stack([windowed.real, windowed.imag], axis=1).astype(np.float32)
