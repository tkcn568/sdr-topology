from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from ripser import ripser
from ..logging import logger


@dataclass
class PersistenceDiagram:
    """
    Persistence diagrams for a single point cloud computation.

    Attributes
    ----------
    h0 : np.ndarray
        H0 (connected components) birth/death pairs, shape (N, 2).
        One point persists to infinity (the last connected component);
        represented as np.inf in the death column.
    h1 : np.ndarray
        H1 (loops) birth/death pairs, shape (N, 2).
        Empty if no loops detected or maxdim < 1.
    h2 : np.ndarray
        H2 (voids) birth/death pairs, shape (N, 2).
        Empty if maxdim < 2.
    maxdim : int
        Maximum homology dimension computed.
    metric : str
        Distance metric used for filtration.

    Notes
    -----
    Birth/death pairs are in filtration parameter units (Euclidean distance
    by default). A feature with death=np.inf persists across the entire
    filtration — for H0 this is expected (one connected component survives);
    for H1 it indicates a loop that never closes within the point cloud scale,
    which may signal insufficient data or a degenerate point cloud.
    """

    h0: np.ndarray
    h1: np.ndarray
    h2: np.ndarray
    maxdim: int
    metric: str | callable

    def __post_init__(self):
        self.h0 = np.array(self.h0, dtype=np.float32)
        self.h1 = np.array(self.h1, dtype=np.float32)
        self.h2 = np.array(self.h2, dtype=np.float32)

    @property
    def diagrams(self) -> list[np.ndarray]:
        """All diagrams as a list ordered by dimension, for persim compatibility."""
        return [self.h0, self.h1, self.h2]


def compute(
    point_cloud: np.ndarray,
    *,
    maxdim: int = 1,
    metric: str | callable = "euclidean",
    distance_matrix: bool = False,
) -> PersistenceDiagram:
    """
    Compute persistent homology of a point cloud.

    Wraps Ripser (Vietoris-Rips persistent homology) and returns a
    PersistenceDiagram dataclass for use in feature extraction and
    profile library storage.

    Parameters
    ----------
    point_cloud : np.ndarray
        Either a point cloud of shape (N, d) or a precomputed distance
        matrix of shape (N, N), depending on distance_matrix flag.
    maxdim : int
        Maximum homology dimension to compute. Default 1 (H0 + H1).
        Use 2 for 3D embeddings. H2 is not meaningful for 2D point clouds.
    metric : str
        Distance metric for Vietoris-Rips filtration. Default "euclidean".
        Passed directly to Ripser — see Ripser documentation for options.
        Ignored if distance_matrix=True.
    distance_matrix : bool
        If True, point_cloud is interpreted as a precomputed distance matrix.
        Avoids redundant pairwise distance computation when comparing multiple
        diagrams from the same point cloud. Default False.

    Returns
    -------
    PersistenceDiagram
        Dataclass containing h0, h1, h2 arrays and computation metadata.

    Raises
    ------
    ValueError
        If point_cloud is 1D (must be at least 2D point cloud or distance matrix).
        If maxdim < 0.
        If distance_matrix=True and point_cloud is not square.

    Notes
    -----
    Ripser complexity scales as O(N^2) in memory and O(N^3) in time for
    dense point clouds. For large captures, subsample before calling compute()
    rather than passing all samples. Practical upper bound is ~5000 points
    on a laptop without significant wait time.

    For 2D point clouds (IQ-plane or 2D delay embedding), maxdim=1 is
    sufficient — H2 requires at least a 3D point cloud to be meaningful.
    """
    if point_cloud.ndim < 2:
        error_msg = f"""
        point_cloud must be at least 2D, got shape {point_cloud.shape}.

        For 1D signals, apply an embedding first
        (sdr_topology.embedding.delay or sdr_topology.embedding.iq).
        """
        logger.error(error_msg)
        raise ValueError(error_msg)

    if maxdim < 0:
        error_msg = f"maxdim must be at least 1, got {maxdim}."
        logger.error("Invalid maxdim value. " + error_msg)
        raise ValueError(error_msg)

    if distance_matrix and point_cloud.shape[0] != point_cloud.shape[1]:
        error_msg = f"""
        point_cloud must be a square shape if distance_matrix=True.
        Got shape ({point_cloud.shape}).
        """
        logger.error(error_msg)
        raise ValueError(error_msg)

    dgms = ripser(
        point_cloud,
        metric=metric if not distance_matrix else "precomputed",
        maxdim=maxdim,
    )["dgms"]

    # Ripser always returns at least H0. Empty values can be padded.
    h0 = dgms[0] if len(dgms) > 0 else np.empty((0, 2), dtype=np.float32)
    h1 = dgms[1] if len(dgms) > 1 else np.empty((0, 2), dtype=np.float32)
    h2 = dgms[2] if len(dgms) > 2 else np.empty((0, 2), dtype=np.float32)

    return PersistenceDiagram(h0=h0, h1=h1, h2=h2, maxdim=maxdim, metric=metric)
