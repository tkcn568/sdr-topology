from __future__ import annotations

import numpy as np
from persim import wasserstein
from ..logging import logger
from .persistence import PersistenceDiagram


def _finite_pairs(diagram: PersistenceDiagram, dim: int) -> np.ndarray:
    """
    Extract finite birth/death pairs for a given homology dimension.

    Parameters
    ----------
    diagram : PersistenceDiagram
    dim : int
        Homology dimension (0, 1, or 2).

    Returns
    -------
    np.ndarray
        Shape (N, 2), only rows where death is finite.
    """
    if dim not in (0, 1, 2):
        error_msg = f"dim must be 0, 1, or 2, go {dim}."
        logger.error(error_msg)
        raise ValueError(error_msg)

    dgm = diagram.diagrams[dim]
    if len(dgm) == 0:
        return np.empty((0, 2), dtype=np.float32)

    return dgm[np.isfinite(dgm[:, 1])]


def lifetimes(diagram: PersistenceDiagram, dim: int = 1) -> np.ndarray:
    """
    Lifetimes (death - birth) of finite persistence features.

    Parameters
    ----------
    diagram : PersistenceDiagram
    dim : int
        Homology dimension. Default 1 (loops).

    Returns
    -------
    np.ndarray
        Float32 array of lifetimes, sorted descending.
        Empty array if no finite features exist.
    """
    pairs = _finite_pairs(diagram, dim)

    if len(pairs) == 0:
        return np.empty(0, dtype=np.float32)

    lt = (pairs[:, 1] - pairs[:, 0]).astype(np.float32)
    return np.sort(lt)[::-1]


def max_persistence(diagram: PersistenceDiagram, dim: int = 1) -> float:
    """
    Maximum finite lifetime in a homology dimension.

    Parameters
    ----------
    diagram : PersistenceDiagram
    dim : int
        Homology dimension. Default 1 (loops).

    Returns
    -------
    float
        Maximum lifetime, or 0.0 if no finite features exist.
    """
    lt = lifetimes(diagram, dim)
    return float(lt[0]) if len(lt) > 0 else 0.0


def betti_numbers(
    diagram: PersistenceDiagram,
    dim: int = 1,
    epsilon: float | None = None,
) -> int:
    """
    Betti number at a specific filtration threshold.

    Convenience wrapper around betti_curve for a single epsilon value.
    If epsilon is None, uses the median death value of finite features
    as a sensible mid-filtration default.

    Parameters
    ----------
    diagram : PersistenceDiagram
    dim : int
        Homology dimension. Default 1.
    epsilon : float, optional
        Filtration threshold. If None, defaults to median death of
        finite features in this dimension.

    Returns
    -------
    int
        Number of H_dim features alive at epsilon.
        Returns 0 if no finite features exist.
    """
    pairs = _finite_pairs(diagram, dim)
    if len(pairs) == 0:
        return 0

    if epsilon is None:
        epsilon = float(np.median(pairs[:, 1])) * 0.9

    return int(np.sum((pairs[:, 0] <= epsilon) & (pairs[:, 1] > epsilon)))


def betti_curve(
    diagram: PersistenceDiagram,
    dim: int = 1,
    n_steps: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Betti curve: count of H_dim features alive at each filtration threshold.

    The Betti number beta_k(epsilon) counts features in H_k where
    birth <= epsilon < death. Returns the curve sampled at n_steps
    evenly spaced epsilon values spanning the range of finite features.

    Parameters
    ----------
    diagram : PersistenceDiagram
    dim : int
        Homology dimension. Default 1.
    n_steps : int
        Number of epsilon values to sample. Default 100.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (epsilons, betti_counts) — both shape (n_steps,).
        epsilons: float32 filtration parameter values.
        betti_counts: int32 number of features alive at each epsilon.

    Notes
    -----
    Returns (empty, empty) if no finite features exist for this dimension.
    """
    pairs = _finite_pairs(diagram, dim)
    if len(pairs) == 0:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int32)

    eps_min = pairs[:, 0].min()
    eps_max = pairs[:, 1].max()
    epsilons = np.linspace(eps_min, eps_max, n_steps, dtype=np.float32)

    counts = np.array(
        [int(np.sum((pairs[:, 0] <= eps) & (pairs[:, 1] > eps))) for eps in epsilons],
        dtype=np.int32,
    )

    return epsilons, counts


def wasserstein_distance(diag_a: np.ndarray, diag_b: np.ndarray, dim: int = 1) -> float:
    """
    Wasserstein distance between two persistence diagrams in dimension dim.

    Measures the cost of the optimal transport matching between two
    persistence diagrams, treating points near the diagonal as
    interchangeable with noise. Smaller distance = more similar topology.

    Parameters
    ----------
    diag_a : PersistenceDiagram
    diag_b : PersistenceDiagram
    dim : int
        Homology dimension to compare. Default 1.

    Returns
    -------
    float
        Wasserstein distance. 0.0 if diagrams are identical.

    Notes
    -----
    Uses persim's Wasserstein implementation. If either diagram has no
    finite features in dim, returns 0.0 — both diagrams are trivially
    equivalent in that dimension.
    """
    dgm_a = diag_a.diagrams[dim]
    dgm_b = diag_b.diagrams[dim]

    # persim requires at least one finite feature in each diagram
    finite_a = dgm_a[np.isfinite(dgm_a[:, 1])] if len(dgm_a) > 0 else np.empty((0, 2))
    finite_b = dgm_b[np.isfinite(dgm_b[:, 1])] if len(dgm_b) > 0 else np.empty((0, 2))

    if len(finite_a) == 0 and len(finite_b) == 0:
        return 0.0

    return float(wasserstein(dgm_a, dgm_b))
