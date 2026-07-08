from .utils import _check_sample_ndim
from ..logging import logger
import numpy as np


def embed(
    samples: np.ndarray,
    dim: int,
    tau: int,
) -> np.ndarray:
    """
    Time-delay embedding of a real-valued scalar time series.

    Constructs a point cloud in R^dim by mapping each time step to a
    vector of dim lagged values separated by tau samples.

    Parameters
    ----------
    samples : np.ndarray
        Real-valued 1D array (float32 or float64).
    dim : int
        Embedding dimension.
    tau : int
        Time delay (lag) in samples.

    Returns
    -------
    np.ndarray
        Shape (N - (dim - 1) * tau, dim). Each row is one point in
        the embedded space.

    Notes
    -----
    Takens' theorem guarantees faithful attractor reconstruction for
    d >= 2m + 1 where m is the system's intrinsic dimensionality.
    Use optimal_dim() to select d empirically via false nearest neighbors.
    """
    _check_sample_ndim(samples=samples)
    if dim < 1:
        logger.error(f"Dimension (dim) must be >= 1, given value was {dim}.")
        raise ValueError(f"Dimension (dim) must be >= 1, given value was {dim}.")
    if tau < 1:
        logger.error(f"tau must be >= 1, given value was {tau}.")
        raise ValueError(f"tau must be >= 1, given value was {tau}.")

    n_points = len(samples) - (dim - 1) * tau
    if n_points <= 0:
        error_msg = f"""
        Not enough samples ({len(samples)}) for dim={dim} and tau={tau}.
        At least {(dim - 1) * tau + 1} are needed.
        """
        logger.error(error_msg)
        raise ValueError(error_msg)

    indices = np.arange(dim) * tau
    return np.stack([samples[i : i + n_points] for i in indices], axis=1)


def ami_curve(samples: np.ndarray, max_tau: int, n_bins: int = 64) -> np.ndarray:
    """
    Average mutual information between a time series and its lagged copies.

    Computes AMI for lags 1..max_tau using histogram-based entropy estimation.
    The first minimum of the AMI curve is the standard criterion for selecting
    the time delay tau for embedding (Fraser & Swinney, 1986).

    Parameters
    ----------
    samples : np.ndarray
        Real-valued 1D array.
    max_tau : int
        Maximum lag to evaluate.
    n_bins : int
        Number of histogram bins for joint/marginal entropy estimation.

    Returns
    -------
    np.ndarray
        AMI values for lags 1..max_tau, shape (max_tau,).
    """
    _check_sample_ndim(samples=samples)
    s_min, s_max = samples.min(), samples.max()
    if s_max == s_min:
        error_msg = "Sample values are constant, cannot compute AMI."
        logger.error(error_msg)
        raise ValueError(error_msg)

    s = (samples - s_min) / (s_max - s_min)

    ami = np.zeros(max_tau, dtype=np.float64)

    for lag in range(1, max_tau + 1):
        x = s[:-lag]
        y = s[lag:]

        # Joint 2D histogram → joint probability
        joint, _, _ = np.histogram2d(x, y, bins=n_bins)
        joint /= joint.sum()

        # Marginal probabilities
        px = joint.sum(axis=1)
        py = joint.sum(axis=0)

        # AMI = H(X) + H(Y) - H(X,Y)
        # Computed directly from joint distribution to avoid
        # separate marginal histogram estimation
        def _entropy(p: np.ndarray) -> float:
            p = p[p > 0]
            return -np.sum(p * np.log2(p))

        ami[lag - 1] = _entropy(px) + _entropy(py) - _entropy(joint)

    return ami


def fnn_curve(
    samples: np.ndarray, tau: int, max_dim: int, rtol: float = 15.0
) -> np.ndarray:
    """
    False nearest neighbors fraction across embedding dimensions.

    For each dimension d in 1..max_dim, computes the fraction of points
    whose nearest neighbor in R^d is 'false' — i.e., becomes much more
    distant when the dimension is increased to d+1. A rapid drop toward
    zero indicates the attractor is unfolded at that dimension.

    Parameters
    ----------
    samples : np.ndarray
        Real-valued 1D array.
    tau : int
        Time delay to use for embedding (from optimal_tau).
    max_dim : int
        Maximum embedding dimension to evaluate.
    rtol : float
        Relative distance threshold for declaring a false neighbor.
        Standard value is 15.0 (Kennel et al., 1992).

    Returns
    -------
    np.ndarray
        FNN fraction for dims 1..max_dim, shape (max_dim,).
    """
    from scipy.spatial import KDTree

    fnn = np.zeros(max_dim, dtype=np.float64)

    for d in range(1, max_dim + 1):
        # Embed at current and next dimension
        cloud_d = embed(samples, d, tau)
        cloud_d1 = embed(samples, d + 1, tau)

        # Align lengths — cloud_d1 is shorter by tau
        n = min(len(cloud_d), len(cloud_d1))
        cloud_d = cloud_d[:n]
        cloud_d1 = cloud_d1[:n]

        tree = KDTree(cloud_d)
        dists, indices = tree.query(
            cloud_d, k=2
        )  # k=2: point itself + nearest neighbor

        nn_dist_d = dists[:, 1]  # distance to nearest neighbor in R^d
        nn_idx = indices[:, 1]  # index of nearest neighbor

        # Distance to same neighbor in R^(d+1)
        nn_dist_d1 = np.linalg.norm(cloud_d1 - cloud_d1[nn_idx], axis=1)

        # False neighbor: relative distance increase exceeds rtol
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(
                nn_dist_d > 0,
                np.sqrt(np.maximum((nn_dist_d1**2 - nn_dist_d**2), 0)) / nn_dist_d,
                0.0,
            )

        fnn[d - 1] = np.mean(ratio > rtol)

    return fnn


def optimal_tau(samples: np.ndarray, max_tau: int, n_bins: int = 64, prominence: float = 0.01) -> int:
    """
    Select embedding delay tau as the first local minimum of the AMI curve.

    Parameters
    ----------
    samples : np.ndarray
        Real-valued 1D array.
    max_tau : int
        Maximum lag to evaluate.
    n_bins : int
        Number of histogram bins passed to ami_curve().
    prominent : float
        .

    Returns
    -------
    int
        Optimal tau (1-indexed lag value, not array index).

    Notes
    -----
    If no local minimum is found (AMI decreases monotonically), returns
    max_tau as a fallback and should be treated as a signal that max_tau
    is too small or the signal has no clear decorrelation scale.
    """
    curve = ami_curve(samples, max_tau, n_bins)

    for i in range(1, len(curve) - 1):
        if curve[i] < curve[i - 1] and curve[i] < curve[i + 1]:
            depth = min(curve[i - 1] - curve[i], curve[i + 1] - curve[i])
            if depth > prominence * curve[i]:
                return i + 1  # convert 0-indexed to 1-indexed lag

    # Monotonic decrease — return argmin as fallback
    return int(np.argmin(curve)) + 1


def optimal_dim(
    samples: np.ndarray,
    tau: int,
    max_dim: int,
    rtol: float = 15.0,
    threshold: float = 0.01,
) -> int:
    """
    Select embedding dimension as first d where FNN fraction drops below threshold.

    Parameters
    ----------
    samples : np.ndarray
        Real-valued 1D array.
    tau : int
        Time delay (from optimal_tau).
    max_dim : int
        Maximum dimension to evaluate.
    rtol : float
        Relative distance threshold for FNN (passed to fnn_curve).
    threshold : float
        FNN fraction below which the attractor is considered unfolded.
        Default 0.01 (1%).

    Returns
    -------
    int
        Optimal embedding dimension.

    Notes
    -----
    If FNN never drops below threshold, returns max_dim as fallback —
    treat as a signal that max_dim is too small or the signal is noisy.
    """
    curve = fnn_curve(samples, tau, max_dim, rtol)

    for i, fnn_val in enumerate(curve):
        if fnn_val < threshold:
            return i + 1  # 1-indexed

    return max_dim
