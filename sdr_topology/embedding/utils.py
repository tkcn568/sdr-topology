from __future__ import annotations
from ..logging import logger
import numpy as np


def _check_sample_ndim(samples: np.ndarray) -> None:
    if samples.ndim != 1:
        logger.error(f"Invalid sample size. Expected: nx1 or 1xn, Got: {samples.shape}")
        raise ValueError(
            f"Invalid sample size. Expected: nx1 or 1xn, Got: {samples.shape}"
        )


def get_magnitude(samples: np.ndarray) -> np.ndarray:
    """
    Extract magnitude from complex IQ samples.

    Parameters
    ----------
    samples : np.ndarray
        Complex64 IQ samples as returned by capture modules.

    Returns
    -------
    np.ndarray
        Float32 magnitude array, same length as input.
    """
    return np.abs(samples).astype(np.float32)
