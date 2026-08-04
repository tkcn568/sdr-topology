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


def estimate_carrier_offset(samples: np.ndarray) -> float:
    """
    Compute average instantaneous frequency by computing phase difference between
    consecutive samples ($$\\Delta\\theta_{n}=\\arg\\left(x_{n+1}\\bar{x}_{n}\\right)$$)
    and averaging it across the capture.

    This estimates the mean rotation rate (radians/sample) caused by imperfect tuner
    frequency lock, distinct from the modulation-driven phase variation which should
    average close to zero over a sufficiently long capture.

    Parameters
    ----------
    samples : np.ndarray
        Complex64 IQ samples as returned by capture modules.

    Returns
    -------
    float
        Mean phase rotation per sample, in radians. Positive values
        indicate the tuner is offset below the true carrier frequency
        (signal appears to rotate forward); negative indicates offset above.

    Notes
    -----
    This estimate conflates carrier offset with any DC bias in the FM
    modulation (i.e., if the transmitted audio has sustained asymmetry).
    For broadcast FM with typical program content, this bias is expected
    to be small relative to the offset error from tuner imprecision, but
    this is an assumption worth stating explicitly rather than treating
    the estimate as exact.
    """
    if not np.iscomplexobj(samples):
        error_msg = f"samples must be complex, got dtype {samples.dtype}."
        logger.error(error_msg)
        raise ValueError(error_msg)
    if len(samples) < 2:
        error_msg = (
            "samples must be a length of at least 2 to estimate " + "phase rotation"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
    phase_differences = np.angle(samples[1:] * np.conj(samples[:-1]))
    return float(np.mean(phase_differences))


def correct_carrier_offset(
    samples: np.ndarray, offset_rad_per_sample: float | None = None
) -> np.ndarray:
    """
    Correct residual carrier frequency offset by counter-rotating the signal.

    Multiplies the signal by a complex exponential at the negative of the
    estimated (or provided) offset rate, removing the constant-rate spin
    caused by imperfect tuner frequency lock.

    Parameters
    ----------
    samples : np.ndarray
        Complex64 IQ samples.
    offset_rad_per_sample : float, optional
        Offset to correct, in radians/sample. If None, estimated via
        estimate_carrier_offset() on the input samples themselves.

    Returns
    -------
    np.ndarray
        Complex64 corrected samples, same length as input.

    Notes
    -----
    If offset_rad_per_sample is None, the correction is self-referential —
    estimated and applied to the same capture. This is appropriate for
    characterizing "what if this specific capture's offset were removed"
    but means the correction is specific to this capture's conditions
    (tuner temperature, exact frequency drift at capture time) and should
    not be assumed to transfer to other captures without re-estimation.
    """
    if offset_rad_per_sample is None:
        offset_rad_per_sample = estimate_carrier_offset(samples)

    n = np.arange(len(samples))
    correction = np.exp(-1j * offset_rad_per_sample * n)
    return (samples * correction).astype(np.complex64)
