import numpy as np
import pytest

from sdr_topology.embedding.utils import estimate_carrier_offset, correct_carrier_offset

# Synthetic: pure carrier offset, no modulation
N = 5000
OFFSET = 0.05  # radians/sample
N_ARR = np.arange(N)
PURE_OFFSET_SIGNAL = np.exp(1j * OFFSET * N_ARR).astype(np.complex64)

# Synthetic: offset + FM-like modulation (small phase wobble on top of rotation)
RNG = np.random.default_rng(42)
MODULATION = 0.01 * np.sin(2 * np.pi * N_ARR / 200)  # slow phase modulation
OFFSET_PLUS_MOD = np.exp(1j * (OFFSET * N_ARR + MODULATION)).astype(np.complex64)


class TestEstimateCarrierOffset:
    def test_returns_float(self):
        result = estimate_carrier_offset(PURE_OFFSET_SIGNAL)
        assert isinstance(result, float)

    def test_recovers_known_offset(self):
        estimated = estimate_carrier_offset(PURE_OFFSET_SIGNAL)
        assert estimated == pytest.approx(OFFSET, abs=1e-4)

    def test_recovers_offset_with_modulation(self):
        # Modulation should average out, leaving the offset estimate close
        estimated = estimate_carrier_offset(OFFSET_PLUS_MOD)
        assert estimated == pytest.approx(OFFSET, abs=1e-3)

    def test_rejects_real_input(self):
        with pytest.raises(ValueError, match="complex"):
            estimate_carrier_offset(np.ones(100, dtype=np.float32))

    def test_rejects_too_short(self):
        with pytest.raises(ValueError):
            estimate_carrier_offset(np.array([1 + 1j], dtype=np.complex64))

    def test_zero_offset_for_constant_signal(self):
        constant = np.ones(100, dtype=np.complex64)
        estimated = estimate_carrier_offset(constant)
        assert estimated == pytest.approx(0.0, abs=1e-6)


class TestCorrectCarrierOffset:
    def test_output_shape_preserved(self):
        corrected = correct_carrier_offset(
            PURE_OFFSET_SIGNAL, offset_rad_per_sample=OFFSET
        )
        assert corrected.shape == PURE_OFFSET_SIGNAL.shape

    def test_output_dtype(self):
        corrected = correct_carrier_offset(
            PURE_OFFSET_SIGNAL, offset_rad_per_sample=OFFSET
        )
        assert corrected.dtype == np.complex64

    def test_corrects_pure_offset_to_near_constant(self):
        # Correcting a pure rotation at the known offset should yield
        # a signal with near-zero residual rotation
        corrected = correct_carrier_offset(
            PURE_OFFSET_SIGNAL, offset_rad_per_sample=OFFSET
        )
        residual_offset = estimate_carrier_offset(corrected)
        assert residual_offset == pytest.approx(0.0, abs=1e-4)

    def test_self_estimated_correction(self):
        # offset_rad_per_sample=None should self-estimate and correct
        corrected = correct_carrier_offset(PURE_OFFSET_SIGNAL)
        residual_offset = estimate_carrier_offset(corrected)
        assert residual_offset == pytest.approx(0.0, abs=1e-4)

    def test_preserves_modulation_after_offset_removal(self):
        # After removing the known offset, the remaining phase should
        # reflect the modulation structure, not be flat
        corrected = correct_carrier_offset(
            OFFSET_PLUS_MOD, offset_rad_per_sample=OFFSET
        )
        phase = np.unwrap(np.angle(corrected))
        # Modulation has real variance — corrected phase should not be ~constant
        assert np.std(phase) > 1e-3
