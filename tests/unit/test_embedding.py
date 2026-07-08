import numpy as np
import pytest
from sdr_topology.embedding.delay import (
    embed,
    ami_curve,
    optimal_tau,
    fnn_curve,
    optimal_dim,
)
from sdr_topology.embedding.utils import get_magnitude


# Synthetic sine wave: 10 Hz, 1000 Hz sample rate, 2 seconds
# Quarter period = 1000 / (4 * 10) = 25 samples — expected optimal tau
FS = 1000
F0 = 10
DURATION = 2.0
T = np.linspace(0, DURATION, int(FS * DURATION), endpoint=False)
SINE = np.sin(2 * np.pi * F0 * T).astype(np.float32)
EXPECTED_TAU = FS // (4 * F0) # should be 25


class TestGetMagnitude:
    def test_real_output(self):
        samples = np.array([3 + 4j, 1 + 0j], dtype=np.complex64)
        mag = get_magnitude(samples=samples)
        np.testing.assert_allclose(mag, [5.0, 1.0], rtol=1e-5)

    def test_output_dtype(self):
        samples = np.array([1 + 1j], dtype=np.complex64)
        assert get_magnitude(samples=samples).dtype == np.float32

    def test_output_length(self):
        samples = np.ones(100, dtype=np.complex64)
        assert len(get_magnitude(samples=samples)) == 100


class TestEmbed:
    def test_output_shape(self):
        cloud = embed(SINE, dim=2, tau=25)
        expected_rows = len(SINE) - (2 - 1) * 25
        assert cloud.shape == (expected_rows, 2)

    def test_dim_1_is_column_vector(self):
        cloud = embed(SINE, dim=1, tau=1)
        assert cloud.shape == (len(SINE), 1)

    def test_rejects_2d_input(self):
        with pytest.raises(ValueError):
            embed(np.ones((10, 2)), dim=2, tau=1)

    def test_rejects_insufficient_samples(self):
        with pytest.raises(ValueError):
            embed(SINE[:10], dim=5, tau=10)

    def test_rejects_zero_dim(self):
        with pytest.raises(ValueError):
            embed(SINE, dim=0, tau=1)

    def test_rejects_zero_tau(self):
        with pytest.raises(ValueError):
            embed(SINE, dim=2, tau=0)

    def test_lag_structure(self):
        # Each column should be the signal shifted by tau
        tau = 5
        cloud = embed(SINE, dim=2, tau=tau)
        np.testing.assert_array_equal(cloud[:, 0], SINE[:len(cloud)])
        np.testing.assert_array_equal(cloud[:, 1], SINE[tau:tau + len(cloud)])


class TestAmiCurve:
    def test_output_length(self):
        curve = ami_curve(SINE, max_tau=50)
        assert len(curve) == 50

    def test_nonnegative(self):
        curve = ami_curve(SINE, max_tau=50)
        assert np.all(curve >= 0)

    def test_first_minimum_near_quarter_period(self):
        # AMI first minimum for a sine wave should be near quarter period
        curve = ami_curve(SINE, max_tau=50)
        first_min = np.argmin(curve) + 1  # 1-indexed
        assert 15 <= first_min <= 35  # within 5 samples


class TestOptimalTau:
    def test_returns_int(self):
        assert isinstance(optimal_tau(SINE, max_tau=50), int)

    def test_near_quarter_period(self):
        tau = optimal_tau(SINE, max_tau=50)
        assert abs(tau - EXPECTED_TAU) <= 25

    def test_within_range(self):
        tau = optimal_tau(SINE, max_tau=50)
        assert 1 <= tau <= 50


class TestFnnCurve:
    def test_output_length(self):
        curve = fnn_curve(SINE, tau=EXPECTED_TAU, max_dim=5)
        assert len(curve) == 5

    def test_drops_at_dim_2(self):
        # Sine wave is 2D — FNN should drop sharply at d=2
        curve = fnn_curve(SINE, tau=EXPECTED_TAU, max_dim=5)
        assert curve[1] < curve[0]  # FNN at d=2 < FNN at d=1


class TestOptimalDim:
    def test_returns_int(self):
        assert isinstance(optimal_dim(SINE, tau=EXPECTED_TAU, max_dim=5), int)

    def test_sine_is_2d(self):
        dim = optimal_dim(SINE, tau=EXPECTED_TAU, max_dim=5)
        assert dim == 2

    def test_within_range(self):
        dim = optimal_dim(SINE, tau=EXPECTED_TAU, max_dim=5)
        assert 1 <= dim <= 5