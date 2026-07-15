import numpy as np
import pytest
from sdr_topology.embedding.iq import embed

# Synthetic: pure tone in IQ — constant amplitude, rotating phase
# At baseband with no frequency offset this would be a perfect circle
N = 1000
THETA = np.linspace(0, 4 * np.pi, N)  # two full rotations
IQ = (np.cos(THETA) + 1j * np.sin(THETA)).astype(np.complex64)


class TestIQEmbed:
    def test_output_shape_default(self):
        cloud = embed(IQ)
        assert cloud.shape == (N, 2)

    def test_output_dtype(self):
        cloud = embed(IQ)
        assert cloud.dtype == np.float32

    def test_columns_are_i_and_q(self):
        cloud = embed(IQ)
        np.testing.assert_allclose(cloud[:, 0], IQ.real, rtol=1e-5)
        np.testing.assert_allclose(cloud[:, 1], IQ.imag, rtol=1e-5)

    def test_n_points_limits_output(self):
        cloud = embed(IQ, n_points=100)
        assert cloud.shape == (100, 2)

    def test_start_offsets_correctly(self):
        cloud_full = embed(IQ)
        cloud_offset = embed(IQ, start=10)
        np.testing.assert_array_equal(cloud_full[10:], cloud_offset)

    def test_stride_subsamples(self):
        cloud = embed(IQ, stride=2)
        assert cloud.shape == (N // 2, 2)

    def test_rejects_real_input(self):
        with pytest.raises(ValueError, match="complex"):
            embed(np.ones(100, dtype=np.float32))

    def test_rejects_zero_stride(self):
        with pytest.raises(ValueError):
            embed(IQ, stride=0)

    def test_rejects_invalid_start(self):
        with pytest.raises(ValueError):
            embed(IQ, start=N + 1)

    def test_rejects_zero_n_points(self):
        with pytest.raises(ValueError):
            embed(IQ, n_points=0)

    def test_circle_geometry(self):
        # Synthetic perfect circle — all points should be at unit radius
        cloud = embed(IQ)
        radii = np.sqrt(cloud[:, 0] ** 2 + cloud[:, 1] ** 2)
        np.testing.assert_allclose(radii, 1.0, atol=1e-5)
