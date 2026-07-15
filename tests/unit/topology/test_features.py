import numpy as np
import pytest
from sdr_topology.topology.persistence import compute, PersistenceDiagram
from sdr_topology.topology.features import (
    lifetimes,
    max_persistence,
    betti_curve,
    betti_numbers,
    wasserstein_distance,
)

# Synthetic circle — one prominent H1 feature
N = 200
THETA = np.linspace(0, 2 * np.pi, N, endpoint=False)
CIRCLE = np.stack([np.cos(THETA), np.sin(THETA)], axis=1).astype(np.float32)
CIRCLE_DIAG = compute(CIRCLE, maxdim=1)

# Two well-separated clusters — two long-lived H0 features
RNG = np.random.default_rng(42)
CLUSTER_A = RNG.normal(loc=[-5, 0], scale=0.2, size=(50, 2)).astype(np.float32)
CLUSTER_B = RNG.normal(loc=[5, 0], scale=0.2, size=(50, 2)).astype(np.float32)
TWO_CLUSTERS = np.vstack([CLUSTER_A, CLUSTER_B])
CLUSTER_DIAG = compute(TWO_CLUSTERS, maxdim=1)

# Noise — no structure
NOISE = RNG.uniform(-1, 1, size=(200, 2)).astype(np.float32)
NOISE_DIAG = compute(NOISE, maxdim=1)


class TestLifetimes:
    def test_returns_array(self):
        lt = lifetimes(CIRCLE_DIAG, dim=1)
        assert isinstance(lt, np.ndarray)

    def test_sorted_descending(self):
        lt = lifetimes(CIRCLE_DIAG, dim=1)
        assert np.all(lt[:-1] >= lt[1:])

    def test_circle_has_one_h1_lifetime(self):
        lt = lifetimes(CIRCLE_DIAG, dim=1)
        assert len(lt) == 1

    def test_nonnegative(self):
        lt = lifetimes(CIRCLE_DIAG, dim=0)
        assert np.all(lt >= 0)

    def test_invalid_dim(self):
        with pytest.raises(ValueError):
            lifetimes(CIRCLE_DIAG, dim=3)

    def test_empty_for_missing_dim(self):
        # H2 not computed at maxdim=1
        lt = lifetimes(CIRCLE_DIAG, dim=2)
        assert len(lt) == 0


class TestMaxPersistence:
    def test_circle_h1_positive(self):
        mp = max_persistence(CIRCLE_DIAG, dim=1)
        assert mp > 1.0

    def test_returns_float(self):
        assert isinstance(max_persistence(CIRCLE_DIAG, dim=1), float)

    def test_zero_for_empty_dim(self):
        mp = max_persistence(CIRCLE_DIAG, dim=2)
        assert mp == 0.0

    def test_noise_h1_less_than_circle(self):
        # Noise should have lower max H1 persistence than a clean circle
        assert max_persistence(NOISE_DIAG, dim=1) < max_persistence(CIRCLE_DIAG, dim=1)


class TestBettiCurve:
    def test_output_shapes(self):
        epsilons, counts = betti_curve(CIRCLE_DIAG, dim=1, n_steps=50)
        assert epsilons.shape == (50,)
        assert counts.shape == (50,)

    def test_epsilons_ascending(self):
        epsilons, _ = betti_curve(CIRCLE_DIAG, dim=1)
        assert np.all(epsilons[:-1] <= epsilons[1:])

    def test_counts_nonnegative(self):
        _, counts = betti_curve(CIRCLE_DIAG, dim=1)
        assert np.all(counts >= 0)

    def test_circle_has_loop_in_curve(self):
        # Circle should show beta_1 = 1 somewhere in the curve
        _, counts = betti_curve(CIRCLE_DIAG, dim=1)
        assert counts.max() >= 1

    def test_empty_for_missing_dim(self):
        epsilons, counts = betti_curve(CIRCLE_DIAG, dim=2)
        assert len(epsilons) == 0
        assert len(counts) == 0


class TestBettiNumbers:
    def test_returns_int(self):
        assert isinstance(betti_numbers(CIRCLE_DIAG, dim=1), int)

    def test_zero_for_empty_dim(self):
        assert betti_numbers(CIRCLE_DIAG, dim=2) == 0

    def test_default_epsilon_gives_valid_count(self):
        # Default epsilon (median death) should find the circle's loop
        bn = betti_numbers(CIRCLE_DIAG, dim=1)
        assert bn >= 1

    def test_explicit_epsilon_zero_gives_zero(self):
        # At epsilon=0, nothing has been born yet
        assert betti_numbers(CIRCLE_DIAG, dim=1, epsilon=0.0) == 0

    def test_two_clusters_h0(self):
        # At mid-filtration, two clusters should show beta_0 >= 1
        bn = betti_numbers(CLUSTER_DIAG, dim=0)
        assert bn >= 1


class TestWassersteinDistance:
    def test_identical_diagrams_zero(self):
        dist = wasserstein_distance(CIRCLE_DIAG, CIRCLE_DIAG, dim=1)
        assert dist == pytest.approx(0.0, abs=1e-4)

    def test_returns_float(self):
        assert isinstance(wasserstein_distance(CIRCLE_DIAG, NOISE_DIAG, dim=1), float)

    def test_circle_vs_noise_positive(self):
        dist = wasserstein_distance(CIRCLE_DIAG, NOISE_DIAG, dim=1)
        assert dist > 0.0

    def test_symmetry(self):
        d_ab = wasserstein_distance(CIRCLE_DIAG, NOISE_DIAG, dim=1)
        d_ba = wasserstein_distance(NOISE_DIAG, CIRCLE_DIAG, dim=1)
        assert d_ab == pytest.approx(d_ba, rel=1e-4)

    def test_both_empty_returns_zero(self):
        # H2 is empty for both — should return 0.0
        dist = wasserstein_distance(CIRCLE_DIAG, NOISE_DIAG, dim=2)
        assert dist == 0.0