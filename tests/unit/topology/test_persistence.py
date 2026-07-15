import numpy as np
import pytest
from sdr_topology.topology.persistence import compute, PersistenceDiagram

# Synthetic circle — should produce one prominent H1 feature
N = 200
THETA = np.linspace(0, 2 * np.pi, N, endpoint=False)
CIRCLE = np.stack([np.cos(THETA), np.sin(THETA)], axis=1).astype(np.float32)

# Synthetic two-cluster point cloud — should produce two H0 features
CLUSTER_A = np.random.default_rng(42).normal(loc=[-5, 0], scale=0.2, size=(50, 2))
CLUSTER_B = np.random.default_rng(42).normal(loc=[5, 0], scale=0.2, size=(50, 2))
TWO_CLUSTERS = np.vstack([CLUSTER_A, CLUSTER_B]).astype(np.float32)


class TestCompute:
    def test_returns_persistence_diagram(self):
        result = compute(CIRCLE)
        assert isinstance(result, PersistenceDiagram)

    def test_h0_nonempty(self):
        result = compute(CIRCLE)
        assert len(result.h0) > 0

    def test_h1_nonempty_for_circle(self):
        result = compute(CIRCLE)
        assert len(result.h1) > 0

    def test_h2_empty_for_2d_cloud(self):
        # H2 not computed at maxdim=1
        result = compute(CIRCLE, maxdim=1)
        assert len(result.h2) == 0

    def test_output_dtypes(self):
        result = compute(CIRCLE)
        assert result.h0.dtype == np.float32
        assert result.h1.dtype == np.float32

    def test_diagrams_property_length(self):
        result = compute(CIRCLE)
        assert len(result.diagrams) == 3

    def test_circle_has_one_prominent_h1(self):
        result = compute(CIRCLE)
        finite = result.h1[np.isfinite(result.h1[:, 1])]
        lifetimes = finite[:, 1] - finite[:, 0]

        # A clean circle produces exactly one H1 feature
        # If multiple exist, the dominant one should have lifetime > 1.0
        # (circle of radius 1 — the loop persists across most of the filtration)
        assert len(lifetimes) >= 1
        assert lifetimes.max() > 1.0

    def test_two_clusters_h0(self):
        # Two well-separated clusters — two long-lived H0 features before merging
        result = compute(TWO_CLUSTERS)
        finite_h0 = result.h0[np.isfinite(result.h0[:, 1])]
        lifetimes = finite_h0[:, 1] - finite_h0[:, 0]
        # At least one long-lived H0 feature (the gap between clusters)
        assert lifetimes.max() > 5.0

    def test_rejects_1d_input(self):
        with pytest.raises(ValueError, match="at least 2D"):
            compute(np.ones(100))

    def test_rejects_negative_maxdim(self):
        with pytest.raises(ValueError):
            compute(CIRCLE, maxdim=-1)

    def test_rejects_nonsquare_distance_matrix(self):
        with pytest.raises(ValueError, match="square"):
            compute(np.ones((10, 5)), distance_matrix=True)

    def test_maxdim_stored_in_result(self):
        result = compute(CIRCLE, maxdim=1)
        assert result.maxdim == 1

    def test_metric_stored_in_result(self):
        result = compute(CIRCLE, metric="euclidean")
        assert result.metric == "euclidean"
