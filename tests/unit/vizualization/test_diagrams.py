# tests/unit/test_diagrams.py

import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for tests
import matplotlib.pyplot as plt
from sdr_topology.topology.persistence import compute
from sdr_topology.visualization.diagrams import plot, plot_lifetime_distribution

# Synthetic circle
N = 200
THETA = np.linspace(0, 2 * np.pi, N, endpoint=False)
CIRCLE = np.stack([np.cos(THETA), np.sin(THETA)], axis=1).astype(np.float32)
CIRCLE_DIAG = compute(CIRCLE, maxdim=1)


class TestPlot:
    def test_returns_figure(self):
        fig = plot(CIRCLE_DIAG)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_to_disk(self, tmp_path):
        out = tmp_path / "test_diagram.png"
        fig = plot(CIRCLE_DIAG, output_path=out)
        assert out.exists()
        assert out.stat().st_size > 0
        plt.close(fig)

    def test_creates_parent_dirs(self, tmp_path):
        out = tmp_path / "subdir" / "nested" / "diagram.png"
        fig = plot(CIRCLE_DIAG, output_path=out)
        assert out.exists()
        plt.close(fig)

    def test_invalid_dim_raises(self):
        with pytest.raises(ValueError):
            plot(CIRCLE_DIAG, dims=[3])

    def test_single_dim(self):
        fig = plot(CIRCLE_DIAG, dims=[1])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_lifetime_cutoff_accepted(self):
        fig = plot(CIRCLE_DIAG, lifetime_cutoff=0.1)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotLifetimeDistribution:
    def test_returns_figure(self):
        fig = plot_lifetime_distribution(CIRCLE_DIAG, dim=1)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_to_disk(self, tmp_path):
        out = tmp_path / "lifetimes.png"
        fig = plot_lifetime_distribution(CIRCLE_DIAG, dim=1, output_path=out)
        assert out.exists()
        assert out.stat().st_size > 0
        plt.close(fig)

    def test_empty_dim_does_not_raise(self):
        # H2 is empty — should render a "no features" message, not crash
        fig = plot_lifetime_distribution(CIRCLE_DIAG, dim=2)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)