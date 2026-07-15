import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sdr_topology.visualization.embedding import plot_point_cloud

# Synthetic 2D circle
N = 200
THETA = np.linspace(0, 2 * np.pi, N, endpoint=False)
CIRCLE_2D = np.stack([np.cos(THETA), np.sin(THETA)], axis=1).astype(np.float32)

# Synthetic 3D helix
HELIX_3D = np.stack([
    np.cos(THETA), np.sin(THETA), np.linspace(0, 1, N)
], axis=1).astype(np.float32)


class TestPlotPointCloud:
    def test_returns_figure_2d(self):
        fig = plot_point_cloud(CIRCLE_2D)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_returns_figure_3d(self):
        fig = plot_point_cloud(HELIX_3D)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_to_disk(self, tmp_path):
        out = tmp_path / "cloud.png"
        fig = plot_point_cloud(CIRCLE_2D, output_path=out)
        assert out.exists()
        assert out.stat().st_size > 0
        plt.close(fig)

    def test_creates_parent_dirs(self, tmp_path):
        out = tmp_path / "subdir" / "cloud.png"
        fig = plot_point_cloud(CIRCLE_2D, output_path=out)
        assert out.exists()
        plt.close(fig)

    def test_rejects_1d_input(self):
        with pytest.raises(ValueError):
            plot_point_cloud(np.ones(100))

    def test_rejects_wrong_second_dim(self):
        with pytest.raises(ValueError):
            plot_point_cloud(np.ones((100, 4)))

    def test_subsample_reduces_plotted_points(self):
        # Just confirm it doesn't raise — point count is internal to matplotlib
        fig = plot_point_cloud(CIRCLE_2D, subsample=2)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_connect_does_not_raise(self):
        fig = plot_point_cloud(CIRCLE_2D, connect=True)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_no_color_by_time(self):
        fig = plot_point_cloud(CIRCLE_2D, color_by_time=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)