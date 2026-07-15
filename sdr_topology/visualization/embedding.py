from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # noqa: F401 — registers 3D projection
from ..logging import logger


def plot_point_cloud(
    point_cloud: np.ndarray,
    output_path: Path | None = None,
    title: str | None = None,
    show: bool = False,
    color_by_time: bool = True,
    subsample: int | None = None,
    connect: bool = False,
) -> plt.Figure:
    """
    Visualize a 2D or 3D point cloud from an embedding module.

    Parameters
    ----------
    point_cloud : np.ndarray
        Shape (N, 2) or (N, 3). Output of embedding.iq.embed() or
        embedding.delay.embed().
    output_path : Path, optional
        Save path. If None, figure is returned without saving.
    title : str, optional
        Figure title.
    show : bool
        If True, calls plt.show(). Default False.
    color_by_time : bool
        If True, colors points by sample index to show trajectory direction.
        Uses a sequential colormap (viridis). Default True.
    subsample : int, optional
        If provided, plots only every Nth point. Useful for large point clouds
        where rendering all points is slow. Does not affect the underlying data.
    connect : bool
        If True, draws lines connecting consecutive points to show trajectory
        path. Useful for short windows; cluttered for large point clouds.
        Default False.

    Returns
    -------
    plt.Figure

    Raises
    ------
    ValueError
        If point_cloud is not 2D or 3D (second axis must be 2 or 3).
    """
    if point_cloud.ndim != 2 or point_cloud.shape[1] not in (2, 3):
        error_msg = f"""
        point_cloud must have a shape (N, 2) or (N, 3).
        Got shape: {point_cloud.shape}
        """
        logger.error(error_msg)
        raise ValueError(error_msg)

    pc = point_cloud[::subsample] if subsample is not None else point_cloud
    n = len(pc)
    colors = np.arange(n) if color_by_time else "steelblue"
    cmap = "viridis" if color_by_time else None
    is_3d = point_cloud.shape[1] == 3

    fig = plt.figure(figsize=(6, 6))

    if is_3d:
        ax = fig.add_subplot(111, projection="3d")
        sc = ax.scatter(
            pc[:, 0], pc[:, 1], pc[:, 2],
            c=colors, cmap=cmap, s=2, alpha=0.6,
        )
        if connect:
            ax.plot(pc[:, 0], pc[:, 1], pc[:, 2],
                    color="gray", linewidth=0.3, alpha=0.4)

        ax.set_xlabel("dim 0")
        ax.set_ylabel("dim 1")
        ax.set_zlabel("dim 2")
    else:
        ax = fig.add_subplot(111)
        sc = ax.scatter(
            pc[:, 0], pc[:, 1],
            c=colors, cmap=cmap, s=2, alpha=0.6,
        )
        if connect:
            ax.plot(pc[:, 0], pc[:, 1],
                    color="gray", linewidth=0.3, alpha=0.4)
        ax.set_xlabel("dim 0")
        ax.set_ylabel("dim 1")
        ax.set_aspect("equal")

    if color_by_time:
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("sample index", fontsize=9)

    if title:
        ax.set_title(title, fontsize=11)

    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig
