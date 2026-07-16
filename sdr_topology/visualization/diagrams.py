from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from persim import plot_diagrams

from ..logging import logger
from ..topology.persistence import PersistenceDiagram
from ..topology.features import lifetimes as compute_lifetimes


def plot(
    diagram: PersistenceDiagram,
    output_path: Path | None = None,
    title: str | None = None,
    show: bool = False,
    dims: list[int] | None = None,
    lifetime_cutoff: float | None = None,
) -> plt.Figure:
    """
    Plot persistence diagram for one or more homology dimensions.

    Parameters
    ----------
    diagram : PersistenceDiagram
    output_path : Path, optional
        If provided, saves figure to this path. Suffix determines format
        (.png, .pdf, .svg). If None, figure is returned without saving.
    title : str, optional
        Figure title. If None, no title is added.
    show : bool
        If True, calls plt.show() after plotting. Default False.
        Use only in interactive sessions — not in pipeline runs.
    dims : list[int], optional
        Which homology dimensions to include. Default [0, 1].
        Pass [1] to show only H1, which is often cleaner for RF signals.
    lifetime_cutoff : float, optional
        If provided, filters out features with lifetime below this threshold
        before plotting. Useful for suppressing noise features near the diagonal
        without modifying the underlying diagram.

    Returns
    -------
    plt.Figure
        The matplotlib figure. Caller is responsible for closing if not needed.

    Notes
    -----
    Uses persim's plot_diagrams internally for consistent birth/death axis
    formatting and diagonal reference line. Output is saved at 150 DPI by
    default — sufficient for writeup inclusion.
    """
    if dims is None:
        dims = [0, 1]

    # Build diagram list for persim — must be ordered by dimension
    dgms_to_plot = []
    labels = []
    for d in sorted(dims):
        if d not in (0, 1, 2):
            error_msg = f"dim must be 0, 1, or 2, got {d}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        dgm = diagram.diagrams[d].copy()

        if lifetime_cutoff is not None and len(dgm) > 0:
            finite_mask = np.isfinite(dgm[:, 1])
            lifetimes = np.where(finite_mask, dgm[:, 1] - dgm[:, 0], np.inf)
            dgm = dgm[lifetimes >= lifetime_cutoff]

        dgms_to_plot.append(dgm)
        labels.append(f"$H_{d}$")

    fig, ax = plt.subplots(figsize=(6, 6))
    plot_diagrams(dgms_to_plot, labels=labels, ax=ax)

    if title:
        ax.set_title(title, fontsize=11)

    ax.set_xlabel("Birth")
    ax.set_ylabel("Death")

    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_lifetime_distribution(
    diagram: PersistenceDiagram,
    dim: int = 1,
    output_path: Path | None = None,
    title: str | None = None,
    show: bool = False,
    n_bins: int = 30,
) -> plt.Figure:
    """
    Histogram of feature lifetimes for a given homology dimension.

    Complements the persistence diagram by showing the distribution of
    lifetimes directly — useful for characterizing whether a signal has
    one dominant feature (spike at high lifetime) or many moderate features
    (spread distribution), as seen in the FM null result.

    Parameters
    ----------
    diagram : PersistenceDiagram
    dim : int
        Homology dimension. Default 1.
    output_path : Path, optional
        Save path. If None, figure is returned without saving.
    title : str, optional
        Figure title.
    show : bool
        If True, calls plt.show(). Default False.
    n_bins : int
        Number of histogram bins. Default 30.

    Returns
    -------
    plt.Figure
    """
    lt = compute_lifetimes(diagram=diagram, dim=dim)

    fig, ax = plt.subplots(figsize=(6, 4))

    if len(lt) == 0:
        ax.text(
            0.5,
            0.5,
            f"No finite H{dim} features",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    else:
        ax.hist(lt, bins=n_bins, color="steelblue", edgecolor="white", linewidth=0.5)
        ax.axvline(
            lt.max(),
            color="crimson",
            linestyle="--",
            linewidth=1.0,
            label=f"max = {lt.max():.3f}",
        )
        ax.legend(fontsize=9)

    ax.set_xlabel("Lifetime (death - birth)")
    ax.set_ylabel("Count")

    if title:
        ax.title(title, fontsize=11)

    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig
