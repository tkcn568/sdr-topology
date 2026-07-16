from __future__ import annotations
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel


console = Console()

DEFAULT_LIBRARY = Path("profiles/data")


@click.group()
def main():
    """SDR Topology Analysis Toolkit — topological analysis of RF signals."""
    pass


# ---------------------------------------------------------------------------
# capture
# ---------------------------------------------------------------------------


@main.command()
@click.option(
    "--freq",
    "-f",
    required=True,
    type=int,
    help="Center frequency in Hz (e.g. 99500000)",
)
@click.option(
    "--rate",
    "-r",
    default=250_000,
    show_default=True,
    type=int,
    help="Sample rate in Hz",
)
@click.option(
    "--samples",
    "-n",
    default=250_000,
    show_default=True,
    type=int,
    help="Number of samples to capture",
)
@click.option(
    "--gain",
    "-g",
    default="auto",
    show_default=True,
    help="Tuner gain in dB, or 'auto'",
)
@click.option(
    "--output",
    "-o",
    required=True,
    type=click.Path(),
    help="Output path (without extension — .npy and .json written)",
)
@click.option("--notes", default="", help="Environment notes for metadata")
def capture(freq, rate, samples, gain, output, notes):
    """Capture IQ samples from the RTL-SDR and save to disk."""
    from .capture.rtlsdr import capture as do_capture

    output_path = Path(output)
    console.print(
        f"[bold]Capturing[/bold] {samples:,} sample at {freq / 1e6:.3f} MHz..."
    )

    try:
        _, metadata = do_capture(
            center_freq_hz=freq,
            sample_rate_hz=rate,
            n_samples=samples,
            output_path=output_path,
            gain=gain if gain == "auto" else float(gain),
            environment_notes=notes,
        )
    except Exception as e:
        console.print(f"[red]Capture failed:[/red] {e}")
        raise click.Abort()

    console.print(
        Panel(
            f"[green]✓[/green] Saved to [bold]{output_path.with_suffix('.npy')}[/bold]\n"
            f"Frequency:   {metadata.center_freq_hz / 1e6:.3f} MHz\n"
            f"Sample rate: {metadata.sample_rate_hz:,} S/s\n"
            f"Samples:     {metadata.n_samples:,}\n"
            f"Timestamp:   {metadata.timestamp_utc}\n"
            f"Tuner:       {metadata.tuner}\n",
            title="Capture complete",
        )
    )


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------


@main.command()
@click.option(
    "--capture-path",
    "-c",
    required=True,
    type=click.Path(exists=True),
    help="Path to .npy capture file",
)
@click.option(
    "--method",
    "-m",
    default="iq",
    show_default=True,
    type=click.Choice(["iq", "delay"]),
    help="Embedding method",
)
@click.option(
    "--label", "-l", required=True, help="Signal type label for the profile entry"
)
@click.option(
    "--n-points",
    default=2000,
    show_default=True,
    type=int,
    help="Number of points to embed",
)
@click.option(
    "--dim",
    default=None,
    type=int,
    help="Embedding dimension (delay only; auto-selected if omitted)",
)
@click.option(
    "--tau",
    default=None,
    type=int,
    help="Time delay in samples (delay only; auto-selected if omitted)",
)
@click.option(
    "--maxdim",
    default=1,
    show_default=True,
    type=int,
    help="Maximum homology dimension for persistence",
)
@click.option(
    "--library",
    default=str(DEFAULT_LIBRARY),
    show_default=True,
    type=click.Path(),
    help="Profile library directory",
)
@click.option("--notes", default="", help="Notes for this profile entry")
@click.option(
    "--no-save",
    is_flag=True,
    default=False,
    help="Compute but do not save to profile library",
)
def analyze(
    capture_path, method, label, n_points, dim, tau, maxdim, library, notes, no_save
):
    """Embed a capture and compute persistent homology. Saves a profile entry."""
    from sdr_topology.pipeline import run_iq, run_delay
    from sdr_topology.topology.features import max_persistence, betti_numbers

    library_dir = Path(library)

    console.print(
        f"""[bold]Analyzing[/bold] {capture_path}
        via [cyan]{method}[/cyan] embedding..."""
    )

    try:
        if method == "iq":
            entry = run_iq(
                label=label,
                capture_path=capture_path,
                library_dir=library_dir,
                n_points=n_points,
                maxdim=maxdim,
                notes=notes,
                save_entry=not no_save,
            )
        else:
            entry = run_delay(
                label=label,
                capture_path=capture_path,
                library_dir=library_dir,
                n_points=n_points,
                dim=dim,
                tau=tau,
                maxdim=maxdim,
                notes=notes,
                save_entry=not no_save,
            )
    except Exception as e:
        console.print(f"[red]Analysis failed:[/red] {e}")
        raise click.Abort()

    mp_h1 = max_persistence(entry.diagram, dim=1)
    bn_h1 = betti_numbers(entry.diagram, dim=1)

    console.print(
        Panel(
            f"[green]✓[/green] Entry key: [bold]{entry.key}[/bold]\n"
            f"Label:              {entry.label}\n"
            f"Embedding:          {entry.embedding_params.method}"
            f" (dim={entry.embedding_params.dim}, "
            f"tau={entry.embedding_params.tau})\n"
            f"Points:         {entry.embedding_params.n_points:,}\n"
            f"Max H1 persistence: {mp_h1:.4f}\n"
            f"Betti number:       {bn_h1}\n"
            f"Saved:              {'no (--no-save)' if no_save else str(library_dir)}\n",
            title="Analysis complete",
        )
    )


# ---------------------------------------------------------------------------
# profile
# ---------------------------------------------------------------------------


@main.group()
def profile():
    """Manage the profile library."""
    pass


@profile.command("list")
@click.option(
    "--library",
    default=str(DEFAULT_LIBRARY),
    show_default=True,
    type=click.Path(),
    help="Profile library directory",
)
def profile_list(library):
    """List all entries in the profile library."""
    from .profiles.library import list_entries, load

    library_dir = Path(library)
    keys = list_entries(library_dir=library_dir)

    if not keys:
        console.print("[yellow]No entries found in library.[/yellow]")
        return

    table = Table(title=f"Profile Library — {library_dir}", show_lines=True)
    table.add_column("Key", style="bold cyan", no_wrap=True)
    table.add_column("Label", style="green")
    table.add_column("Frequency (MHz)", justify="right")
    table.add_column("Method")
    table.add_column("Max H1", justify="right")

    from sdr_topology.topology.features import max_persistence

    for key in keys:
        entry = load(key, library_dir=library_dir)
        mp = max_persistence(entry.diagram, dim=1)
        table.add_row(
            key,
            entry.label,
            f"{entry.capture_metadata.center_freq_hz / 1e6:.3f}",
            entry.embedding_params.method,
            f"{mp:.4f}",
        )

    console.print(table)


@profile.command("show")
@click.argument("key")
@click.option(
    "--library",
    default=str(DEFAULT_LIBRARY),
    show_default=True,
    type=click.Path(),
    help="Profile library directory",
)
def profile_show(key, library):
    """Show details for a single profile entry."""
    from sdr_topology.profiles.library import load
    from sdr_topology.topology.features import max_persistence, betti_numbers, lifetimes

    library_dir = Path(library)

    try:
        entry = load(key, library_dir=library_dir)
    except FileNotFoundError:
        console.print(f"[red]No entry with key '{key}' in {library_dir}[/red]")
        raise click.Abort()

    lt = lifetimes(diagram=entry.diagram, dim=1)
    mp = max_persistence(diagram=entry.diagram, dim=1)
    bn = betti_numbers(diagram=entry.diagram, dim=1)

    console.print(
        Panel(
            f"[bold]Key:[/bold] {entry.key}\n"
            f"[bold]Label:[/bold] {entry.label}\n"
            f"[bold]Notes[/bold] {entry.notes}\n"
            f"[bold cyan]Capture[/bold cyan]\n"
            f"  Frequency:   {entry.capture_metadata.center_freq_hz / 1e6:.3f} MHz\n"
            f"  Sample rate: {entry.capture_metadata.sample_rate_hz:,} S/s\n"
            f"  Samples:     {entry.capture_metadata.n_samples:,}\n"
            f"  Timestamp:   {entry.capture_metadata.timestamp_utc}\n"
            f"  Tuner:       {entry.capture_metadata.tuner}\n"
            f"  Environment: {entry.capture_metadata.environment_notes}\n"
            f"[bold cyan]Embedding:[/bold cyan]\n"
            f"  Method: {entry.embedding_params.method}\n"
            f"  Dim:    {entry.embedding_params.dim}\n"
            f"  Tau:    {entry.embedding_params.tau}\n"
            f"  Points: {entry.embedding_params.n_points}\n"
            f"[bold cyan]Topology (H1):[/bold cyan]\n"
            f"  Max persistence: {mp:.4f}\n"
            f"  Betti number:    {bn}\n"
            f"  Features:        {len(lt)}",
            title="Profile Entry — {key}",
        )
    )


@profile.command("compare")
@click.option("--key-a", default=None, help="First entry key")
@click.option("--key-b", default=None, help="Second entry key")
@click.option(
    "--label", default=None, help="Compare all entries with this label pairwise"
)
@click.option(
    "--dim",
    default=1,
    show_default=True,
    type=int,
    help="Homology dimension for Wasserstein distance",
)
@click.option(
    "--library",
    default=str(DEFAULT_LIBRARY),
    show_default=True,
    type=click.Path(),
    help="Profile library directory",
)
def profile_compare(key_a, key_b, label, dim, library):
    """
    Compare profile entries by Wasserstein distance.

    Either provide --key-a and --key-b for a direct comparison,
    or --label to compare all entries of that label pairwise.
    """
    from sdr_topology.profiles.library import load, query
    from sdr_topology.topology.features import wasserstein_distance

    library_dir = Path(library)

    if key_a and key_b:
        try:
            entry_a = load(key_a, library_dir)
            entry_b = load(key_b, library_dir)
        except FileNotFoundError as e:
            console.print(f"[red]{e}[/red]")
            raise click.Abort()

        dist = wasserstein_distance(entry_a.diagram, entry_b.diagram, dim=dim)

        console.print(
            Panel(
                f"[bold]Entry A:[/bold] {entry_a.key} ({entry_a.label})\n"
                f"[bold]Entry B:[/bold] {entry_b.key} ({entry_b.label})\n"
                f"[bold cyan]Wasserstein distance (H{dim}):[/bold cyan] {dist:.6f}",
                title="Pairwise comparison",
            )
        )
    elif label:
        # Pairwise comparison across all entries with this label
        entries = query(library_dir=library_dir, label=label)

        if len(entries) < 2:
            console.print(
                f"[yellow]Need at least 2 entries with label '{label}'"
                f" to compare. Found {len(entries)}.[/yellow]"
            )
            return

        table = Table(
            title=f"Pairwise Wasserstein Distance (H{dim}) — label: {label}",
            show_lines=True,
        )
        table.add_column("Entry A", style="cyan", no_wrap=True)
        table.add_column("Entry B", style="cyan", no_wrap=True)
        table.add_column(f"W-dist (H{dim})", justify="right", style="bold")

        for i, ea in enumerate(entries):
            for eb in entries[i + 1 :]:
                dist = wasserstein_distance(ea.diagram, eb.diagram, dim=dim)
                table.add_row(ea.key, eb.key, f"{dist:.6f}")

        console.print(table)
    else:
        console.print("[red]Provide either --key-a and --key-b, or --label.[/red]")
        raise click.Abort()


if __name__ == "__main__":
    main()
