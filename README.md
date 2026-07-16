# SDR Topology

A Python project for analyzing Software Defined Radio (SDR) topology using topological data analysis and signal processing.

## Features

- SDR signal capture and analysis
- Topological data analysis with persistent homology
- Visualization of signal characteristics
- Python 3.14+ support

## Installation

### Prerequisites

- Python 3.14 or later

### As a Package (End Users)

Install directly from GitHub:

```bash
pip install git+https://github.com/tkcn568/sdr-topology.git
```

Or with uv:

```bash
uv pip install git+https://github.com/tkcn568/sdr-topology.git
```

Once installed, the `sdrtopo` command is available system-wide.

### For Development

Clone the repository and use [uv](https://docs.astral.sh/uv/) for dependency management:

```bash
git clone https://github.com/tkcn568/sdr-topology.git
cd sdr-topology
uv sync
```

In development mode, use `uv run sdrtopo` to invoke the CLI.

## Usage

### Command-line interface

The `sdrtopo` command provides tools for SDR signal analysis. Usage varies by installation method:

**Installed as a package:**
```bash
sdrtopo <command> [options]
```

**Development mode (from cloned repo):**
```bash
uv run sdrtopo <command> [options]
```

### Common commands

```bash
# Capture IQ samples from RTL-SDR hardware
sdrtopo capture --freq 99500000 --samples 250000 --output capture.npy

# Analyze a capture with IQ-plane embedding
sdrtopo analyze --capture-path capture.npy --method iq --label fm_broadcast

# Analyze with time-delay embedding
sdrtopo analyze --capture-path capture.npy --method delay --label noise_floor

# List saved signal profiles
sdrtopo list

# Show a specific profile entry
sdrtopo show <profile_key>

# Compare two profiles using Wasserstein distance
sdrtopo compare <profile_key_1> <profile_key_2>

# Plot a persistence diagram
sdrtopo plot <profile_key> --output diagram.png
```

For full command options, run:
```bash
sdrtopo --help
```

## Development

### Installing development dependencies

Development dependencies are defined in `pyproject.toml`. To install them:

```bash
uv sync
```

### Running tests

```bash
uv run pytest
```

### Code quality

This project uses Ruff for linting and formatting. Run checks with:

```bash
uv run ruff check .
uv run ruff format .
```

### CI and Security

GitHub Actions workflows run automatically on pull requests and pushes to main:

- **CI workflow** (`ci.yml`) — Format check, lint, and test suite
- **CodeQL scanning** (`codeql.yml`) — Security analysis on PR, push, and weekly schedule

**Dependabot:** Enable automated dependency updates in repository settings
(Settings → Code security and analysis → Dependabot).

## Dependencies

### Core
- **numpy** - Numerical computing
- **scipy** - Scientific computing
- **matplotlib** - Plotting and visualization
- **ripser** - Persistent homology computation
- **persim** - Persistence diagram similarity measures

### CLI & RTL-SDR
- **click** - Command-line interface framework
- **rich** - Terminal formatting and rendering
- **pyrtlsdr** - RTL-SDR device interface
- **pyrtlsdrlib** - RTL-SDR library bindings

## Project Structure

- `sdr_topology/` - Main package
  - `cli.py` - Command-line interface (entry point: `sdrtopo`)
  - `pipeline.py` - End-to-end workflows for embedding and persistence
  - `capture/` - RTL-SDR hardware capture
  - `embedding/` - Embedding methods (IQ plane, time-delay)
  - `topology/` - Persistent homology computation and feature extraction
  - `profiles/` - Signal profile library management
  - `visualization/` - Diagram and embedding plotting
  - `logging.py` - Logging configuration
- `tests/` - Test suite (organized by module)
- `docs/` - Documentation and examples

## Gotchas

### macOS RTL-SDR Issues

This project was developed on macOS and encountered several platform-specific challenges with RTL-SDR:

#### LIBUSB Overflow Errors

Direct calls to `pyrtlsdr` can trigger `LIBUSB_ERROR_OVERFLOW` errors, particularly with synchronous read operations. See [pyrtlsdr#4](https://github.com/pyrtlsdr/pyrtlsdr/issues/4). Chunk-based read workarounds also fail on macOS.

**Solution:** Use the CLI command `rtl_sdr` via `subprocess` instead of direct library calls. This is more reliable and is implemented in `sdr_topology/capture/rtlsdr.py`.

#### librtlsdr Dynamic Linker Issues

macOS Homebrew installs librtlsdr 2.0.2, but the dylib is for version 2.0.1. Building from source doesn't resolve the dynamic linker mismatch.

**Solution:** Install `pyrtlsdrlib`, which includes pre-built binaries that `pyrtlsdr` can locate. This dependency is included in the project.

## Note on AI-Generated Content

Documentation, commit messages, and unit tests in this project may have been generated or assisted by large language models. While reviewed and validated, readers should be aware of this when evaluating the material.

## License

See LICENSE file for details.
