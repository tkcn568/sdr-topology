# SDR Topology

A Python project for analyzing Software Defined Radio (SDR) topology using topological data analysis and signal processing.

## Features

- SDR signal capture and analysis
- Topological data analysis with persistent homology
- Visualization of signal characteristics
- Python 3.14+ support

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

### Prerequisites

- Python 3.14 or later
- uv package manager

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/tkcn568/sdr-topology.git
   cd sdr-topology
   ```

2. Install dependencies:
   ```bash
   uv sync
   ```

## Usage

### Command-line interface

The `sdrtopo` command provides tools for SDR signal analysis:

```bash
# Capture IQ samples from RTL-SDR hardware
uv run sdrtopo capture --freq 99500000 --samples 250000 --output capture.npy

# Analyze a capture with IQ-plane embedding
uv run sdrtopo analyze --capture-path capture.npy --method iq --label fm_broadcast

# Analyze with time-delay embedding
uv run sdrtopo analyze --capture-path capture.npy --method delay --label noise_floor

# List saved signal profiles
uv run sdrtopo list

# Show a specific profile entry
uv run sdrtopo show <profile_key>

# Compare two profiles using Wasserstein distance
uv run sdrtopo compare <profile_key_1> <profile_key_2>

# Plot a persistence diagram
uv run sdrtopo plot <profile_key> --output diagram.png
```

For full command options, run:
```bash
uv run sdrtopo --help
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
