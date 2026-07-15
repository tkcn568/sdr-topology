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

### Running the main script

```bash
uv run python main.py
```

### Running tests

```bash
uv run pytest
```

## Development

### Installing development dependencies

Development dependencies are defined in `pyproject.toml`. To install them:

```bash
uv sync
```

### Code quality

This project uses Ruff for linting and formatting. Run checks with:

```bash
uv run ruff check .
uv run ruff format .
```

## Dependencies

- **matplotlib** - Plotting and visualization
- **numpy** - Numerical computing
- **scipy** - Scientific computing
- **pyrtlsdr** - RTL-SDR device interface
- **pyrtlsdrlib** - RTL-SDR library bindings
- **ripser** - Persistent homology computation
- **persim** - Persistence diagram similarity measures

## Project Structure

- `sdr_topology/` - Main package
- `tests/` - Test suite
- `docs/` - Documentation
- `main.py` - Entry point script

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
