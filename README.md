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
   git clone <repository-url>
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
- `captures/` - Signal capture files
- `docs/` - Documentation
- `main.py` - Entry point script

## License

See LICENSE file for details.
