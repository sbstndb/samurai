# Samurai Python Bindings

Python bindings for the **Samurai** library - Adaptive Mesh Refinement (AMR) and Multiresolution Analysis for numerical PDE solvers.

## Status: Standalone Package

**As of v0.30.0, the Python bindings are now a standalone package!** This directory contains a complete, independently buildable Python package that requires the Samurai C++ library to be installed separately.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Development](#development)
- [Documentation](#documentation)
- [Examples](#examples)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

---

## Installation

### Prerequisites

The Python bindings require the **Samurai C++ library** to be installed on your system. The C++ library must be installed in a location where CMake can find it (via `CMAKE_PREFIX_PATH` or standard system paths).

### Option 1: Install via Conda (Recommended)

```bash
# Install the C++ library
conda install -c conda-forge samurai

# Install the Python bindings
conda install -c conda-forge samurai-python
```

### Option 2: Build from Source with pip

**Step 1: Install the C++ library**

```bash
# From the samurai_pybind11 root directory
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
sudo cmake --install build --prefix /usr/local
```

**Step 2: Install the Python bindings**

```bash
cd python/
pip install .
```

### Option 3: Development Installation

For active development, install in editable mode:

```bash
cd python/
pip install -e .
```

### Verify Installation

```python
import samurai_python as sam
print(sam.__version__)  # Should print version number
```

---

## Quick Start

Here's a minimal example to get you started:

```python
import samurai_python as sam

# 1. Create a computational domain
box = sam.geometry.box([0., 0.], [1., 1.])

# 2. Create a mesh with AMR capability
mesh = sam.mesh.make(box, min_level=2, max_level=6)

# 3. Create a field
u = sam.field.scalar(mesh, "u")

# 4. Initialize the field
for cell in mesh.cells():
    u[cell] = initial_condition(cell.center)

# 5. Apply boundary conditions
bc = sam.boundary.dirichlet(u, value=0.0)

# 6. Perform mesh adaptation
MRadapt = sam.adaptation.make_MRAdapt(u)
MRadapt(epsilon=1e-4, regularity=1)
```

For more detailed tutorials, see the [examples/](examples/) directory.

---

## Development

### Setting Up Development Environment

```bash
# Create a conda environment
conda create -n samurai-dev python=3.11
conda activate samurai-dev

# Install dependencies
pip install numpy pytest black isort ruff mypy

# Install in editable mode
cd python/
pip install -e .
```

### Building from Source

#### Using the dev.py script (recommended)

```bash
cd python/

# Build only
python dev.py build

# Build and install (editable mode)
python dev.py install

# Run tests
python dev.py test

# Clean build artifacts
python dev.py clean

# Do everything
python dev.py all
```

#### Using CMake directly

```bash
cd python/
mkdir build && cd build

# Configure
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DSAMURAI_PYTHON_STANDALONE=ON \
    -DCMAKE_PREFIX_PATH=/path/to/samurai/install

# Build
cmake --build .

# Test
python -c "import samurai_python; print(samurai_python.__version__)"
```

#### Using Make

```bash
cd python/
make build          # Build
make install        # Install (editable mode)
make test           # Run tests
make clean          # Clean
make all            # Build + install + test
```

### Code Quality

```bash
# Format code
make fmt

# Lint
make lint

# Type check
make check

# Run all checks
make check-all
```

---

## Documentation

### API Documentation

The Python bindings are organized into submodules:

- **`sam.geometry`** - Geometric primitives (`Box`, `DomainBuilder`)
- **`sam.config`** - Mesh configuration (`MeshConfig`, `MRAConfig`)
- **`sam.mesh`** - Mesh types (`UniformMesh`, `MRMesh`)
- **`sam.field`** - Fields (`ScalarField`, `VectorField`)
- **`sam.algorithms`** - Iteration algorithms (`for_each_cell`, `for_each_interval`)
- **`sam.operators`** - Differential operators (`upwind`, `diffusion`)
- **`sam.boundary`** - Boundary conditions (`Dirichlet`, `Neumann`)
- **`sam.adaptation`** - Mesh adaptation (`MRAdapt`, `update_ghost_mr`)
- **`sam.io`** - HDF5 I/O (`save`, `load`)

### Full Documentation

For complete documentation, visit: [https://hpc-math-samurai.readthedocs.io](https://hpc-math-samurai.readthedocs.io)

---

## Examples

The `examples/` directory contains complete, runnable examples:

| Example | Description |
|---------|-------------|
| `advection.py` | 2D advection equation with AMR |
| `burgers.py` | 2D Burgers equation with WENO5 |
| `convection.py` | Linear convection with obstacles |
| `demo_progress.py` | Progress bar demonstration |
| `demo_visualization.py` | Real-time visualization |

Run an example:

```bash
cd examples/
python advection.py
```

---

## Testing

### Run All Tests

```bash
cd python/
pytest tests/ -v
```

### Run Specific Tests

```bash
# Basic tests
pytest tests/test_basic.py -v

# Field tests
pytest tests/test_field.py -v

# Adaptation tests
pytest tests/test_adapt.py -v
```

### Standalone Tests

For testing the standalone build specifically:

```bash
pytest tests/test_standalone.py -v
```

---

## Contributing

We welcome contributions! Please see the main [Samurai repository](https://github.com/hpc-maths/samurai) for guidelines.

### Development Workflow

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add/update tests
5. Run code quality checks: `make check-all`
6. Submit a pull request

---

## Migration from Old Build System

If you were using the old `BUILD_PYTHON_BINDINGS` CMake option, you need to update your workflow:

### Old Way (Deprecated)

```bash
# From samurai_pybind11 root
cmake -B build -DBUILD_PYTHON_BINDINGS=ON
cmake --build build
```

### New Way

```bash
# 1. Build and install the C++ library
cd /path/to/samurai_pybind11
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
sudo cmake --install build --prefix /usr/local

# 2. Build and install Python bindings separately
cd python/
pip install .
```

See [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) for detailed migration instructions.

---

## Troubleshooting

### "samurai C++ library not found"

**Error:** `Could not find samurai`

**Solution:** Make sure the C++ library is installed:
```bash
cmake --install build --prefix /usr/local
# Or set CMAKE_PREFIX_PATH:
cmake -B build -DCMAKE_PREFIX_PATH=/path/to/samurai/install
```

### Import Errors

**Error:** `ImportError: cannot import name 'samurai_python'`

**Solution:** Make sure you've installed the package:
```bash
cd python/
pip install -e .
```

### Version Mismatch

**Error:** Compilation errors or template instantiation failures

**Solution:** Make sure the C++ library and Python bindings are built with the same configuration options (container types, etc.)

---

## License

This project is licensed under the **BSD-3-Clause License**. See the [LICENSE](../LICENSE) file for details.

---

## Citation

If you use Samurai in your research, please cite:

```bibtex
@software{samurai,
  author = {Samurai Development Team},
  title = {Samurai: Adaptive Mesh Refinement Library},
  url = {https://github.com/hpc-maths/samurai},
  year = {2024}
}
```

---

## Contact

- **Issues:** [GitHub Issues](https://github.com/hpc-maths/samurai/issues)
- **Discussions:** [GitHub Discussions](https://github.com/hpc-maths/samurai/discussions)
- **Email:** samurai@lists.sciencesconf.org

---

## Acknowledgments

- Built with [pybind11](https://github.com/pybind/pybind11)
- Uses [scikit-build-core](https://github.com/scikit-build/scikit-build-core) for modern Python packaging
- Depends on [xtensor](https://github.com/xtensor-stack/xtensor) for array operations
