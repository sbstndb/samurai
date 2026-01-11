# Build System Migration Guide: Standalone Python Bindings

**Version:** 0.30.0+
**Date:** January 2026

---

## Overview

Starting with version 0.30.0, the Python bindings for Samurai have been refactored into a **standalone package** with its own build system. This guide explains what changed and how to update your build workflow.

**Note:** This is separate from the API migration guide (see `MIGRATION_GUIDE.md` for API changes).

---

## What Changed?

### Before (v0.29 and earlier)

The Python bindings were built as part of the main Samurai C++ build:

```bash
# From samurai_pybind11 root
cmake -B build -DBUILD_PYTHON_BINDINGS=ON
cmake --build build
# Module built at: build/python/samurai_python*.so
```

### After (v0.30+)

The Python bindings are now a **separately buildable package**:

```bash
# Step 1: Build and install C++ library
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
cmake --install build --prefix /usr/local

# Step 2: Build Python bindings separately
cd python/
pip install .
```

---

## Why This Change?

### Benefits

1. **Cleaner Separation of Concerns**
   - C++ library can be used independently
   - Python bindings can be versioned separately
   - Easier to maintain and test

2. **Standard Python Packaging**
   - Works with `pip` and `conda`
   - Editable installs for development
   - Standard Python tooling (pytest, black, etc.)

3. **Better Distribution**
   - Can be distributed via PyPI and conda-forge
   - Pre-built wheels for common platforms
   - No need to rebuild C++ library for every Python change

4. **Faster Development**
   - Don't need to rebuild C++ library when editing Python code
   - Incremental builds for C++ bindings only

---

## Migration Checklist

Use this checklist to migrate your workflow:

- [ ] Update installation procedure
- [ ] Update CI/CD pipelines
- [ ] Update development environment
- [ ] Update user scripts (if any)
- [ ] Update documentation

---

## Migration Scenarios

### Scenario 1: C++ Developer (C++ only)

**Your use case:** You work on the Samurai C++ library, not Python.

**Action required:** None! The C++ build is unchanged.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
cmake --install build --prefix /usr/local
```

---

### Scenario 2: Python User (Install only)

**Your use case:** You just want to use the Python bindings.

**Old way:**
```bash
# Had to build entire Samurai including C++
cmake -B build -DBUILD_PYTHON_BINDINGS=ON
cmake --build build
export PYTHONPATH=build/python:$PYTHONPATH
```

**New way (recommended - conda):**
```bash
conda install -c conda-forge samurai samurai-python
```

**New way (from source):**
```bash
# Install C++ library first (once)
sudo cmake --install build --prefix /usr/local

# Then install Python bindings
cd python/
pip install .
```

---

### Scenario 3: Python Developer

**Your use case:** You develop both C++ and Python code.

**Old way:**
```bash
cmake -B build -DBUILD_PYTHON_BINDINGS=ON
# Make changes to bindings
cmake --build build  # Rebuilds everything
export PYTHONPATH=build/python:$PYTHONPATH
pytest tests/
```

**New way:**
```bash
# Initial setup
sudo cmake --install build --prefix /usr/local
cd python/
pip install -e .  # Editable install

# Development cycle
# Make changes to bindings
python dev.py build  # Only rebuilds bindings
pytest tests/        # Tests use editable install
```

---

### Scenario 4: CI/CD Pipeline

**Old way:**
```yaml
- name: Build with Python bindings
  run: |
    cmake -B build -DBUILD_PYTHON_BINDINGS=ON
    cmake --build build
- name: Test
  run: |
    export PYTHONPATH=build/python:$PYTHONPATH
    pytest tests/
```

**New way:**
```yaml
- name: Install Samurai C++
  run: |
    cmake -B build
    cmake --build build
    sudo cmake --install build --prefix /usr/local

- name: Build Python bindings
  run: |
    cd python/
    pip install .

- name: Test
  run: |
    pytest tests/
```

---

### Scenario 5: Conda Package Builder

**New in v0.30+:** Dedicated conda recipe provided!

**Old way:** No conda support, had to build from source

**New way:**
```bash
cd python/conda-recipe/
conda build .
```

---

## Build Command Reference

### Quick Reference Table

| Task | Old Way (deprecated) | New Way |
|------|---------------------|---------|
| **Build C++** | `cmake --build build` | `cmake --build build` (unchanged) |
| **Install C++** | Manual | `cmake --install build --prefix /usr/local` |
| **Build Python** | `cmake --build build` (with `-DBUILD_PYTHON_BINDINGS=ON`) | `cd python && python dev.py build` |
| **Install Python** | Manual `PYTHONPATH` | `cd python && pip install -e .` |
| **Run tests** | `export PYTHONPATH=... && pytest` | `cd python && pytest tests/` |
| **Clean** | `rm -rf build` | `python dev.py clean` |

### New Helper Scripts

The `python/` directory now includes helper scripts:

#### Linux/macOS

```bash
cd python/
python dev.py build    # Build bindings
python dev.py install  # Install (editable mode)
python dev.py test     # Run tests
python dev.py clean    # Clean artifacts
python dev.py all      # Do everything
```

Or use Makefile:

```bash
cd python/
make build
make install
make test
make clean
make all
```

#### Windows

```cmd
cd python\
dev.bat build
dev.bat install
dev.bat test
dev.bat clean
dev.bat all
```

---

## Troubleshooting

### Problem: "samurai C++ library not found"

**Error message:**
```
Could not find samurai
```

**Cause:** The C++ library is not installed or not in CMake's search path.

**Solution:**

Option 1: Install to system location
```bash
sudo cmake --install build --prefix /usr/local
```

Option 2: Set `CMAKE_PREFIX_PATH`
```bash
cmake -B build -DCMAKE_PREFIX_PATH=/path/to/samurai/install
```

Option 3: Use conda
```bash
conda install -c conda-forge samurai
```

---

### Problem: Import errors after installation

**Error message:**
```
ImportError: cannot import name 'samurai_python'
```

**Cause:** The module is not in Python's search path.

**Solution:**

Option 1: Install properly
```bash
cd python/
pip install -e .
```

Option 2: Check installation
```bash
pip show samurai-python
python -c "import sys; print('\n'.join(sys.path))"
```

---

### Problem: Version mismatch

**Error message:**
```
Template instantiation errors
```

**Cause:** C++ library and Python bindings built with different configurations.

**Solution:**

Make sure both are built with the same options:
```bash
# For C++ library
cmake -B build -DSAMURAI_FIELD_CONTAINER=xtensor

# For Python bindings (uses same config)
cd python/
cmake -B build -DSAMURAI_PYTHON_STANDALONE=ON
```

---

## Rollback Plan

If you encounter issues and need to use the old build method:

```bash
# Still works in v0.30 (with warning)
cmake -B build -DBUILD_PYTHON_BINDINGS=ON
cmake --build build
```

**Note:** This will be removed in v1.0.

---

## Questions?

- **Documentation:** See [README.md](README.md)
- **Issues:** [GitHub Issues](https://github.com/hpc-maths/samurai/issues)
- **Discussions:** [GitHub Discussions](https://github.com/hpc-maths/samurai/discussions)
- **Email:** samurai@lists.sciencesconf.org

---

## Summary

| Aspect | Change | Action Required |
|--------|--------|-----------------|
| **Installation** | Separate from C++ build | Update build scripts |
| **Import** | No change | None |
| **API** | No change | None |
| **Development** | New helper scripts | Optional: adopt new scripts |
| **CI/CD** | Two-step build | Update pipelines |
| **Distribution** | Now supports conda | Optional: use conda packages |

---

**Last updated:** January 2026
**For version:** 0.30.0+
