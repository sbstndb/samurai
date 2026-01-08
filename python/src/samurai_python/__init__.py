"""
Samurai Python: Adaptive Mesh Refinement library

This package provides the Python bindings for Samurai, combining:
- Native C++ implementations (via pybind11 compiled module)
- Python utility modules (progress bars, visualization helpers)
"""

import sys
import os
import ctypes
import ctypes.util

# Get the build/python directory (parent of samurai_python package)
_package_dir = os.path.dirname(__file__)
_build_dir = os.path.abspath(os.path.join(_package_dir, '..'))

# Find the compiled .so file from build directory
_so_files = []
if os.path.exists(_build_dir):
    import glob
    _so_files = glob.glob(os.path.join(_build_dir, 'samurai_python*.so'))

if _so_files:
    _so_path = _so_files[0]
else:
    raise ImportError(
        f"Cannot find samurai_python compiled module in {_build_dir}. "
        "Please build the project first."
    )

# Load the .so file using ctypes and call PyInit_samurai_python directly
# This bypasses the Python import machinery that requires module name matching
_so = ctypes.PyDLL(_so_path)
# Call the module init function - it returns a PyObject* which we convert to Python object
init_func_name = "PyInit_samurai_python"
if not hasattr(_so, init_func_name):
    raise ImportError(f"Cannot find {init_func_name} in {_so_path}")

# Get the init function and call it - it returns a Python module object
PyInit_samurai_python = getattr(_so, init_func_name)
PyInit_samurai_python.restype = ctypes.py_object
_compiled_module = PyInit_samurai_python()

if _compiled_module is None:
    raise ImportError(f"Failed to initialize module from {_so_path}")

# Copy all public symbols from compiled module to this package namespace
for attr_name in dir(_compiled_module):
    if not attr_name.startswith('_'):
        globals()[attr_name] = getattr(_compiled_module, attr_name)

# Import the Python utility submodules
from . import utils

# Export utilities at package level for convenience
from .utils import progress

__all__ = [
    # Utilities
    "utils",
    "progress",
]
