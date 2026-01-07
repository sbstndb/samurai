#!/usr/bin/env python3
"""
Test script for VectorField __setitem__ and __getitem__
"""

import sys
import os

# Add the build directory to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
build_path = os.path.join(project_root, '..', 'build', 'python')
if os.path.exists(build_path):
    sys.path.insert(0, build_path)
else:
    sys.path.insert(0, os.path.join(os.getcwd(), 'build', 'python'))

import samurai_python as sam

print("=" * 60)
print("Testing VectorField __setitem__ and __getitem__")
print("=" * 60)

# ============================================================
# Test 1: VectorField2D_2 __setitem__ and __getitem__
# ============================================================
print("\nTest 1: VectorField2D_2 __setitem__ and __getitem__")
print("-" * 60)

try:
    # Setup 2D mesh
    box = sam.geometry.box([-1., -1.], [1., 1.])
    config = sam.config.make(2)
    config.min_level = 3
    config.max_level = 5
    config.max_stencil_size = 6

    mesh = sam.mesh.MRMesh2D(box, config)

    # Create VectorField2D_2
    u = sam.field.vector(mesh, "u", n_components=2, init=0.0)

    # Test __setitem__ with for_each_cell
    def init_velocity(cell):
        x, y = cell.center()
        u[cell.index] = [y, -x]  # u = y, v = -x

    sam.for_each_cell(mesh, init_velocity)

    # Test __getitem__
    idx = 100  # Some arbitrary cell index
    values = u[idx]
    print(f"  Cell {idx} values: {values}")
    assert len(values) == 2, f"Expected 2 values, got {len(values)}"
    assert isinstance(values[0], float), f"Expected float, got {type(values[0])}"

    print(f"  Field size: {mesh.nb_cells} cells")
    print(f"  Number of components: {u.n_components}")
    print("  PASSED ✓")

except Exception as e:
    print(f"  FAILED ✗")
    print(f"  Error: {e}")
    import traceback
    traceback.print_exc()

# ============================================================
# Test 2: VectorField3D_3 __setitem__ and __getitem__
# ============================================================
print("\nTest 2: VectorField3D_3 __setitem__ and __getitem__")
print("-" * 60)

try:
    # Setup 3D mesh
    box = sam.geometry.box([-1., -1., -1.], [1., 1., 1.])
    config = sam.config.make(3)
    config.min_level = 2
    config.max_level = 3
    config.max_stencil_size = 6

    mesh = sam.mesh.MRMesh3D(box, config)

    # Create VectorField3D_3
    u = sam.field.vector(mesh, "u", n_components=3, init=0.0)

    # Test __setitem__ with for_each_cell
    def init_velocity(cell):
        x, y, z = cell.center()
        u[cell.index] = [x, y, z]

    sam.for_each_cell(mesh, init_velocity)

    # Test __getitem__
    idx = 50  # Some arbitrary cell index
    values = u[idx]
    print(f"  Cell {idx} values: {values}")
    assert len(values) == 3, f"Expected 3 values, got {len(values)}"

    print(f"  Field size: {mesh.nb_cells} cells")
    print(f"  Number of components: {u.n_components}")
    print("  PASSED ✓")

except Exception as e:
    print(f"  FAILED ✗")
    print(f"  Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("VectorField __setitem__/__getitem__ tests complete!")
print("=" * 60)
