#!/usr/bin/env python3
"""
Test script for VectorField WENO5 convection operators (Burgers equation)
"""

import sys
import os

# Add the build directory to path
# Get the project root (2 levels up from tests/ directory)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
build_path = os.path.join(project_root, '..', 'build', 'python')
if os.path.exists(build_path):
    sys.path.insert(0, build_path)
else:
    # Fallback: try adding relative to current directory
    sys.path.insert(0, os.path.join(os.getcwd(), 'build', 'python'))

import samurai_python as sam
import numpy as np

print("=" * 60)
print("Testing VectorField WENO5 Convection Operators")
print("=" * 60)

# ============================================================
# Test 1: VectorField2D_2 WENO5 (Burgers 2D)
# ============================================================
print("\nTest 1: 2D VectorField WENO5 (Burgers 2D)")
print("-" * 60)

try:
    # Setup 2D mesh
    box = sam.Box2D([-1., -1.], [1., 1.])
    config2d = sam.MeshConfig2D()
    config2d.min_level = 3
    config2d.max_level = 5
    config2d.max_stencil_size = 6  # Required for WENO5

    mesh2d = sam.MRMesh2D(box, config2d)

    # Create VectorField2D_2 (velocity field u = [u, v])
    u_vec = sam.field.vector(mesh2d, "u", n_components=2, init=0.0)

    # Initialize with a simple velocity field
    def init_velocity_2d(cell):
        x, y = cell.center()
        # Simple rotational velocity field
        u_vec[cell.index] = [y, -x]  # u = y, v = -x

    sam.for_each_cell(mesh2d, init_velocity_2d)

    # Apply WENO5 convection operator
    flux = sam.make_convection_weno5(u_vec)

    print(f"  Input field type: {type(u_vec).__name__}")
    print(f"  Input field name: {u_vec.name}")
    print(f"  Flux field type: {type(flux).__name__}")
    print(f"  Flux field name: {flux.name}")
    print("  PASSED ✓")

except Exception as e:
    print(f"  FAILED ✗")
    print(f"  Error: {e}")
    import traceback
    traceback.print_exc()

# ============================================================
# Test 2: VectorField3D_3 WENO5 (Burgers 3D)
# ============================================================
print("\nTest 2: 3D VectorField WENO5 (Burgers 3D)")
print("-" * 60)

try:
    # Setup 3D mesh (smaller for memory)
    box = sam.Box3D([-1., -1., -1.], [1., 1., 1.])
    config3d = sam.MeshConfig3D()
    config3d.min_level = 2
    config3d.max_level = 3
    config3d.max_stencil_size = 6  # Required for WENO5

    mesh3d = sam.MRMesh3D(box, config3d)

    # Create VectorField3D_3 (velocity field u = [u, v, w])
    u_vec = sam.field.vector(mesh3d, "u", n_components=3, init=0.0)

    # Initialize with a simple velocity field
    def init_velocity_3d(cell):
        x, y, z = cell.center()
        # Simple diverging velocity field
        u_vec[cell.index] = [x, y, z]

    sam.for_each_cell(mesh3d, init_velocity_3d)

    # Apply WENO5 convection operator
    flux = sam.make_convection_weno5(u_vec)

    print(f"  Input field type: {type(u_vec).__name__}")
    print(f"  Input field name: {u_vec.name}")
    print(f"  Flux field type: {type(flux).__name__}")
    print(f"  Flux field name: {flux.name}")
    print("  PASSED ✓")

except Exception as e:
    print(f"  FAILED ✗")
    print(f"  Error: {e}")
    import traceback
    traceback.print_exc()

# ============================================================
# Test 3: RK3 time stepping (Burgers 2D)
# ============================================================
print("\nTest 3: RK3 Time Stepping (Burgers 2D)")
print("-" * 60)

try:
    # Setup 2D mesh
    box = sam.Box2D([-1., -1.], [1., 1.])
    config = sam.MeshConfig2D()
    config.min_level = 3
    config.max_level = 5
    config.max_stencil_size = 6

    mesh = sam.MRMesh2D(box, config)

    # Create fields for RK3
    u = sam.field.vector(mesh, "u", n_components=2, init=0.0)
    u1 = sam.field.vector(mesh, "u1", n_components=2, init=0.0)
    u2 = sam.field.vector(mesh, "u2", n_components=2, init=0.0)
    unp1 = sam.field.vector(mesh, "unp1", n_components=2, init=0.0)

    # Initialize with a Gaussian pulse
    def init_gaussian(cell):
        x, y = cell.center()
        r2 = x*x + y*y
        val = np.exp(-10 * r2)
        u[cell.index] = [val, val]  # u = v = Gaussian

    sam.for_each_cell(mesh, init_gaussian)

    # Boundary conditions
    sam.make_dirichlet_bc(u, [0.0, 0.0], order=1)
    sam.make_dirichlet_bc(u1, [0.0, 0.0], order=1)
    sam.make_dirichlet_bc(u2, [0.0, 0.0], order=1)
    sam.make_dirichlet_bc(unp1, [0.0, 0.0], order=1)

    # Time step
    dt = 0.001

    # RK3 scheme (using assign for consistency with AMR examples)
    u1.assign(u - dt * sam.make_convection_weno5(u))
    u2.assign(3./4 * u + 1./4 * (u1 - dt * sam.make_convection_weno5(u1)))
    unp1.assign(1./3 * u + 2./3 * (u2 - dt * sam.make_convection_weno5(u2)))

    print(f"  Successfully performed RK3 time step")
    print(f"  dt = {dt}")
    print(f"  u type: {type(u).__name__}")
    print(f"  unp1 type: {type(unp1).__name__}")
    print("  PASSED ✓")

except Exception as e:
    print(f"  FAILED ✗")
    print(f"  Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("VectorField WENO5 tests complete!")
print("=" * 60)
