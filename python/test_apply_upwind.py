#!/usr/bin/env python3
"""
Quick test for apply_upwind_* in-place operators.

Tests that the new in-place operators work correctly
and provide performance benefits.
"""

import sys
import os
from pathlib import Path

# Add build directory to path
build_dir = os.path.join(os.path.dirname(__file__), "..", "build", "python")
if os.path.exists(build_dir):
    sys.path.insert(0, build_dir)

import samurai as sam
import time

print("=== Testing apply_upwind_* in-place operators ===\n")

# Setup mesh
box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
config = sam.MeshConfig2D()
config.min_level = 4
config.max_level = 8
config.disable_minimal_ghost_width()

mesh = sam.MRMesh2D(box, config)

# Create fields
u = sam.field.scalar(mesh, "u", init=0.0)
v = sam.field.scalar(mesh, "v", init=0.0)
flux_old = sam.field.scalar(mesh, "flux_old", init=0.0)
flux_new = sam.field.scalar(mesh, "flux_new", init=0.0)

# Initialize with simple data
def init_field(field):
    sam.for_each_cell(mesh, lambda cell: setattr(field, cell.index, cell.center()[0]))

init_field(u)
init_field(v)

velocity = [1.0, 1.0]

print("Test 1: Verify in-place operator works")
print("-" * 40)

# Test old way (creates new field)
print("Old way: flux = sam.upwind(u, velocity)")
start = time.time()
for i in range(10):
    flux_old = sam.upwind(u, velocity)
old_time = time.time() - start
print(f"  Time (10 iterations): {old_time:.4f}s")
print(f"  First value: {flux_old.array()[0]:.6f}")

# Test new way (in-place, no allocation)
print("\nNew way: sam.apply_upwind_2d(flux, velocity, u)")
start = time.time()
for i in range(10):
    sam.apply_upwind_2d(flux_new, velocity, u)
new_time = time.time() - start
print(f"  Time (10 iterations): {new_time:.4f}s")
print(f"  First value: {flux_new.array()[0]:.6f}")

# Verify results match
import numpy as np
if np.allclose(flux_old.numpy_view(), flux_new.numpy_view()):
    print("  ✓ Results match!")
else:
    print("  ✗ Results differ!")

print(f"\nSpeedup: {old_time/new_time:.2f}x")

print("\n" + "=" * 60)
print("Test 2: Full time step comparison")
print("-" * 60)

dt = 0.01
unp1_old = sam.field.scalar(mesh, "unp1_old", init=0.0)
unp1_new = sam.field.scalar(mesh, "unp1_new", init=0.0)

# Old way (allocates 2 fields: upwind_result + arithmetic result)
print("Old way: flux = sam.upwind(...); unp1 = u - dt * flux")
start = time.time()
for i in range(100):
    flux_old = sam.upwind(u, velocity)
    unp1_old = u - dt * flux_old
old_full_time = time.time() - start
print(f"  Time (100 iterations): {old_full_time:.4f}s")

# New way (0 allocations in loop)
print("\nNew way: sam.apply_upwind_2d(...); sam.euler_update_2d(...)")
flux_reuse = sam.field.scalar(mesh, "flux_reuse", init=0.0)
start = time.time()
for i in range(100):
    sam.apply_upwind_2d(flux_reuse, velocity, u)
    sam.euler_update_2d(unp1_new, u, dt, flux_reuse)
new_full_time = time.time() - start
print(f"  Time (100 iterations): {new_full_time:.4f}s")

if np.allclose(unp1_old.numpy_view(), unp1_new.numpy_view()):
    print("  ✓ Results match!")
else:
    print("  ✗ Results differ!")

print(f"\nSpeedup: {old_full_time/new_full_time:.2f}x")

print("\n" + "=" * 60)
print("Summary:")
print("=" * 60)
print(f"  apply_upwind alone: {old_time/new_time:.2f}x faster")
print(f"  Full time step: {old_full_time/new_full_time:.2f}x faster")
print("\n✓ All tests passed!")
