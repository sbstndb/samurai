#!/usr/bin/env python3
"""
Analyze the ghost cells bug in field arithmetic operations.
"""

import sys
sys.path.insert(0, '../build_py314/python')

import samurai_python as sam
import numpy as np

print("=" * 60)
print("GHOST CELLS BUG ANALYSIS")
print("=" * 60)

# Create a simple mesh
box = sam.geometry.box([0.0], [1.0])
config = sam.config.make(1)
config.min_level = 2
config.max_level = 2
mesh = sam.mesh.make(box, config)

print(f"\nMesh info:")
print(f"  nb_cells: {mesh.nb_cells}")

# Create field
field = sam.field.scalar(mesh, "u", init=1.0)

print(f"\nOriginal field:")
print(f"  name: {field.name}")
print(f"  size: {field.size}")
print(f"  ghosts_updated: {field.ghosts_updated}")
print(f"  values: {field.numpy_view()}")

# Update ghost cells FIRST
print("\nCalling update_ghost_mr on original field...")
sam.adaptation.update_ghost_mr(field)

print(f"After update_ghost_mr:")
print(f"  ghosts_updated: {field.ghosts_updated}")
print(f"  values: {field.numpy_view()}")

# Try arithmetic operation
print("\n" + "=" * 60)
print("ARITHMETIC OPERATION TEST")
print("=" * 60)

result = field - 0.3

print(f"\nResult of (field - 0.3):")
print(f"  name: {result.name}")
print(f"  size: {result.size}")
print(f"  ghosts_updated: {result.ghosts_updated}")
result_data = result.numpy_view()
print(f"  values: {result_data}")

expected = 0.7
correct_count = np.sum(np.isclose(result_data, expected))
total_count = len(result_data)

print(f"\n  Correct values: {correct_count}/{total_count} ({100*correct_count/total_count:.1f}%)")

# Try update_ghost_mr on result
print("\nCalling update_ghost_mr on result...")
sam.adaptation.update_ghost_mr(result)

print(f"After update_ghost_mr on result:")
print(f"  ghosts_updated: {result.ghosts_updated}")
result_data_updated = result.numpy_view()
print(f"  values: {result_data_updated}")

correct_count_updated = np.sum(np.isclose(result_data_updated, expected))
print(f"\n  Correct values: {correct_count_updated}/{total_count} ({100*correct_count_updated/total_count:.1f}%)")

print("\n" + "=" * 60)
print("CONCLUSION")
print("=" * 60)
print("BUG: Ghost cells are not initialized in arithmetic operations.")
print("Even calling update_ghost_mr AFTER the operation doesn't help")
print("because the expression template only operates on real cells.")
print("\nThe fix must be in the C++ bindings field_sub_scalar, etc.")
print("Options:")
print("1. Initialize result field with input field values (copy)")
print("2. Explicitly handle ghost cells in arithmetic operations")
