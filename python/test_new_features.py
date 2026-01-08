#!/usr/bin/env python3
"""
Test script for new Python bindings features:
1. VectorField with lambda initialization
2. copy_bc_from method
3. WENO5 with VectorField velocity
4. DomainBuilder with obstacle
"""

import sys

sys.path.insert(0, '/home/sbstndbs/sbstndbs/samurai/build/python')


import samurai_python as sam

print("=" * 60)
print("Testing New Python Bindings Features")
print("=" * 60)

# Test 1: DomainBuilder with obstacle
print("\n[Test 1] DomainBuilder with obstacle")
print("-" * 40)

domain = sam.DomainBuilder2D([-1., -1.], [1., 1.])
print(f"Initial domain: {domain}")

# Remove a rectangular region to create an obstacle
domain.remove([0.0, 0.0], [0.4, 0.4])
print(f"After remove: {domain}")
print(f"Added boxes: {len(domain.added_boxes)}")
print(f"Removed boxes: {len(domain.removed_boxes)}")

# Test 2: Create mesh from DomainBuilder
print("\n[Test 2] Create mesh from DomainBuilder")
print("-" * 40)

config = sam.MeshConfig2D()
config.min_level = 1
config.max_level = 3

mesh = sam.MRMesh2D(domain, config)
print(f"Mesh created with {mesh.nb_cells} cells")
print(f"Mesh min level: {mesh.min_level}, max level: {mesh.max_level}")

# Test 3: VectorField with lambda initialization (constant velocity)
print("\n[Test 3] VectorField with lambda initialization")
print("-" * 40)

constant_vel = [1.0, -1.0]
velocity = sam.make_vector_field(mesh, "velocity", lambda center: constant_vel, 2)

# Check some values using for_each_cell
cell_count = [0]
def check_velocity(cell):
    cell_count[0] += 1
    if cell_count[0] <= 3:
        val = velocity[cell.index]
        print(f"  velocity at {cell.center()} = {val}")

sam.for_each_cell(mesh, check_velocity)
print(f"Total cells checked: {cell_count[0]}")

# Test 4: copy_bc_from method (skip for now - BC not fully implemented)
print("\n[Test 4] copy_bc_from method")
print("-" * 40)

# Create two scalar fields
u = sam.field.scalar(mesh, "u", init=1.0)
v = sam.field.scalar(mesh, "v", init=0.0)

# Test that copy_bc_from method exists
try:
    # This method exists but may not work fully since BCs are handled internally
    v.copy_bc_from(u)
    print("Field 'v' has copy_bc_from method (BC handling is internal)")
except AttributeError as e:
    print(f"copy_bc_from not available: {e}")

# Test 5: WENO5 with VectorField velocity
print("\n[Test 5] WENO5 convection with VectorField velocity")
print("-" * 40)

# Create a scalar field for testing
u = sam.field.scalar(mesh, "u", init=1.0)

# Try to apply WENO5 convection
try:
    flux = sam.make_convection_weno5(u, velocity)
    print("WENO5 convection operator created successfully")
    print(f"Flux field name: {flux.name()}")
except Exception as e:
    print(f"WENO5 convection error: {e}")

# Test 6: VectorField with spatially varying velocity
print("\n[Test 6] VectorField with spatially varying velocity")
print("-" * 40)

def varying_velocity(center):
    x, y = center[0], center[1]
    # Velocity depends on position
    return [1.0 + 0.5 * x, -1.0 + 0.5 * y]

var_velocity = sam.make_vector_field(mesh, "var_velocity", varying_velocity, 2)

# Check some values to verify spatial variation
cell_count = [0]
def check_var_velocity(cell):
    cell_count[0] += 1
    if cell_count[0] <= 3:
        val = var_velocity[cell.index]
        x, y = cell.center()[0], cell.center()[1]
        expected = [1.0 + 0.5 * x, -1.0 + 0.5 * y]
        print(f"  var_velocity at ({x:.2f}, {y:.2f}) = {val}, expected = {expected}")

sam.for_each_cell(mesh, check_var_velocity)
print(f"Total cells checked: {cell_count[0]}")

print("\n" + "=" * 60)
print("All tests completed!")
print("=" * 60)
