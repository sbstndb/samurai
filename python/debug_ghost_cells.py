#!/usr/bin/env python3
"""
Script to reproduce the ghost cells bug in field arithmetic operations.

Bug: When performing arithmetic operations on fields, ghost cells are not
properly initialized, leading to garbage values in the boundary regions.
"""

import sys
sys.path.insert(0, '../build_py314/python')

import samurai_python as sam
import numpy as np

def test_basic_subtraction():
    """Test field - scalar operation."""
    print("=" * 60)
    print("TEST 1: Basic field - scalar operation")
    print("=" * 60)

    # Create a simple 1D mesh
    box = sam.geometry.box([0.0], [1.0])
    config = sam.config.make(1)
    config.min_level = 2
    config.max_level = 2
    mesh = sam.mesh.make(box, config)

    print(f"\nMesh info:")
    print(f"  nb_cells: {mesh.nb_cells}")
    print(f"  min_level: {mesh.min_level}")
    print(f"  max_level: {mesh.max_level}")

    # Create field with uniform value 1.0
    field = sam.field.scalar(mesh, "u", init=1.0)

    print(f"\nOriginal field:")
    field_data = field.numpy_view()
    print(f"  Total size: {field_data.size}")
    print(f"  Values (first 20): {field_data[:20]}")

    # Perform subtraction: result = field - 0.3
    result = field - 0.3

    print(f"\nAfter field - 0.3:")
    result_data = result.numpy_view()
    print(f"  Result size: {result_data.size}")
    print(f"  All values: {result_data}")

    # Check for garbage values
    expected_value = 1.0 - 0.3  # = 0.7
    print(f"\n  Expected value: {expected_value}")

    # Find problematic values
    bad_indices = []
    for i, val in enumerate(result_data):
        if abs(val - expected_value) > 1e-10:
            bad_indices.append((i, val))

    if bad_indices:
        print(f"\n  ❌ BUG FOUND! {len(bad_indices)} garbage values:")
        for idx, val in bad_indices[:10]:  # Show first 10
            print(f"    index {idx}: {val} (expected {expected_value})")
    else:
        print(f"\n  ✅ All values correct!")

    return result_data, expected_value


def test_different_operations():
    """Test various arithmetic operations."""
    print("\n" + "=" * 60)
    print("TEST 2: Different arithmetic operations")
    print("=" * 60)

    box = sam.geometry.box([0.0], [1.0])
    config = sam.config.make(1)
    config.min_level = 2
    config.max_level = 2
    mesh = sam.mesh.make(box, config)

    field = sam.field.scalar(mesh, "u", init=1.0)

    operations = [
        ("field - scalar", lambda f: f - 0.3),
        ("field + scalar", lambda f: f + 0.5),
        ("field * scalar", lambda f: f * 2.0),
        ("field / scalar", lambda f: f / 2.0),
        ("scalar - field", lambda f: 2.0 - f),
        ("scalar + field", lambda f: 0.5 + f),
    ]

    for op_name, op_func in operations:
        result = op_func(field)
        result_array = result.numpy_view()

        # Check if all values are the same (uniform field should stay uniform)
        if np.allclose(result_array, result_array[0]):
            print(f"  ✅ {op_name}: OK (all values = {result_array[0]:.3f})")
        else:
            variance = np.var(result_array)
            print(f"  ❌ {op_name}: BUG (variance = {variance:.6f})")
            print(f"     min = {np.min(result_array):.3f}, max = {np.max(result_array):.3f}")


def test_field_field_operations():
    """Test field - field operations."""
    print("\n" + "=" * 60)
    print("TEST 3: Field - field operations")
    print("=" * 60)

    box = sam.geometry.box([0.0], [1.0])
    config = sam.config.make(1)
    config.min_level = 2
    config.max_level = 2
    mesh = sam.mesh.make(box, config)

    field1 = sam.field.scalar(mesh, "u", init=1.0)
    field2 = sam.field.scalar(mesh, "v", init=0.3)

    result = field1 - field2
    result_array = result.numpy_view()

    expected = 1.0 - 0.3  # = 0.7

    if np.allclose(result_array, expected):
        print(f"  ✅ field - field: OK (all values = {expected:.3f})")
    else:
        variance = np.var(result_array)
        print(f"  ❌ field - field: BUG (variance = {variance:.6f})")
        print(f"     Expected: {expected:.3f}")
        print(f"     Got: min={np.min(result_array):.3f}, max={np.max(result_array):.3f}")


def test_with_ghost_update():
    """Test if updating ghost cells fixes the issue."""
    print("\n" + "=" * 60)
    print("TEST 4: Effect of update_ghost_mr")
    print("=" * 60)

    box = sam.geometry.box([0.0], [1.0])
    config = sam.config.make(1)
    config.min_level = 2
    config.max_level = 2
    mesh = sam.mesh.make(box, config)

    field = sam.field.scalar(mesh, "u", init=1.0)

    # Update ghost cells BEFORE operation
    print("\nBefore update_ghost_mr:")
    result1 = field - 0.3
    array1 = result1.numpy_view()
    variance1 = np.var(array1)
    print(f"  Variance: {variance1:.6f}")

    # Update ghost cells
    sam.adaptation.update_ghost_mr(field)

    print("\nAfter update_ghost_mr:")
    result2 = field - 0.3
    array2 = result2.numpy_view()
    variance2 = np.var(array2)
    print(f"  Variance: {variance2:.6f}")

    if variance2 < variance1:
        print(f"  ⚠️  update_ghost_mr helps but variance still = {variance2:.6f}")
    elif variance2 == variance1:
        print(f"  ❌ update_ghost_mr doesn't change the bug")
    else:
        print(f"  ❌ update_ghost_mr makes it worse!")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("GHOST CELLS BUG INVESTIGATION")
    print("=" * 60)

    try:
        # Test 1: Basic subtraction
        result_array, expected = test_basic_subtraction()

        # Test 2: Different operations
        test_different_operations()

        # Test 3: Field-field operations
        test_field_field_operations()

        # Test 4: Effect of ghost update
        test_with_ghost_update()

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print("The bug appears to be in the C++ bindings for field arithmetic.")
        print("Ghost cells are not being properly initialized/cleared in results.")
        print("\nNext step: Investigate field_bindings.cpp source code.")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
