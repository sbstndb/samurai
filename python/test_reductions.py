#!/usr/bin/env python3
"""
Test reduction operations (sum, mean, max, min) for ScalarField and VectorField.

This test verifies the NumPy-like reduction API that was added to make the Python
interface more Pythonic and intuitive.
"""

import sys
sys.path.insert(0, '/home/sbstndbs/sbstndbs/samurai/build/python')

import samurai_python as sam
import numpy as np


def test_scalarfield_reductions_1d():
    """Test reductions on 1D ScalarField."""
    print("Testing ScalarField1D reductions...")

    # Create a simple 1D mesh
    box = sam.geometry.box([0.0], [1.0])
    config = sam.MeshConfig1D(min_level=0, max_level=0)
    mesh = sam.MRMesh1D(box, config)

    # Create a scalar field
    u = sam.field.scalar(mesh, "u", init=0.0)

    # Fill with known values
    arr = u.numpy_view()
    arr[:] = [1.0, 2.0, 3.0, 4.0, 5.0]

    # Test sum
    total = u.sum()
    expected = np.sum(arr)
    assert abs(total - expected) < 1e-10, f"sum failed: {total} != {expected}"
    print(f"  sum: {total} (expected {expected}) ✓")

    # Test mean
    avg = u.mean()
    expected = np.mean(arr)
    assert abs(avg - expected) < 1e-10, f"mean failed: {avg} != {expected}"
    print(f"  mean: {avg} (expected {expected}) ✓")

    # Test max
    mx = u.max()
    expected = np.max(arr)
    assert abs(mx - expected) < 1e-10, f"max failed: {mx} != {expected}"
    print(f"  max: {mx} (expected {expected}) ✓")

    # Test min
    mn = u.min()
    expected = np.min(arr)
    assert abs(mn - expected) < 1e-10, f"min failed: {mn} != {expected}"
    print(f"  min: {mn} (expected {expected}) ✓")


def test_scalarfield_reductions_2d():
    """Test reductions on 2D ScalarField."""
    print("\nTesting ScalarField2D reductions...")

    # Create a simple 2D mesh
    box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
    config = sam.MeshConfig2D(min_level=0, max_level=0)
    mesh = sam.MRMesh2D(box, config)

    # Create a scalar field
    u = sam.field.scalar(mesh, "u", init=0.0)

    # Fill with known values
    arr = u.numpy_view()
    n_cells = arr.shape[0]
    arr[:] = np.arange(1, n_cells + 1, dtype=float)  # 1 to n_cells

    # Test sum
    total = u.sum()
    expected = np.sum(arr)
    assert abs(total - expected) < 1e-10, f"sum failed: {total} != {expected}"
    print(f"  sum: {total} (expected {expected}) ✓")

    # Test mean
    avg = u.mean()
    expected = np.mean(arr)
    assert abs(avg - expected) < 1e-10, f"mean failed: {avg} != {expected}"
    print(f"  mean: {avg} (expected {expected}) ✓")

    # Test max
    mx = u.max()
    expected = np.max(arr)
    assert abs(mx - expected) < 1e-10, f"max failed: {mx} != {expected}"
    print(f"  max: {mx} (expected {expected}) ✓")

    # Test min
    mn = u.min()
    expected = np.min(arr)
    assert abs(mn - expected) < 1e-10, f"min failed: {mn} != {expected}"
    print(f"  min: {mn} (expected {expected}) ✓")


def test_vectorfield_reductions_2d():
    """Test reductions on 2D VectorField."""
    print("\nTesting VectorField2D reductions...")

    # Create a simple 2D mesh
    box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
    config = sam.MeshConfig2D(min_level=0, max_level=0)
    mesh = sam.MRMesh2D(box, config)

    # Create a vector field with 2 components
    vel = sam.field.vector(mesh, "velocity", n_components=2, init=0.0)

    # Fill with known values
    arr = vel.numpy_view()  # Shape (n_cells, n_components)
    n_cells = arr.shape[0]
    for i in range(n_cells):
        arr[i, 0] = float(i + 1)      # x component: 1, 2, 3, ...
        arr[i, 1] = float(i + 1) * 2  # y component: 2, 4, 6, ...

    # Test sum over all elements
    total = vel.sum()
    expected = np.sum(arr)
    assert abs(total - expected) < 1e-10, f"sum failed: {total} != {expected}"
    print(f"  sum (all): {total} (expected {expected}) ✓")

    # Test sum by component
    sum_by_comp = vel.sum(axis="components")
    expected_sum = np.sum(arr, axis=0)
    assert np.allclose(sum_by_comp, expected_sum), f"sum by component failed"
    print(f"  sum (by component): {sum_by_comp} (expected {expected_sum}) ✓")

    # Test mean
    avg = vel.mean()
    expected = np.mean(arr)
    assert abs(avg - expected) < 1e-10, f"mean failed: {avg} != {expected}"
    print(f"  mean: {avg} (expected {expected}) ✓")

    # Test max
    mx = vel.max()
    expected = np.max(arr)
    assert abs(mx - expected) < 1e-10, f"max failed: {mx} != {expected}"
    print(f"  max: {mx} (expected {expected}) ✓")

    # Test min
    mn = vel.min()
    expected = np.min(arr)
    assert abs(mn - expected) < 1e-10, f"min failed: {mn} != {expected}"
    print(f"  min: {mn} (expected {expected}) ✓")

    # Test magnitude
    mag = vel.magnitude()
    # Compute expected magnitude manually
    expected_mag = np.sqrt(arr[:, 0]**2 + arr[:, 1]**2)
    assert np.allclose(mag, expected_mag), "magnitude failed"
    print(f"  magnitude (first 5): {mag[:5]} (expected {expected_mag[:5]}) ✓")


def test_vectorfield_3d():
    """Test reductions on 3D VectorField."""
    print("\nTesting VectorField3D reductions...")

    # Create a simple 2D mesh with 3D vectors
    box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
    config = sam.MeshConfig2D(min_level=0, max_level=0)
    mesh = sam.MRMesh2D(box, config)

    # Create a vector field with 3 components
    vel = sam.field.vector(mesh, "velocity", n_components=3, init=0.0)

    # Fill with known values
    arr = vel.numpy_view()
    n_cells = arr.shape[0]
    for i in range(n_cells):
        arr[i, 0] = float(i + 1)
        arr[i, 1] = float(i + 1) * 2
        arr[i, 2] = float(i + 1) * 3

    # Test sum over all elements
    total = vel.sum()
    expected = np.sum(arr)
    assert abs(total - expected) < 1e-10, f"sum failed: {total} != {expected}"
    print(f"  sum (all): {total} (expected {expected}) ✓")

    # Test sum by component
    sum_by_comp = vel.sum(axis="components")
    expected_sum = np.sum(arr, axis=0)
    assert np.allclose(sum_by_comp, expected_sum), f"sum by component failed"
    print(f"  sum (by component): {sum_by_comp} (expected {expected_sum}) ✓")

    # Test magnitude
    mag = vel.magnitude()
    expected_mag = np.sqrt(arr[:, 0]**2 + arr[:, 1]**2 + arr[:, 2]**2)
    assert np.allclose(mag, expected_mag), "magnitude failed"
    print(f"  magnitude (first 5): {mag[:5]} (expected {expected_mag[:5]}) ✓")


def test_comparison_with_numpy():
    """Test that reductions give same results as NumPy operations."""
    print("\nTesting consistency with NumPy...")

    # Create a simple 2D mesh
    box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
    config = sam.MeshConfig2D(min_level=0, max_level=0)
    mesh = sam.MRMesh2D(box, config)

    # Create a scalar field with random values
    u = sam.field.scalar(mesh, "u", init=0.0)
    arr = u.numpy_view()
    np.random.seed(42)
    random_values = np.random.randn(arr.shape[0])
    arr[:] = random_values

    # Compare all reductions
    assert abs(u.sum() - np.sum(random_values)) < 1e-10, "sum != np.sum"
    assert abs(u.mean() - np.mean(random_values)) < 1e-10, "mean != np.mean"
    assert abs(u.max() - np.max(random_values)) < 1e-10, "max != np.max"
    assert abs(u.min() - np.min(random_values)) < 1e-10, "min != np.min"

    print("  All reductions match NumPy ✓")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing NumPy-like Reduction Methods for Samurai Fields")
    print("=" * 60)

    try:
        test_scalarfield_reductions_1d()
        test_scalarfield_reductions_2d()
        test_vectorfield_reductions_2d()
        test_vectorfield_3d()
        test_comparison_with_numpy()

        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        return 0

    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
