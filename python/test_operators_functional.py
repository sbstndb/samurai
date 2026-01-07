#!/usr/bin/env python3
"""
Functional test script to verify the sam.operators submodule works with actual mesh and fields.
"""
import sys
import numpy as np

# Import the samurai_python module directly from the .so file
import importlib.util
spec = importlib.util.spec_from_file_location("samurai_python", "/home/sbstndbs/sbstndbs/samurai/build/python/samurai_python.cpython-312-x86_64-linux-gnu.so")
samurai_python = importlib.util.module_from_spec(spec)
sys.modules["samurai_python"] = samurai_python
spec.loader.exec_module(samurai_python)

sam = samurai_python

def test_1d_upwind_operator_callable():
    """Test that the 1D upwind operator can be called from both APIs."""
    print("Testing 1D upwind operator is callable...")

    # Create a 1D mesh
    box = sam.Box([0.0], [1.0])
    config = sam.MRConfig1D().min_level(0).max_level(0)
    mesh = sam.MRMesh1D(box, config)

    # Create a field
    u = sam.ScalarField1D("u", mesh, 0.0)

    # Initialize using for_each_cell
    def init_cell(cell):
        u[cell.index] = np.sin(2 * np.pi * cell.center()[0])

    sam.for_each_cell(mesh, init_cell)

    # Test using sam.upwind (old API) - should work without errors
    try:
        flux1 = sam.upwind(1.0, u)
        print("✓ sam.upwind(1.0, u) executed successfully")
    except Exception as e:
        print(f"✗ sam.upwind failed: {e}")
        raise

    # Test using sam.operators.upwind (new API) - should work without errors
    try:
        flux2 = sam.operators.upwind(u, 1.0)
        print("✓ sam.operators.upwind(u, 1.0) executed successfully")
    except Exception as e:
        print(f"✗ sam.operators.upwind failed: {e}")
        raise

    # Verify both produce field results
    assert flux1 is not None, "sam.upwind returned None"
    assert flux2 is not None, "sam.operators.upwind returned None"
    print("✓ Both APIs returned valid field objects")

def test_1d_apply_upwind_operator_callable():
    """Test that the 1D apply_upwind operator can be called from both APIs."""
    print("\nTesting 1D apply_upwind operator is callable...")

    # Create a 1D mesh
    box = sam.Box([0.0], [1.0])
    config = sam.MRConfig1D().min_level(0).max_level(0)
    mesh = sam.MRMesh1D(box, config)

    # Create input and output fields
    u = sam.ScalarField1D("u", mesh, 0.0)
    flux_old = sam.ScalarField1D("flux_old", mesh, 0.0)
    flux_new = sam.ScalarField1D("flux_new", mesh, 0.0)

    # Initialize using for_each_cell
    def init_cell(cell):
        u[cell.index] = np.sin(2 * np.pi * cell.center()[0])

    sam.for_each_cell(mesh, init_cell)

    # Test using sam.apply_upwind_1d (old API)
    try:
        sam.apply_upwind_1d(flux_old, 1.0, u)
        print("✓ sam.apply_upwind_1d(flux, 1.0, u) executed successfully")
    except Exception as e:
        print(f"✗ sam.apply_upwind_1d failed: {e}")
        raise

    # Test using sam.operators.apply_upwind_1d (new API)
    try:
        sam.operators.apply_upwind_1d(flux_new, 1.0, u)
        print("✓ sam.operators.apply_upwind_1d(flux, 1.0, u) executed successfully")
    except Exception as e:
        print(f"✗ sam.operators.apply_upwind_1d failed: {e}")
        raise

    print("✓ Both APIs executed successfully")

def test_2d_upwind_operator_callable():
    """Test that the 2D upwind operator can be called from both APIs."""
    print("\nTesting 2D upwind operator is callable...")

    # Create a 2D mesh
    box = sam.Box([0.0, 0.0], [1.0, 1.0])
    config = sam.MRConfig2D().min_level(0).max_level(0)
    mesh = sam.MRMesh2D(box, config)

    # Create a field
    u = sam.ScalarField2D("u", mesh, 0.0)

    # Initialize using for_each_cell
    def init_cell(cell):
        cx, cy = cell.center()
        u[cell.index] = np.sin(2 * np.pi * cx) * np.sin(2 * np.pi * cy)

    sam.for_each_cell(mesh, init_cell)

    velocity = [1.0, 1.0]

    # Test using sam.upwind (old API)
    try:
        flux1 = sam.upwind(velocity, u)
        print("✓ sam.upwind(velocity, u) executed successfully for 2D")
    except Exception as e:
        print(f"✗ sam.upwind failed for 2D: {e}")
        raise

    # Test using sam.operators.upwind (new API)
    try:
        flux2 = sam.operators.upwind(u, velocity)
        print("✓ sam.operators.upwind(u, velocity) executed successfully for 2D")
    except Exception as e:
        print(f"✗ sam.operators.upwind failed for 2D: {e}")
        raise

    print("✓ Both APIs executed successfully for 2D")

def test_weno5_operator_callable():
    """Test that the WENO5 operator can be called from both APIs."""
    print("\nTesting WENO5 convection operator is callable...")

    # Create a 1D mesh
    box = sam.Box([0.0], [1.0])
    config = sam.MRConfig1D().min_level(0).max_level(0)
    mesh = sam.MRMesh1D(box, config)

    # Create a field
    u = sam.ScalarField1D("u", mesh, 0.0)

    # Initialize using for_each_cell
    def init_cell(cell):
        u[cell.index] = np.sin(2 * np.pi * cell.center()[0])

    sam.for_each_cell(mesh, init_cell)

    # Test using sam.make_convection_weno5 (old API)
    try:
        flux1 = sam.make_convection_weno5(u)
        print("✓ sam.make_convection_weno5(u) executed successfully")
    except Exception as e:
        print(f"✗ sam.make_convection_weno5 failed: {e}")
        raise

    # Test using sam.operators.make_convection_weno5 (new API)
    try:
        flux2 = sam.operators.make_convection_weno5(u)
        print("✓ sam.operators.make_convection_weno5(u) executed successfully")
    except Exception as e:
        print(f"✗ sam.operators.make_convection_weno5 failed: {e}")
        raise

    print("✓ Both APIs executed successfully for WENO5")

def main():
    """Run all functional tests."""
    print("Running functional tests for sam.operators submodule...")
    print()

    try:
        test_1d_upwind_operator_callable()
        test_1d_apply_upwind_operator_callable()
        test_2d_upwind_operator_callable()
        test_weno5_operator_callable()

        print()
        print("=" * 60)
        print("All functional tests passed! ✓")
        print("=" * 60)
        print()
        print("Summary:")
        print("- sam.operators submodule is accessible")
        print("- All operator functions are available in sam.operators")
        print("- Backward compatibility maintained (old API still works)")
        print("- Operators can be called successfully on real meshes and fields")
        return 0

    except AssertionError as e:
        print()
        print("=" * 60)
        print(f"Test failed: {e}")
        print("=" * 60)
        return 1
    except Exception as e:
        print()
        print("=" * 60)
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 60)
        return 2

if __name__ == '__main__':
    sys.exit(main())
