#!/usr/bin/env python3
"""
Test script to verify the sam.operators submodule is accessible and functional.
"""
import sys

# Import the samurai_python module directly from the .so file
import importlib.util
spec = importlib.util.spec_from_file_location("samurai_python", "/home/sbstndbs/sbstndbs/samurai/build/python/samurai_python.cpython-312-x86_64-linux-gnu.so")
samurai_python = importlib.util.module_from_spec(spec)
sys.modules["samurai_python"] = samurai_python
spec.loader.exec_module(samurai_python)

sam = samurai_python

def test_operators_submodule_exists():
    """Test that the operators submodule exists."""
    assert hasattr(sam, 'operators'), "sam.operators submodule does not exist"
    print("✓ sam.operators submodule exists")

def test_operators_has_functions():
    """Test that the operators submodule has the expected functions."""
    ops = sam.operators

    # Check for the main operator functions
    assert hasattr(ops, 'upwind'), "sam.operators.upwind does not exist"
    print("✓ sam.operators.upwind exists")

    assert hasattr(ops, 'apply_upwind_1d'), "sam.operators.apply_upwind_1d does not exist"
    print("✓ sam.operators.apply_upwind_1d exists")

    assert hasattr(ops, 'apply_upwind_2d'), "sam.operators.apply_upwind_2d does not exist"
    print("✓ sam.operators.apply_upwind_2d exists")

    assert hasattr(ops, 'apply_upwind_3d'), "sam.operators.apply_upwind_3d does not exist"
    print("✓ sam.operators.apply_upwind_3d exists")

    assert hasattr(ops, 'make_convection_weno5'), "sam.operators.make_convection_weno5 does not exist"
    print("✓ sam.operators.make_convection_weno5 exists")

def test_backward_compatibility():
    """Test that operators are still accessible from the main samurai module."""
    # Check that operators are still in the main module
    assert hasattr(sam, 'upwind'), "sam.upwind does not exist (backward compatibility broken)"
    print("✓ sam.upwind exists (backward compatible)")

    assert hasattr(sam, 'apply_upwind_1d'), "sam.apply_upwind_1d does not exist (backward compatibility broken)"
    print("✓ sam.apply_upwind_1d exists (backward compatible)")

    assert hasattr(sam, 'make_convection_weno5'), "sam.make_convection_weno5 does not exist (backward compatibility broken)"
    print("✓ sam.make_convection_weno5 exists (backward compatible)")

def test_functions_are_same():
    """Test that the functions in operators submodule are the same as in the main module."""
    ops = sam.operators

    # Check that they reference the same function objects
    assert ops.upwind is sam.upwind, "sam.operators.upwind and sam.upwind are not the same object"
    print("✓ sam.operators.upwind and sam.upwind are the same object")

    assert ops.apply_upwind_1d is sam.apply_upwind_1d, "sam.operators.apply_upwind_1d and sam.apply_upwind_1d are not the same object"
    print("✓ sam.operators.apply_upwind_1d and sam.apply_upwind_1d are the same object")

    assert ops.make_convection_weno5 is sam.make_convection_weno5, "sam.operators.make_convection_weno5 and sam.make_convection_weno5 are not the same object"
    print("✓ sam.operators.make_convection_weno5 and sam.make_convection_weno5 are the same object")

def test_operators_docstring():
    """Test that the operators submodule has a docstring."""
    ops = sam.operators
    # The submodule should have a __doc__ attribute
    assert ops.__doc__ is not None, "sam.operators does not have a docstring"
    print(f"✓ sam.operators has a docstring: '{ops.__doc__}'")

def main():
    """Run all tests."""
    print("Testing sam.operators submodule implementation...")
    print()

    try:
        test_operators_submodule_exists()
        test_operators_has_functions()
        test_backward_compatibility()
        test_functions_are_same()
        test_operators_docstring()

        print()
        print("=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
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
        print("=" * 60)
        return 2

if __name__ == '__main__':
    sys.exit(main())
