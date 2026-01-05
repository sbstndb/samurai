#!/usr/bin/env python3
"""
Test script for Samurai Box2D Python bindings (proof-of-concept)

This script tests the basic functionality of the Box2D class exposed
to Python via pybind11.
"""

import sys
import numpy as np

# The module will be built in build/python directory
sys.path.insert(0, 'build/python')

try:
    import samurai_core
    print("=== samurai_core module imported successfully ===")
    print(f"Module version: {samurai_core.__version__}")
    print()
except ImportError as e:
    print(f"Failed to import samurai_core: {e}")
    print("Did you build the module with:")
    print("  cmake -B build -DBUILD_PYTHON_BINDINGS=ON")
    print("  cmake --build build")
    sys.exit(1)


def test_box_construction():
    """Test basic Box2D construction."""
    print("--- Test 1: Box Construction ---")

    # Create a box from corners (use numpy arrays)
    box = samurai_core.Box2D(np.array([0., 0.]), np.array([1., 1.]))
    print(f"Created box: {box}")

    # Check corners via properties
    min_corner = box.min
    max_corner = box.max
    print(f"  min_corner: {min_corner} (type: {type(min_corner)})")
    print(f"  max_corner: {max_corner} (type: {type(max_corner)})")

    # Verify values
    assert np.allclose(min_corner, [0., 0.]), f"Expected min=[0, 0], got {min_corner}"
    assert np.allclose(max_corner, [1., 1.]), f"Expected max=[1, 1], got {max_corner}"

    # Check that they are numpy arrays
    assert isinstance(min_corner, np.ndarray), "min_corner should be numpy array"
    assert isinstance(max_corner, np.ndarray), "max_corner should be numpy array"

    print("  PASSED")


def test_box_default_constructor():
    """Test default Box2D constructor."""
    print("\n--- Test 2: Default Constructor ---")

    box = samurai_core.Box2D()
    print(f"Default box: {box}")
    print(f"  min: {box.min}")
    print(f"  max: {box.max}")

    # Default should have corners at origin
    assert np.allclose(box.min, [0., 0.]), "Default min should be [0, 0]"
    assert np.allclose(box.max, [0., 0.]), "Default max should be [0, 0]"

    print("  PASSED")


def test_box_properties():
    """Test Box2D properties and methods."""
    print("\n--- Test 3: Box Properties ---")

    box = samurai_core.Box2D(np.array([-1., -2.]), np.array([3., 4.]))
    print(f"Box: {box}")

    # Test length method
    length = box.length()
    print(f"  length: {length}")
    expected_length = np.array([4., 6.])  # [3-(-1), 4-(-2)]
    assert np.allclose(length, expected_length), f"Expected length {expected_length}, got {length}"

    # Test min_length method
    min_len = box.min_length()
    print(f"  min_length: {min_len}")
    assert min_len == 4.0, f"Expected min_length=4.0, got {min_len}"

    # Test is_valid
    is_valid = box.is_valid()
    print(f"  is_valid: {is_valid}")
    assert is_valid, "Box should be valid"

    print("  PASSED")


def test_box_geometric_operations():
    """Test Box2D geometric operations."""
    print("\n--- Test 4: Geometric Operations ---")

    box1 = samurai_core.Box2D(np.array([0., 0.]), np.array([2., 2.]))
    box2 = samurai_core.Box2D(np.array([1., 1.]), np.array([3., 3.]))

    print(f"box1: {box1}")
    print(f"box2: {box2}")

    # Test intersection
    intersects = box1.intersects(box2)
    print(f"  box1.intersects(box2): {intersects}")
    assert intersects, "Boxes should intersect"

    # Get intersection box
    inter_box = box1.intersection(box2)
    print(f"  intersection box: {inter_box}")
    assert np.allclose(inter_box.min, [1., 1.]), "Intersection min should be [1, 1]"
    assert np.allclose(inter_box.max, [2., 2.]), "Intersection max should be [2, 2]"

    # Test non-intersecting boxes
    box3 = samurai_core.Box2D(np.array([10., 10.]), np.array([12., 12.]))
    intersects_not = box1.intersects(box3)
    print(f"  box1.intersects(box3): {intersects_not}")
    assert not intersects_not, "Boxes should not intersect"

    print("  PASSED")


def test_box_difference():
    """Test Box2D difference operation."""
    print("\n--- Test 5: Box Difference ---")

    box = samurai_core.Box2D(np.array([-1., -1.]), np.array([1., 1.]))
    box_to_remove = samurai_core.Box2D(np.array([-0.5, -0.5]), np.array([0.5, 0.5]))

    print(f"Original box: {box}")
    print(f"Box to remove: {box_to_remove}")

    # Get difference
    diff_boxes = box.difference(box_to_remove)
    print(f"  Number of difference boxes: {len(diff_boxes)}")

    # Should have 8 boxes (3^2 - 1) for a centered hole
    assert len(diff_boxes) == 8, f"Expected 8 difference boxes, got {len(diff_boxes)}"

    for i, b in enumerate(diff_boxes):
        print(f"    diff[{i}]: {b}")

    print("  PASSED")


def test_box_operators():
    """Test Box2D operators."""
    print("\n--- Test 6: Operators ---")

    box1 = samurai_core.Box2D(np.array([0., 0.]), np.array([1., 1.]))
    box2 = samurai_core.Box2D(np.array([0., 0.]), np.array([1., 1.]))
    box3 = samurai_core.Box2D(np.array([1., 1.]), np.array([2., 2.]))

    # Equality
    assert box1 == box2, "box1 should equal box2"
    assert box1 != box3, "box1 should not equal box3"
    print("  Equality operators: OK")

    # Scaling
    scaled = box1 * 2.0
    print(f"  box1 * 2.0: min={scaled.min}, max={scaled.max}")
    assert np.allclose(scaled.min, [0., 0.]), "Scaled min should be [0, 0]"
    assert np.allclose(scaled.max, [2., 2.]), "Scaled max should be [2, 2]"

    # Reverse scaling
    scaled_rev = 2.0 * box1
    assert np.allclose(scaled_rev.min, scaled.min), "Reverse scaling should work"
    assert np.allclose(scaled_rev.max, scaled.max), "Reverse scaling should work"

    # In-place scaling
    box4 = samurai_core.Box2D(np.array([0., 0.]), np.array([1., 1.]))
    box4 *= 3.0
    assert np.allclose(box4.max, [3., 3.]), "In-place scaling should work"
    print("  Scaling operators: OK")

    print("  PASSED")


def test_box_setters():
    """Test Box2D setters."""
    print("\n--- Test 7: Setters ---")

    box = samurai_core.Box2D()

    # Set min corner
    box.set_min(np.array([1., 2.]))
    print(f"After set_min([1, 2]): min={box.min}")
    assert np.allclose(box.min, [1., 2.]), "set_min failed"

    # Set max corner
    box.set_max(np.array([3., 4.]))
    print(f"After set_max([3, 4]): max={box.max}")
    assert np.allclose(box.max, [3., 4.]), "set_max failed"

    print("  PASSED")


def run_all_tests():
    """Run all tests."""
    print("=" * 50)
    print("Samurai Box2D Python Bindings - Test Suite")
    print("=" * 50)

    tests = [
        test_box_construction,
        test_box_default_constructor,
        test_box_properties,
        test_box_geometric_operations,
        test_box_difference,
        test_box_operators,
        test_box_setters,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            failed += 1

    print("\n" + "=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 50)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
