#!/usr/bin/env python3
"""
Comprehensive test suite for auto-detection and defaults in Box API

Tests:
1. Dimension auto-detection from array size
2. Default dtype (double)
3. Error cases (mismatched dimensions, invalid dimensions)
4. All dimension/type combinations
"""

import sys
import numpy as np

sys.path.insert(0, 'build/python')

import samurai_core

print("=" * 60)
print("Auto-Detection and Defaults - Test Suite")
print("=" * 60)

passed = 0
failed = 0


def test_1d_auto_detection():
    """Test 1D auto-detection."""
    print("\n--- Test 1: 1D Auto-Detection ---")

    # From 1-element arrays
    box = samurai_core.Box(np.array([0.]), np.array([1.]))
    print(f"   box = Box(np.array([0.]), np.array([1.]))")
    print(f"   Result: {box}")
    print(f"   Type: {type(box).__name__}")

    # Should be Box1D_double
    assert type(box).__name__ == "Box1D_double", f"Expected Box1D_double, got {type(box).__name__}"
    assert np.allclose(box.min, [0.])
    assert np.allclose(box.max, [1.])
    assert box.length().shape[0] == 1

    print("  ✓ PASSED")
    return True


def test_2d_auto_detection():
    """Test 2D auto-detection."""
    print("\n--- Test 2: 2D Auto-Detection ---")

    # From 2-element arrays
    box = samurai_core.Box(np.array([0., 0.]), np.array([1., 1.]))
    print(f"   box = Box(np.array([0., 0.]), np.array([1., 1.]))")
    print(f"   Result: {box}")
    print(f"   Type: {type(box).__name__}")

    assert type(box).__name__ == "Box2D_double", f"Expected Box2D_double, got {type(box).__name__}"
    assert np.allclose(box.min, [0., 0.])
    assert np.allclose(box.max, [1., 1.])
    assert box.length().shape[0] == 2

    print("  ✓ PASSED")
    return True


def test_3d_auto_detection():
    """Test 3D auto-detection."""
    print("\n--- Test 3: 3D Auto-Detection ---")

    # From 3-element arrays
    box = samurai_core.Box(np.array([0., 0., 0.]), np.array([1., 1., 1.]))
    print(f"   box = Box(np.array([0., 0., 0.]), np.array([1., 1., 1.]))")
    print(f"   Result: {box}")
    print(f"   Type: {type(box).__name__}")

    assert type(box).__name__ == "Box3D_double", f"Expected Box3D_double, got {type(box).__name__}"
    assert np.allclose(box.min, [0., 0., 0.])
    assert np.allclose(box.max, [1., 1., 1.])
    assert box.length().shape[0] == 3

    print("  ✓ PASSED")
    return True


def test_default_dtype():
    """Test that default dtype is double."""
    print("\n--- Test 4: Default dtype is double ---")

    box = samurai_core.Box(np.array([0., 0.]), np.array([1., 1.]))
    print(f"   box = Box(np.array([0., 0.]), np.array([1., 1.]))")
    print(f"   Type: {type(box).__name__}")

    # Should be double precision
    assert "double" in type(box).__name__, f"Expected 'double' in type name, got {type(box).__name__}"

    # Compare with explicit double
    box_explicit = samurai_core.Box2D_double(np.array([0., 0.]), np.array([1., 1.]))
    assert type(box) == type(box_explicit), "Default should be same as explicit double"

    print("  ✓ PASSED")
    return True


def test_float_dtype():
    """Test explicit float dtype."""
    print("\n--- Test 5: Explicit float dtype ---")

    box = samurai_core.Box(np.array([0., 0.]), np.array([1., 1.]), dtype='float')
    print(f"   box = Box(np.array([0., 0.]), np.array([1., 1.]), dtype='float')")
    print(f"   Type: {type(box).__name__}")

    assert type(box).__name__ == "Box2D_float", f"Expected Box2D_float, got {type(box).__name__}"

    # Test float32 alias
    box32 = samurai_core.Box(np.array([0., 0.]), np.array([1., 1.]), dtype='float32')
    assert type(box32).__name__ == "Box2D_float", "dtype='float32' should work too"

    print("  ✓ PASSED")
    return True


def test_dtype_aliases():
    """Test all dtype aliases."""
    print("\n--- Test 6: Dtype aliases ---")

    # Create boxes with different dtype specifications
    box_double = samurai_core.Box(np.array([0., 0.]), np.array([1., 1.]), dtype='double')
    box_float64 = samurai_core.Box(np.array([0., 0.]), np.array([1., 1.]), dtype='float64')
    box_float = samurai_core.Box(np.array([0., 0.]), np.array([1., 1.]), dtype='float')
    box_float32 = samurai_core.Box(np.array([0., 0.]), np.array([1., 1.]), dtype='float32')

    print(f"   dtype='double':  {type(box_double).__name__}")
    print(f"   dtype='float64': {type(box_float64).__name__}")
    print(f"   dtype='float':   {type(box_float).__name__}")
    print(f"   dtype='float32': {type(box_float32).__name__}")

    # All should match expected types
    assert type(box_double).__name__ == "Box2D_double"
    assert type(box_float64).__name__ == "Box2D_double"
    assert type(box_float).__name__ == "Box2D_float"
    assert type(box_float32).__name__ == "Box2D_float"

    print("  ✓ PASSED")
    return True


def test_all_dimensions_with_auto_detection():
    """Test all dimensions using auto-detection."""
    print("\n--- Test 7: All dimensions with auto-detection ---")

    boxes = []
    for dim, (min_arr, max_arr) in [
        (1, (np.array([0.]), np.array([1.]))),
        (2, (np.array([0., 0.]), np.array([1., 1.]))),
        (3, (np.array([0., 0., 0.]), np.array([1., 1., 1.]))),
    ]:
        box = samurai_core.Box(min_arr, max_arr)
        boxes.append(box)
        print(f"   {dim}D: {box} (type: {type(box).__name__})")
        assert box.length().shape[0] == dim

    print("  ✓ PASSED")
    return True


def test_all_dimensions_with_float():
    """Test all dimensions with float dtype."""
    print("\n--- Test 8: All dimensions with float dtype ---")

    test_cases = [
        (1, np.array([0.]), np.array([1.]), "Box1D_float"),
        (2, np.array([0., 0.]), np.array([1., 1.]), "Box2D_float"),
        (3, np.array([0., 0., 0.]), np.array([1., 1., 1.]), "Box3D_float"),
    ]

    for dim, min_arr, max_arr, expected_type in test_cases:
        box = samurai_core.Box(min_arr, max_arr, dtype='float')
        print(f"   {dim}D float: {box} (type: {type(box).__name__})")
        assert type(box).__name__ == expected_type

    print("  ✓ PASSED")
    return True


def test_error_mismatched_dimensions():
    """Test error when min and max have different sizes."""
    print("\n--- Test 9: Error - mismatched dimensions ---")

    try:
        box = samurai_core.Box(np.array([0.]), np.array([1., 1.]))
        print("  ✗ FAILED: Should have raised error for mismatched dimensions")
        return False
    except RuntimeError as e:
        print(f"   Correctly raised error: {e}")
        assert "same size" in str(e).lower(), f"Error message should mention size mismatch, got: {e}"

    print("  ✓ PASSED")
    return True


def test_error_invalid_dimension():
    """Test error for invalid dimensions (> 3D)."""
    print("\n--- Test 10: Error - invalid dimension (4D) ---")

    try:
        box = samurai_core.Box(np.array([0., 0., 0., 0.]), np.array([1., 1., 1., 1.]))
        print("  ✗ FAILED: Should have raised error for 4D")
        return False
    except RuntimeError as e:
        print(f"   Correctly raised error: {e}")
        assert "1, 2, or 3" in str(e) or "dimension" in str(e).lower(), f"Error should mention valid dimensions, got: {e}"

    print("  ✓ PASSED")
    return True


def test_error_invalid_dtype():
    """Test error for invalid dtype."""
    print("\n--- Test 11: Error - invalid dtype ---")

    try:
        box = samurai_core.Box(np.array([0., 0.]), np.array([1., 1.]), dtype='int')
        print("  ✗ FAILED: Should have raised error for invalid dtype")
        return False
    except RuntimeError as e:
        print(f"   Correctly raised error: {e}")
        assert "dtype" in str(e).lower() or "unsupported" in str(e).lower(), f"Error should mention dtype, got: {e}"

    print("  ✓ PASSED")
    return True


def test_operations_with_auto_detected_boxes():
    """Test that operations work correctly with auto-detected boxes."""
    print("\n--- Test 12: Operations with auto-detected boxes ---")

    # 2D
    box2d_a = samurai_core.Box(np.array([0., 0.]), np.array([2., 2.]))
    box2d_b = samurai_core.Box(np.array([1., 1.]), np.array([3., 3.]))

    assert box2d_a.intersects(box2d_b), "2D boxes should intersect"
    inter = box2d_a.intersection(box2d_b)
    assert np.allclose(inter.min, [1., 1.])
    assert np.allclose(inter.max, [2., 2.])
    print(f"   2D intersection: {inter}")

    # 3D
    box3d_a = samurai_core.Box(np.array([0., 0., 0.]), np.array([2., 2., 2.]))
    box3d_b = samurai_core.Box(np.array([1., 1., 1.]), np.array([3., 3., 3.]))

    assert box3d_a.intersects(box3d_b), "3D boxes should intersect"
    inter = box3d_a.intersection(box3d_b)
    assert np.allclose(inter.min, [1., 1., 1.])
    assert np.allclose(inter.max, [2., 2., 2.])
    print(f"   3D intersection: {inter}")

    # Scaling
    scaled = box2d_a * 2.0
    assert np.allclose(scaled.max, [4., 4.])
    print(f"   Scaled 2D box: {scaled}")

    print("  ✓ PASSED")
    return True


def test_aliases_vs_auto_detection():
    """Test that aliases and auto-detection produce same type."""
    print("\n--- Test 13: Aliases vs auto-detection ---")

    # 1D
    box1d_auto = samurai_core.Box(np.array([0.]), np.array([1.]))
    box1d_alias = samurai_core.Box1D(np.array([0.]), np.array([1.]))
    assert type(box1d_auto) == type(box1d_alias), "Box() and Box1D() should have same type"
    print(f"   1D: {type(box1d_auto).__name__} == {type(box1d_alias).__name__}")

    # 2D
    box2d_auto = samurai_core.Box(np.array([0., 0.]), np.array([1., 1.]))
    box2d_alias = samurai_core.Box2D(np.array([0., 0.]), np.array([1., 1.]))
    assert type(box2d_auto) == type(box2d_alias), "Box() and Box2D() should have same type"
    print(f"   2D: {type(box2d_auto).__name__} == {type(box2d_alias).__name__}")

    # 3D
    box3d_auto = samurai_core.Box(np.array([0., 0., 0.]), np.array([1., 1., 1.]))
    box3d_alias = samurai_core.Box3D(np.array([0., 0., 0.]), np.array([1., 1., 1.]))
    assert type(box3d_auto) == type(box3d_alias), "Box() and Box3D() should have same type"
    print(f"   3D: {type(box3d_auto).__name__} == {type(box3d_alias).__name__}")

    print("  ✓ PASSED")
    return True


def test_numpy_list_conversion():
    """Test that Python lists work (converted to NumPy internally)."""
    print("\n--- Test 14: Python lists (NumPy conversion) ---")

    # Lists should be converted to NumPy arrays
    box = samurai_core.Box([0., 0.], [1., 1.])
    print(f"   box = Box([0., 0.], [1., 1.])")
    print(f"   Result: {box}")
    print(f"   Type: {type(box).__name__}")

    assert type(box).__name__ == "Box2D_double"
    assert np.allclose(box.min, [0., 0.])

    print("  ✓ PASSED")
    return True


def test_backward_compatibility():
    """Test that old Box2D code still works."""
    print("\n--- Test 15: Backward compatibility ---")

    # Old API
    box_old = samurai_core.Box2D(np.array([0., 0.]), np.array([1., 1.]))

    # New API
    box_new = samurai_core.Box(np.array([0., 0.]), np.array([1., 1.]))

    # Should be identical
    assert type(box_old) == type(box_new), "Old and new API should produce same type"
    assert np.allclose(box_old.min, box_new.min)
    assert np.allclose(box_old.max, box_new.max)

    print(f"   Old API: {box_old}")
    print(f"   New API: {box_new}")
    print(f"   Are equal: {box_old == box_new}")

    print("  ✓ PASSED")
    return True


def run_all_tests():
    """Run all tests."""
    tests = [
        test_1d_auto_detection,
        test_2d_auto_detection,
        test_3d_auto_detection,
        test_default_dtype,
        test_float_dtype,
        test_dtype_aliases,
        test_all_dimensions_with_auto_detection,
        test_all_dimensions_with_float,
        test_error_mismatched_dimensions,
        test_error_invalid_dimension,
        test_error_invalid_dtype,
        test_operations_with_auto_detected_boxes,
        test_aliases_vs_auto_detection,
        test_numpy_list_conversion,
        test_backward_compatibility,
    ]

    global passed, failed

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except AssertionError as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    if success:
        print("✓ All auto-detection tests passed!")
    else:
        print("✗ Some tests failed")

    sys.exit(0 if success else 1)
