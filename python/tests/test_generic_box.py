#!/usr/bin/env python3
"""
Test script for generic Samurai Box Python bindings

Tests the factory function and multiple dimensions/types.
"""

import sys
import numpy as np

sys.path.insert(0, 'build/python')

try:
    import samurai_core
    print("=== samurai_core module imported successfully ===")
    print(f"Module version: {samurai_core.__version__}")
    print()
except ImportError as e:
    print(f"Failed to import samurai_core: {e}")
    sys.exit(1)


def test_factory_function():
    """Test Box factory function."""
    print("--- Test 1: Factory Function ---")

    # 2D double precision
    box2d = samurai_core.Box('double', 2, np.array([0., 0.]), np.array([1., 1.]))
    print(f"2D double: {box2d}")
    assert np.allclose(box2d.min, [0., 0.])

    # 3D double precision
    box3d = samurai_core.Box('double', 3, np.array([0., 0., 0.]), np.array([1., 1., 1.]))
    print(f"3D double: {box3d}")
    assert np.allclose(box3d.min, [0., 0., 0.])
    assert box3d.length().shape[0] == 3

    # 2D float precision
    box2d_float = samurai_core.Box('float', 2, np.array([0., 0.]), np.array([1., 1.]))
    print(f"2D float: {box2d_float}")

    # Using aliases
    box_f64 = samurai_core.Box('float64', 2, np.array([0., 0.]), np.array([1., 1.]))
    box_f32 = samurai_core.Box('float32', 2, np.array([0., 0.]), np.array([1., 1.]))
    print(f"float64 alias: {box_f64}")
    print(f"float32 alias: {box_f32}")

    print("  PASSED")


def test_specific_classes():
    """Test specific Box classes."""
    print("\n--- Test 2: Specific Classes ---")

    # Box1D
    box1d = samurai_core.Box1D_double(np.array([0.]), np.array([1.]))
    print(f"Box1D: {box1d}")
    assert box1d.length().shape[0] == 1

    # Box2D
    box2d = samurai_core.Box2D_double(np.array([0., 0.]), np.array([1., 1.]))
    print(f"Box2D: {box2d}")
    assert box2d.length().shape[0] == 2

    # Box3D
    box3d = samurai_core.Box3D_double(np.array([0., 0., 0.]), np.array([1., 1., 1.]))
    print(f"Box3D: {box3d}")
    assert box3d.length().shape[0] == 3

    # Float variants
    box2d_f = samurai_core.Box2D_float(np.array([0., 0.]), np.array([1., 1.]))
    print(f"Box2D_float: {box2d_f}")

    print("  PASSED")


def test_backwards_compatibility():
    """Test backwards compatibility (Box2D alias)."""
    print("\n--- Test 3: Backwards Compatibility ---")

    # Old Box2D should still work
    box = samurai_core.Box2D(np.array([0., 0.]), np.array([1., 1.]))
    print(f"Box2D alias: {box}")
    assert np.allclose(box.min, [0., 0.])

    print("  PASSED")


def test_3d_operations():
    """Test 3D box operations."""
    print("\n--- Test 4: 3D Operations ---")

    box1 = samurai_core.Box('double', 3,
                            np.array([0., 0., 0.]),
                            np.array([2., 2., 2.]))
    box2 = samurai_core.Box('double', 3,
                            np.array([1., 1., 1.]),
                            np.array([3., 3., 3.]))

    print(f"box1: {box1}")
    print(f"box2: {box2}")

    # Intersection
    assert box1.intersects(box2)
    inter = box1.intersection(box2)
    print(f"intersection: {inter}")
    assert np.allclose(inter.min, [1., 1., 1.])
    assert np.allclose(inter.max, [2., 2., 2.])

    # Length
    length = box1.length()
    print(f"length: {length}")
    assert np.allclose(length, [2., 2., 2.])

    # Scaling
    scaled = box1 * 2.0
    assert np.allclose(scaled.max, [4., 4., 4.])
    print(f"scaled * 2: {scaled}")

    print("  PASSED")


def test_1d_operations():
    """Test 1D box operations."""
    print("\n--- Test 5: 1D Operations ---")

    box1 = samurai_core.Box1D_double(np.array([0.]), np.array([2.]))
    box2 = samurai_core.Box1D_double(np.array([1.]), np.array([3.]))

    print(f"box1: {box1}")
    print(f"box2: {box2}")

    # Intersection
    assert box1.intersects(box2)
    inter = box1.intersection(box2)
    print(f"intersection: {inter}")
    assert np.allclose(inter.min, [1.])
    assert np.allclose(inter.max, [2.])

    # Non-intersecting
    box3 = samurai_core.Box1D_double(np.array([10.]), np.array([12.]))
    assert not box1.intersects(box3)

    print("  PASSED")


def test_float_precision():
    """Test float precision boxes."""
    print("\n--- Test 6: Float Precision ---")

    # Float boxes should work the same
    box = samurai_core.Box2D_float(np.array([0., 0.]), np.array([1., 1.]))
    print(f"Float box: {box}")
    assert np.allclose(box.min, [0., 0.])
    assert box.is_valid()

    # Scaling with float (Python doesn't have float literals like C++)
    scaled = box * 2.0
    assert np.allclose(scaled.max, [2., 2.])
    print(f"Scaled float box: {scaled}")

    print("  PASSED")


def run_all_tests():
    """Run all tests."""
    print("=" * 50)
    print("Samurai Generic Box Bindings - Test Suite")
    print("=" * 50)

    tests = [
        test_factory_function,
        test_specific_classes,
        test_backwards_compatibility,
        test_3d_operations,
        test_1d_operations,
        test_float_precision,
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
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 50)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
