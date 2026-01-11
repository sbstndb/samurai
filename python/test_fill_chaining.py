#!/usr/bin/env python3
"""
Test script to verify Field.fill() returns self for method chaining.
"""
import sys

sys.path.insert(0, 'python/src')

try:
    import numpy as np

    import samurai_python as sam

    print("Testing Field.fill() method chaining...")

    # Create a simple 2D mesh
    box = sam.geometry.box([0., 0.], [1., 1.])
    config = sam.MeshConfig2D()
    config.min_level = 0
    config.max_level = 2

    mesh = sam.MRMesh2D(box, config)

    # Test 1: ScalarField fill() returns self
    print("\n1. Testing ScalarField.fill() chaining...")
    u = sam.field.scalar(mesh, "u")
    result = u.fill(5.0)
    assert result is u, "ERROR: fill() should return self"
    print("   ✓ ScalarField.fill() returns self")

    # Test 2: ScalarField fill().resize() chaining
    print("\n2. Testing ScalarField.fill().resize() chaining...")
    u2 = sam.field.scalar(mesh, "u2")
    result = u2.fill(3.0).resize()
    assert result is u2, "ERROR: fill().resize() should return self"
    print("   ✓ ScalarField.fill().resize() returns self")

    # Test 3: VectorField fill(value) returns self
    print("\n3. Testing VectorField.fill(value) chaining...")
    vel = sam.field.vector(mesh, "velocity", n_components=2)
    result = vel.fill(2.0)
    assert result is vel, "ERROR: fill(value) should return self"
    print("   ✓ VectorField.fill(value) returns self")

    # Test 4: VectorField fill([values]) returns self
    print("\n4. Testing VectorField.fill([values]) chaining...")
    vel2 = sam.field.vector(mesh, "velocity2", n_components=2)
    result = vel2.fill([1.0, 2.0])
    assert result is vel2, "ERROR: fill([values]) should return self"
    print("   ✓ VectorField.fill([values]) returns self")

    # Test 5: VectorField fill().resize() chaining
    print("\n5. Testing VectorField.fill().resize() chaining...")
    vel3 = sam.field.vector(mesh, "velocity3", n_components=2)
    result = vel3.fill([3.0, 4.0]).resize()
    assert result is vel3, "ERROR: fill([values]).resize() should return self"
    print("   ✓ VectorField.fill([values]).resize() returns self")

    # Test 6: Verify fill() actually works
    print("\n6. Verifying fill() functionality...")
    u_test = sam.field.scalar(mesh, "u_test")
    u_test.fill(7.0)
    arr = u_test.numpy_view()
    assert np.allclose(arr, 7.0), "ERROR: fill() didn't set values correctly"
    print("   ✓ fill() correctly sets field values")

    print("\n✅ All tests passed! Field.fill() returns self for method chaining.")
    sys.exit(0)

except ImportError as e:
    print(f"ERROR: Could not import samurai_python: {e}")
    print("Build the module first with: cmake --build build --target samurai_python")
    sys.exit(1)
except AssertionError as e:
    print(f"\n❌ Test failed: {e}")
    sys.exit(1)
except Exception as e:
    print(f"\n❌ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
