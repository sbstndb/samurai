#!/usr/bin/env python3
"""
Comprehensive test suite for Samurai VectorField Python Bindings

Tests:
1. VectorField creation (2 and 3 components)
2. Component access (get/set all components)
3. Single component access
4. Fill operations
5. NumPy integration
6. Cell iteration
7. Factory functions
"""

import sys
import numpy as np

sys.path.insert(0, 'build/python')

try:
    import samurai_vector_fields
    print("=== samurai_vector_fields module imported successfully ===")
    print(f"Module version: {samurai_vector_fields.__version__}")
    print()
except ImportError as e:
    print(f"Failed to import samurai_vector_fields: {e}")
    sys.exit(1)


def test_2d_vector_field_2_creation():
    """Test 2D VectorField with 2 components creation."""
    print("\n--- Test 1: 2D VectorField (2 components) Creation ---")

    mesh = samurai_vector_fields.make_uniform_mesh_2d(
        np.array([0., 0.]), np.array([1., 1.]), level=2
    )
    field = samurai_vector_fields.VectorField2D_2(mesh, "velocity")

    print(f"   Field: {field}")
    print(f"   Name: {field.name}")
    print(f"   Components: {field.n_components()}")
    print(f"   Cells: {field.nb_cells()}")

    assert field.name == "velocity"
    assert field.n_components() == 2
    assert field.nb_cells() == mesh.nb_cells()
    print("  ✓ PASSED")
    return True


def test_2d_vector_field_3_creation():
    """Test 2D VectorField with 3 components creation."""
    print("\n--- Test 2: 2D VectorField (3 components) Creation ---")

    mesh = samurai_vector_fields.make_uniform_mesh_2d(
        np.array([0., 0.]), np.array([1., 1.]), level=2
    )
    field = samurai_vector_fields.VectorField2D_3(mesh, "magnetic_field")

    print(f"   Field: {field}")
    print(f"   Name: {field.name}")
    print(f"   Components: {field.n_components()}")
    print(f"   Cells: {field.nb_cells()}")

    assert field.name == "magnetic_field"
    assert field.n_components() == 3
    assert field.nb_cells() == mesh.nb_cells()
    print("  ✓ PASSED")
    return True


def test_vector_field_fill():
    """Test VectorField fill operation."""
    print("\n--- Test 3: VectorField Fill ---")

    mesh = samurai_vector_fields.make_uniform_mesh_2d(
        np.array([0., 0.]), np.array([1., 1.]), level=2
    )
    field = samurai_vector_fields.VectorField2D_2(mesh, "u")

    field.fill(3.14)
    arr = field.array()

    print(f"   Filled field with 3.14")
    print(f"   Array shape: {arr.shape}")
    print(f"   First cell: {arr[0]}")
    print(f"   Last cell: {arr[-1]}")

    assert np.allclose(arr, 3.14), "All values should be 3.14"
    print("  ✓ PASSED")
    return True


def test_vector_field_fill_component():
    """Test VectorField fill specific component."""
    print("\n--- Test 4: VectorField Fill Component ---")

    mesh = samurai_vector_fields.make_uniform_mesh_2d(
        np.array([0., 0.]), np.array([1., 1.]), level=2
    )
    field = samurai_vector_fields.VectorField2D_2(mesh, "u")

    # Fill all with 0
    field.fill(0.0)

    # Fill component 0 with 1.5
    field.fill_component(0, 1.5)

    # Fill component 1 with 2.7
    field.fill_component(1, 2.7)

    arr = field.array()

    print(f"   Filled component 0 with 1.5")
    print(f"   Filled component 1 with 2.7")
    print(f"   First cell: {arr[0]}")
    print(f"   Last cell: {arr[-1]}")

    assert np.allclose(arr[:, 0], 1.5), "Component 0 should be 1.5"
    assert np.allclose(arr[:, 1], 2.7), "Component 1 should be 2.7"
    print("  ✓ PASSED")
    return True


def test_vector_field_get_set_index():
    """Test VectorField get/set by linear index."""
    print("\n--- Test 5: VectorField Get/Set by Index ---")

    mesh = samurai_vector_fields.make_uniform_mesh_2d(
        np.array([0., 0.]), np.array([1., 1.]), level=2
    )
    field = samurai_vector_fields.VectorField2D_2(mesh, "u")
    field.fill(0.0)

    # Set some cells
    field.set_index(0, [1.0, 2.0])
    field.set_index(5, [3.0, 4.0])
    field.set_index(10, [-1.0, -2.0])

    print(f"   Set index 0 to [1.0, 2.0]")
    print(f"   Set index 5 to [3.0, 4.0]")
    print(f"   Set index 10 to [-1.0, -2.0]")

    # Get values
    vals0 = field.get_index(0)
    vals5 = field.get_index(5)
    vals10 = field.get_index(10)

    print(f"   Retrieved index 0: {vals0}")
    print(f"   Retrieved index 5: {vals5}")
    print(f"   Retrieved index 10: {vals10}")

    assert np.allclose(vals0, [1.0, 2.0])
    assert np.allclose(vals5, [3.0, 4.0])
    assert np.allclose(vals10, [-1.0, -2.0])

    print("  ✓ PASSED")
    return True


def test_vector_field_component_access():
    """Test VectorField single component access."""
    print("\n--- Test 6: VectorField Component Access ---")

    mesh = samurai_vector_fields.make_uniform_mesh_2d(
        np.array([0., 0.]), np.array([1., 1.]), level=2
    )
    field = samurai_vector_fields.VectorField2D_3(mesh, "u")
    field.fill(0.0)

    # Set individual components
    field.set_component(0, 0, 1.5)
    field.set_component(0, 1, 2.7)
    field.set_component(0, 2, 3.9)

    print(f"   Set component 0 at index 0 to 1.5")
    print(f"   Set component 1 at index 0 to 2.7")
    print(f"   Set component 2 at index 0 to 2.9")

    # Get values
    c0 = field.get_component(0, 0)
    c1 = field.get_component(0, 1)
    c2 = field.get_component(0, 2)

    print(f"   Retrieved: [{c0}, {c1}, {c2}]")

    assert np.isclose(c0, 1.5)
    assert np.isclose(c1, 2.7)
    assert np.isclose(c2, 3.9)

    print("  ✓ PASSED")
    return True


def test_vector_field_cell_access():
    """Test VectorField get/set by cell."""
    print("\n--- Test 7: VectorField Cell Access ---")

    mesh = samurai_vector_fields.make_uniform_mesh_2d(
        np.array([0., 0.]), np.array([1., 1.]), level=2
    )
    field = samurai_vector_fields.VectorField2D_2(mesh, "u")
    field.fill(0.0)

    cells = field.cells()
    if len(cells) > 0:
        cell = cells[0]
        print(f"   Cell: {cell}")
        print(f"   Cell center: {cell.center}")

        field.set(cell, [5.0, 7.0])
        values = field.get(cell)

        print(f"   Set cell to [5.0, 7.0]")
        print(f"   Retrieved: {values}")

        assert np.allclose(values, [5.0, 7.0])
    else:
        print("   Warning: No cells in field")

    print("  ✓ PASSED")
    return True


def test_vector_field_numpy_integration():
    """Test VectorField NumPy integration."""
    print("\n--- Test 8: VectorField NumPy Integration ---")

    mesh = samurai_vector_fields.make_uniform_mesh_2d(
        np.array([0., 0.]), np.array([1., 1.]), level=2
    )
    field = samurai_vector_fields.VectorField2D_2(mesh, "u")

    # Create numpy array (n_cells, 2)
    nc = field.nb_cells()
    data = np.random.rand(nc, 2) * 10
    field.set_array(data)

    print(f"   Set field from 2D array")
    print(f"   Input array shape: {data.shape}")

    # Get array back
    result = field.array()
    print(f"   Output array shape: {result.shape}")

    assert np.allclose(data, result), "Arrays should match"
    print(f"   Arrays match: {np.allclose(data, result)}")

    # Test operations
    result_doubled = result * 2.0
    print(f"   Doubled first cell: {result_doubled[0]}")
    assert np.allclose(result_doubled[0], data[0] * 2)

    print("  ✓ PASSED")
    return True


def test_vector_field_1d_array_input():
    """Test VectorField with 1D numpy array input."""
    print("\n--- Test 9: VectorField 1D Array Input ---")

    mesh = samurai_vector_fields.make_uniform_mesh_2d(
        np.array([0., 0.]), np.array([1., 1.]), level=2
    )
    field = samurai_vector_fields.VectorField2D_2(mesh, "u")

    # Create 1D numpy array (flattened)
    nc = field.nb_cells()
    data = np.linspace(0, 1, nc * 2)
    field.set_array(data)

    print(f"   Set field from 1D array")
    print(f"   Input array shape: {data.shape}")

    # Get array back
    result = field.array()
    print(f"   Output array shape: {result.shape}")

    # Check flattened result matches
    result_flat = result.flatten()
    assert np.allclose(data, result_flat), "Flattened arrays should match"

    print("  ✓ PASSED")
    return True


def test_vector_field_cell_iteration():
    """Test VectorField cell iteration."""
    print("\n--- Test 10: VectorField Cell Iteration ---")

    mesh = samurai_vector_fields.make_uniform_mesh_2d(
        np.array([0., 0.]), np.array([1., 1.]), level=2
    )
    field = samurai_vector_fields.VectorField2D_2(mesh, "u")
    field.fill(0.0)

    cells = field.cells()
    print(f"   Number of cells: {len(cells)}")

    # Set values using iteration
    for i, cell in enumerate(cells[:5]):  # Just first 5 for speed
        field.set(cell, [float(i), float(i) * 2])

    # Check values
    for i, cell in enumerate(cells[:5]):
        values = field.get(cell)
        expected = [float(i), float(i) * 2]
        print(f"   Cell {i}: {values}, expected: {expected}")
        assert np.allclose(values, expected)

    print("  ✓ PASSED")
    return True


def test_factory_function():
    """Test VectorField factory function."""
    print("\n--- Test 11: Factory Function ---")

    mesh = samurai_vector_fields.make_uniform_mesh_2d(
        np.array([0., 0.]), np.array([1., 1.]), level=2
    )

    # Create with factory function (2 components)
    field2 = samurai_vector_fields.make_vector_field_2(mesh, "vel", 1.5)

    print(f"   Factory created field (2 comp): {field2}")
    print(f"   Name: {field2.name}")
    print(f"   Components: {field2.n_components()}")

    assert field2.name == "vel"
    assert field2.n_components() == 2
    assert np.isclose(field2.array()[0, 0], 1.5)

    # Create with factory function (3 components)
    field3 = samurai_vector_fields.make_vector_field_3(mesh, "mag", 2.5)

    print(f"   Factory created field (3 comp): {field3}")
    print(f"   Name: {field3.name}")
    print(f"   Components: {field3.n_components()}")

    assert field3.name == "mag"
    assert field3.n_components() == 3
    assert np.isclose(field3.array()[0, 0], 2.5)

    print("  ✓ PASSED")
    return True


def test_vector_field_properties():
    """Test VectorField properties."""
    print("\n--- Test 12: VectorField Properties ---")

    mesh = samurai_vector_fields.make_uniform_mesh_2d(
        np.array([0., 0.]), np.array([1., 1.]), level=2
    )
    field = samurai_vector_fields.VectorField2D_2(mesh, "original_name")

    print(f"   Original name: {field.name}")
    print(f"   Number of components: {field.n_components()}")
    print(f"   Number of cells: {field.nb_cells()}")
    print(f"   Total size: {field.size()}")

    field.name = "new_name"
    print(f"   Changed name: {field.name}")

    assert field.name == "new_name"
    assert field.n_components() == 2
    assert field.nb_cells() == mesh.nb_cells()
    assert field.size() == mesh.nb_cells() * 2

    print("  ✓ PASSED")
    return True


def test_vector_field_magnitude():
    """Test computing magnitude of vector field."""
    print("\n--- Test 13: VectorField Magnitude ---")

    mesh = samurai_vector_fields.make_uniform_mesh_2d(
        np.array([0., 0.]), np.array([1., 1.]), level=2
    )
    field = samurai_vector_fields.VectorField2D_2(mesh, "velocity")

    # Set some known values
    field.fill(0.0)
    field.set_index(0, [3.0, 4.0])  # magnitude = 5
    field.set_index(1, [5.0, 12.0])  # magnitude = 13
    field.set_index(2, [1.0, 0.0])  # magnitude = 1

    arr = field.array()

    # Compute magnitude
    magnitude = np.sqrt(np.sum(arr ** 2, axis=1))

    print(f"   Magnitude at index 0: {magnitude[0]} (expected 5.0)")
    print(f"   Magnitude at index 1: {magnitude[1]} (expected 13.0)")
    print(f"   Magnitude at index 2: {magnitude[2]} (expected 1.0)")

    assert np.isclose(magnitude[0], 5.0)
    assert np.isclose(magnitude[1], 13.0)
    assert np.isclose(magnitude[2], 1.0)

    print("  ✓ PASSED")
    return True


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Samurai VectorField Bindings - Test Suite")
    print("=" * 60)

    tests = [
        test_2d_vector_field_2_creation,
        test_2d_vector_field_3_creation,
        test_vector_field_fill,
        test_vector_field_fill_component,
        test_vector_field_get_set_index,
        test_vector_field_component_access,
        test_vector_field_cell_access,
        test_vector_field_numpy_integration,
        test_vector_field_1d_array_input,
        test_vector_field_cell_iteration,
        test_factory_function,
        test_vector_field_properties,
        test_vector_field_magnitude,
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

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed == 0:
        print("✓ All VectorField tests passed!")
    else:
        print("✗ Some tests failed")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
