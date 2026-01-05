#!/usr/bin/env python3
"""
Comprehensive test suite for Samurai Field Python Bindings

Tests:
1. Mesh creation (1D and 2D)
2. ScalarField creation
3. Field operations (fill, get, set)
4. Array access (numpy integration)
5. Cell iteration
6. Cell properties
"""

import sys
import numpy as np

sys.path.insert(0, 'build/python')

import samurai_fields

print("=" * 60)
print("Samurai Field Bindings - Test Suite")
print("=" * 60)

passed = 0
failed = 0


def test_1d_mesh_creation():
    """Test 1D mesh creation."""
    print("\n--- Test 1: 1D Mesh Creation ---")

    mesh = samurai_fields.make_uniform_mesh_1d(
        np.array([0.]), np.array([1.]), level=4
    )
    print(f"   1D mesh created with level 4")
    print(f"   Mesh has {mesh.nb_cells()} cells")

    assert mesh.nb_cells() > 0, "Mesh should have cells"
    print("  ✓ PASSED")
    return True


def test_2d_mesh_creation():
    """Test 2D mesh creation."""
    print("\n--- Test 2: 2D Mesh Creation ---")

    mesh = samurai_fields.make_uniform_mesh_2d(
        np.array([0., 0.]), np.array([1., 1.]), level=3
    )
    print(f"   2D mesh created with level 3")
    print(f"   Mesh has {mesh.nb_cells()} cells")

    assert mesh.nb_cells() > 0, "Mesh should have cells"
    print("  ✓ PASSED")
    return True


def test_1d_scalar_field_creation():
    """Test 1D ScalarField creation."""
    print("\n--- Test 3: 1D ScalarField Creation ---")

    mesh = samurai_fields.make_uniform_mesh_1d(
        np.array([0.]), np.array([1.]), level=4
    )
    field = samurai_fields.ScalarField1D(mesh, "u")

    print(f"   Field: {field}")
    print(f"   Name: {field.name}")
    print(f"   Size: {field.size()}")

    assert field.name == "u"
    assert field.size() == mesh.nb_cells()
    print("  ✓ PASSED")
    return True


def test_2d_scalar_field_creation():
    """Test 2D ScalarField creation."""
    print("\n--- Test 4: 2D ScalarField Creation ---")

    mesh = samurai_fields.make_uniform_mesh_2d(
        np.array([0., 0.]), np.array([1., 1.]), level=3
    )
    field = samurai_fields.ScalarField2D(mesh, "v")

    print(f"   Field: {field}")
    print(f"   Name: {field.name}")
    print(f"   Size: {field.size()}")

    assert field.name == "v"
    assert field.size() == mesh.nb_cells()
    print("  ✓ PASSED")
    return True


def test_field_fill():
    """Test field fill operation."""
    print("\n--- Test 5: Field Fill ---")

    mesh = samurai_fields.make_uniform_mesh_2d(
        np.array([0., 0.]), np.array([1., 1.]), level=2
    )
    field = samurai_fields.ScalarField2D(mesh, "u")

    field.fill(3.14)
    arr = field.array()

    print(f"   Filled field with 3.14")
    print(f"   First value: {arr[0]}")
    print(f"   Last value: {arr[-1]}")

    assert np.allclose(arr, 3.14), "All values should be 3.14"
    print("  ✓ PASSED")
    return True


def test_field_get_set_by_index():
    """Test field get/set by linear index."""
    print("\n--- Test 6: Field Get/Set by Index ---")

    mesh = samurai_fields.make_uniform_mesh_2d(
        np.array([0., 0.]), np.array([1., 1.]), level=2
    )
    field = samurai_fields.ScalarField2D(mesh, "u")
    field.fill(0.0)

    # Set some values
    field.set_index(0, 1.5)
    field.set_index(5, 2.7)
    field.set_index(10, -0.5)

    print(f"   Set index 0 to 1.5")
    print(f"   Set index 5 to 2.7")
    print(f"   Set index 10 to -0.5")

    assert field.get_index(0) == 1.5
    assert field.get_index(5) == 2.7
    assert field.get_index(10) == -0.5

    print(f"   Retrieved: {field.get_index(0)}, {field.get_index(5)}, {field.get_index(10)}")
    print("  ✓ PASSED")
    return True


def test_field_get_set_by_cell():
    """Test field get/set by cell."""
    print("\n--- Test 7: Field Get/Set by Cell ---")

    mesh = samurai_fields.make_uniform_mesh_2d(
        np.array([0., 0.]), np.array([1., 1.]), level=2
    )
    field = samurai_fields.ScalarField2D(mesh, "u")
    field.fill(0.0)

    cells = field.cells()
    if len(cells) > 0:
        cell = cells[0]
        print(f"   Cell: {cell}")
        print(f"   Cell center: {cell.center}")
        print(f"   Cell indices: {cell.indices}")

        field.set(cell, 5.0)
        value = field.get(cell)

        print(f"   Set cell value to 5.0")
        print(f"   Retrieved value: {value}")

        assert value == 5.0
    else:
        print("   Warning: No cells in field")

    print("  ✓ PASSED")
    return True


def test_field_numpy_integration():
    """Test numpy array integration."""
    print("\n--- Test 8: NumPy Integration ---")

    mesh = samurai_fields.make_uniform_mesh_2d(
        np.array([0., 0.]), np.array([1., 1.]), level=2
    )
    field = samurai_fields.ScalarField2D(mesh, "u")

    # Create numpy array
    data = np.linspace(0, 1, field.size())
    field.set_array(data)

    print(f"   Set field data from numpy array")
    print(f"   Input array shape: {data.shape}")

    # Get array back
    result = field.array()
    print(f"   Output array shape: {result.shape}")

    assert np.allclose(data, result), "Arrays should match"
    print(f"   Arrays match: {np.allclose(data, result)}")

    # Test operations
    result_doubled = result * 2.0
    print(f"   Doubled first value: {result_doubled[0]}")
    assert np.isclose(result_doubled[0], data[0] * 2)

    print("  ✓ PASSED")
    return True


def test_cell_properties():
    """Test cell properties."""
    print("\n--- Test 9: Cell Properties ---")

    mesh = samurai_fields.make_uniform_mesh_2d(
        np.array([0., 0.]), np.array([1., 1.]), level=2
    )
    field = samurai_fields.ScalarField2D(mesh, "u")

    cells = field.cells()
    if len(cells) > 0:
        cell = cells[0]

        print(f"   Cell level: {cell.level}")
        print(f"   Cell index: {cell.index}")
        print(f"   Cell indices: {cell.indices}")
        print(f"   Cell center: {cell.center}")
        print(f"   Cell corner: {cell.corner}")

        assert cell.index == 0
        assert len(cell.center) == 2
        assert len(cell.corner) == 2
        assert len(cell.indices) == 2

        # Check coordinates are in [0, 1]
        for coord in cell.center:
            assert 0 <= coord <= 1, f"Center coordinate {coord} not in [0, 1]"
    else:
        print("   Warning: No cells in field")

    print("  ✓ PASSED")
    return True


def test_cell_iteration():
    """Test iteration over cells."""
    print("\n--- Test 10: Cell Iteration ---")

    mesh = samurai_fields.make_uniform_mesh_2d(
        np.array([0., 0.]), np.array([1., 1.]), level=2
    )
    field = samurai_fields.ScalarField2D(mesh, "u")
    field.fill(1.0)

    cells = field.cells()
    print(f"   Number of cells: {len(cells)}")

    # Set values using iteration
    for i, cell in enumerate(cells[:5]):  # Just first 5 for speed
        field.set(cell, float(i) * 0.5)

    # Check values
    for i, cell in enumerate(cells[:5]):
        value = field.get(cell)
        expected = float(i) * 0.5
        print(f"   Cell {i}: value = {value}, expected = {expected}")
        assert np.isclose(value, expected)

    print("  ✓ PASSED")
    return True


def test_factory_function():
    """Test factory function for scalar field."""
    print("\n--- Test 11: Factory Function ---")

    mesh = samurai_fields.make_uniform_mesh_2d(
        np.array([0., 0.]), np.array([1., 1.]), level=2
    )

    # Create with factory function
    field = samurai_fields.make_scalar_field(mesh, "factory_field", 2.5)

    print(f"   Factory created field: {field}")
    print(f"   Name: {field.name}")
    print(f"   Initial value: {field.array()[0]}")

    assert field.name == "factory_field"
    assert np.isclose(field.array()[0], 2.5)

    print("  ✓ PASSED")
    return True


def test_field_name_property():
    """Test field name property."""
    print("\n--- Test 12: Field Name Property ---")

    mesh = samurai_fields.make_uniform_mesh_2d(
        np.array([0., 0.]), np.array([1., 1.]), level=2
    )
    field = samurai_fields.ScalarField2D(mesh, "original_name")

    print(f"   Original name: {field.name}")

    field.name = "new_name"
    print(f"   Changed name: {field.name}")

    assert field.name == "new_name"

    print("  ✓ PASSED")
    return True


def run_all_tests():
    """Run all tests."""
    tests = [
        test_1d_mesh_creation,
        test_2d_mesh_creation,
        test_1d_scalar_field_creation,
        test_2d_scalar_field_creation,
        test_field_fill,
        test_field_get_set_by_index,
        test_field_get_set_by_cell,
        test_field_numpy_integration,
        test_cell_properties,
        test_cell_iteration,
        test_factory_function,
        test_field_name_property,
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
        print("✓ All field tests passed!")
    else:
        print("✗ Some tests failed")

    sys.exit(0 if success else 1)
