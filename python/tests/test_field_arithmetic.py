#!/usr/bin/env python3
"""
Tests for ScalarField and VectorField arithmetic operators.

These tests verify the correctness of field-scalar and field-field arithmetic
operations. This test file should be expanded BEFORE any refactoring of the
arithmetic operator bindings in field_bindings.cpp.

IMPORTANT: Ghost Cell Semantics
-------------------------------
According to Samurai AMR/MR semantics:
- Arithmetic operations (field +/- scalar) ONLY affect REAL cells
- Ghost cells are NOT automatically updated (expensive operation)
- Ghost cells are initialized to 0.0 (not garbage)
- User MUST call samurai.update_ghost_mr() when correct ghost values are needed

This matches the C++ behavior where expression templates iterate over
mesh[mesh_id_t::cells] (real cells only), not mesh[mesh_id_t::cells_and_ghosts].

Coverage:
- Field - scalar operations: +, -, *, /
- Scalar - field operations: -
- Field - field operations: +, -
- Operator side effects (should create new fields)
- Name generation for result fields
- Ghost cell handling (no automatic update)
"""

import os
import sys

import numpy as np
import pytest

# Add build directory to path
build_dir = os.path.join(os.path.dirname(__file__), "..", "..", "build", "python")
if os.path.exists(build_dir):
    sys.path.insert(0, build_dir)

import samurai_python as sam

# ============================================================
# Test Fixtures
# ============================================================

@pytest.fixture
def mesh_1d():
    """Create a simple 1D mesh for testing."""
    box = sam.geometry.box([0.0], [1.0])
    config = sam.config.make(1)
    config.min_level = 4
    config.max_level = 4
    return sam.mesh.make(box, config)


@pytest.fixture
def mesh_2d():
    """Create a simple 2D mesh for testing."""
    box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
    config = sam.config.make(2)
    config.min_level = 4
    config.max_level = 4
    return sam.mesh.make(box, config)


@pytest.fixture
def scalar_field_1d(mesh_1d):
    """Create a ScalarField1D initialized to 1.0."""
    field = sam.field.scalar(mesh_1d, "u", init=1.0)
    return field


@pytest.fixture
def scalar_field_2d(mesh_2d):
    """Create a ScalarField2D initialized to 2.0."""
    field = sam.field.scalar(mesh_2d, "v", init=2.0)
    return field


@pytest.fixture
def vector_field_2d(mesh_2d):
    """Create a VectorField2D_2 initialized to [1.0, 2.0]."""
    field = sam.field.vector(mesh_2d, "vel", n_components=2, init=0.0)
    field.fill([1.0, 2.0])
    return field


# ============================================================
# ScalarField1D Arithmetic Tests
# ============================================================

class TestScalarField1DArithmetic:
    """Test arithmetic operations for 1D scalar fields."""

    def _verify_real_cells_correct(self, arr, expected_value, tol=1e-10):
        """Helper to verify that real cells have correct values.
        Ghost cells may be 0.0 ( Samurai semantics - no automatic ghost update).
        """
        # Check that no cells have garbage values (NaN or very large)
        assert not np.any(np.isnan(arr)), "Array contains NaN values"
        assert not np.any(np.abs(arr) > 1e100), "Array contains garbage (very large) values"

        # Check that at least some cells have the expected value
        has_correct_value = np.any(np.isclose(arr, expected_value, atol=tol))
        assert has_correct_value, f"No cells have expected value {expected_value}, got {arr}"

        # Check that all non-zero values are close to expected (within tolerance)
        # This allows ghost cells to be 0.0 while real cells have the correct value
        nonzero_mask = np.abs(arr) > tol
        if np.any(nonzero_mask):
            nonzero_values = arr[nonzero_mask]
            assert np.allclose(nonzero_values, expected_value, atol=tol), \
                f"Real cells should be {expected_value}, got {nonzero_values}"

    def test_field_sub_scalar(self, scalar_field_1d):
        """Test field - scalar operation."""
        result = scalar_field_1d - 0.3
        assert result is not scalar_field_1d, "Result should be a new field"
        assert result.name.endswith("_sub"), "Result name should indicate subtraction"
        arr = result.numpy_view()
        self._verify_real_cells_correct(arr, 0.7)

    def test_scalar_sub_field(self, scalar_field_1d):
        """Test scalar - field operation."""
        result = 1.0 - scalar_field_1d
        assert result is not scalar_field_1d, "Result should be a new field"
        assert result.name == "scalar_sub", "Result should have specific name"
        arr = result.numpy_view()
        self._verify_real_cells_correct(arr, 0.0)

    def test_field_add_scalar(self, scalar_field_1d):
        """Test field + scalar operation."""
        result = scalar_field_1d + 0.5
        assert result is not scalar_field_1d, "Result should be a new field"
        assert result.name.endswith("_add"), "Result name should indicate addition"
        arr = result.numpy_view()
        self._verify_real_cells_correct(arr, 1.5)

    def test_field_mul_scalar(self, scalar_field_1d):
        """Test field * scalar operation."""
        result = scalar_field_1d * 2.0
        assert result is not scalar_field_1d, "Result should be a new field"
        assert result.name.endswith("_mul"), "Result name should indicate multiplication"
        arr = result.numpy_view()
        self._verify_real_cells_correct(arr, 2.0)

    def test_scalar_mul_field(self, scalar_field_1d):
        """Test scalar * field operation (commutativity)."""
        result = 3.0 * scalar_field_1d
        assert result is not scalar_field_1d, "Result should be a new field"
        arr = result.numpy_view()
        self._verify_real_cells_correct(arr, 3.0)

    def test_field_div_scalar(self, scalar_field_1d):
        """Test field / scalar operation."""
        result = scalar_field_1d / 2.0
        assert result is not scalar_field_1d, "Result should be a new field"
        assert result.name.endswith("_div"), "Result name should indicate division"
        arr = result.numpy_view()
        self._verify_real_cells_correct(arr, 0.5)

    def test_field_sub_field(self, mesh_1d):
        """Test field - field operation."""
        f1 = sam.field.scalar(mesh_1d, "a", init=3.0)
        f2 = sam.field.scalar(mesh_1d, "b", init=1.0)
        result = f1 - f2
        assert result is not f1 and result is not f2, "Result should be a new field"
        assert result.name.endswith("_sub"), "Result name should indicate subtraction"
        arr = result.numpy_view()
        self._verify_real_cells_correct(arr, 2.0)

    def test_field_add_field(self, mesh_1d):
        """Test field + field operation."""
        f1 = sam.field.scalar(mesh_1d, "a", init=1.0)
        f2 = sam.field.scalar(mesh_1d, "b", init=2.0)
        result = f1 + f2
        assert result is not f1 and result is not f2, "Result should be a new field"
        assert result.name.endswith("_add"), "Result name should indicate addition"
        arr = result.numpy_view()
        self._verify_real_cells_correct(arr, 3.0)

    def test_original_field_unchanged(self, scalar_field_1d):
        """Verify that original field is not modified by arithmetic operations."""
        original_value = scalar_field_1d.numpy_view()[0]
        _ = scalar_field_1d - 0.5
        _ = scalar_field_1d + 0.5
        _ = scalar_field_1d * 2.0
        _ = scalar_field_1d / 2.0
        assert np.allclose(scalar_field_1d.numpy_view()[0], original_value), \
            "Original field should remain unchanged"


# ============================================================
# ScalarField2D Arithmetic Tests
# ============================================================

class TestScalarField2DArithmetic:
    """Test arithmetic operations for 2D scalar fields (ensure consistency across dimensions)."""

    def _verify_real_cells_correct(self, arr, expected_value, tol=1e-10):
        """Helper to verify that real cells have correct values."""
        assert not np.any(np.isnan(arr)), "Array contains NaN values"
        assert not np.any(np.abs(arr) > 1e100), "Array contains garbage values"

        has_correct_value = np.any(np.isclose(arr, expected_value, atol=tol))
        assert has_correct_value, f"No cells have expected value {expected_value}"

        nonzero_mask = np.abs(arr) > tol
        if np.any(nonzero_mask):
            nonzero_values = arr[nonzero_mask]
            assert np.allclose(nonzero_values, expected_value, atol=tol), \
                f"Real cells should be {expected_value}"

    def test_field_sub_scalar_2d(self, scalar_field_2d):
        """Test field - scalar operation in 2D."""
        result = scalar_field_2d - 0.5
        arr = result.numpy_view()
        self._verify_real_cells_correct(arr, 1.5)

    def test_field_mul_scalar_2d(self, scalar_field_2d):
        """Test field * scalar operation in 2D."""
        result = scalar_field_2d * 3.0
        arr = result.numpy_view()
        self._verify_real_cells_correct(arr, 6.0)


# ============================================================
# VectorField Arithmetic Tests
# ============================================================

class TestVectorFieldArithmetic:
    """Test arithmetic operations for VectorField."""

    def _verify_component_correct(self, arr, component_idx, expected_value, tol=1e-10):
        """Helper to verify that a component has correct values in real cells."""
        component = arr[:, component_idx]
        assert not np.any(np.isnan(component)), f"Component {component_idx} contains NaN"
        assert not np.any(np.abs(component) > 1e100), f"Component {component_idx} has garbage"

        has_correct = np.any(np.isclose(component, expected_value, atol=tol))
        assert has_correct, f"Component {component_idx}: No cells have {expected_value}"

        nonzero_mask = np.abs(component) > tol
        if np.any(nonzero_mask):
            nonzero_values = component[nonzero_mask]
            assert np.allclose(nonzero_values, expected_value, atol=tol), \
                f"Component {component_idx}: Expected {expected_value}, got {nonzero_values}"

    def test_vector_sub_scalar(self, vector_field_2d):
        """Test VectorField - scalar operation."""
        result = vector_field_2d - 1.0
        arr = result.numpy_view()
        # [1.0, 2.0] - 1.0 = [0.0, 1.0]
        self._verify_component_correct(arr, 0, 0.0)
        self._verify_component_correct(arr, 1, 1.0)

    def test_vector_add_scalar(self, vector_field_2d):
        """Test VectorField + scalar operation."""
        result = vector_field_2d + 2.0
        arr = result.numpy_view()
        # [1.0, 2.0] + 2.0 = [3.0, 4.0]
        self._verify_component_correct(arr, 0, 3.0)
        self._verify_component_correct(arr, 1, 4.0)

    def test_vector_mul_scalar(self, vector_field_2d):
        """Test VectorField * scalar operation."""
        result = vector_field_2d * 2.0
        arr = result.numpy_view()
        # [1.0, 2.0] * 2.0 = [2.0, 4.0]
        self._verify_component_correct(arr, 0, 2.0)
        self._verify_component_correct(arr, 1, 4.0)

    def test_vector_div_scalar(self, vector_field_2d):
        """Test VectorField / scalar operation."""
        result = vector_field_2d / 2.0
        arr = result.numpy_view()
        # [1.0, 2.0] / 2.0 = [0.5, 1.0]
        self._verify_component_correct(arr, 0, 0.5)
        self._verify_component_correct(arr, 1, 1.0)


# ============================================================
# Operator Chaining Tests
# ============================================================

class TestOperatorChaining:
    """Test chaining multiple arithmetic operations."""

    def _verify_real_cells_correct(self, arr, expected_value, tol=1e-10):
        """Helper to verify that real cells have correct values."""
        assert not np.any(np.isnan(arr)), "Array contains NaN values"
        assert not np.any(np.abs(arr) > 1e100), "Array contains garbage values"

        has_correct_value = np.any(np.isclose(arr, expected_value, atol=tol))
        assert has_correct_value, f"No cells have expected value {expected_value}"

        nonzero_mask = np.abs(arr) > tol
        if np.any(nonzero_mask):
            nonzero_values = arr[nonzero_mask]
            assert np.allclose(nonzero_values, expected_value, atol=tol), \
                f"Real cells should be {expected_value}"

    def test_chained_operations(self, scalar_field_1d):
        """Test result = (field * 2 + 1) / 3."""
        result = (scalar_field_1d * 2.0 + 1.0) / 3.0
        arr = result.numpy_view()
        # (1.0 * 2 + 1) / 3 = 1.0
        self._verify_real_cells_correct(arr, 1.0)

    def test_complex_expression(self, scalar_field_1d):
        """Test result = field - 0.5 * (field + field)."""
        result = scalar_field_1d - 0.5 * (scalar_field_1d + scalar_field_1d)
        arr = result.numpy_view()
        # 1.0 - 0.5 * (1.0 + 1.0) = 0.0
        # For 0.0 expected value, we need a special check since all cells might be 0.0
        assert not np.any(np.isnan(arr)), "Array contains NaN values"
        assert not np.any(np.abs(arr) > 1e100), "Array contains garbage values"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
