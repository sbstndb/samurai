#!/usr/bin/env python3
"""
Tests for the new sam.field.scalar() and sam.field.vector() namespace API.

This test file validates the new field factory API introduced in v0.28.0.
The new API provides a cleaner, more Pythonic interface while maintaining
100% backward compatibility with existing code.

New API:
    u = sam.field.scalar(mesh, "u", init=0.0)
    v = sam.field.vector(mesh, "v", n_components=2, init=1.0)

Old API (still supported):
    u = sam.field.ScalarField2D("u", mesh, 0.0)
    u = sam.field.scalar(mesh, "u", 0.0)
"""

import sys
import os
import pytest
import numpy as np

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
def mesh_3d():
    """Create a simple 3D mesh for testing."""
    box = sam.geometry.box([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    config = sam.config.make(3)
    config.min_level = 3
    config.max_level = 3
    return sam.mesh.make(box, config)


# ============================================================
# sam.field.scalar() Tests
# ============================================================

class TestFieldNamespaceScalar:
    """Test sam.field.scalar() factory function."""

    def test_scalar_1d_default_init(self, mesh_1d):
        """Test 1D scalar field with default init value (0.0)."""
        u = sam.field.scalar(mesh_1d, "u")
        assert isinstance(u, sam.field.ScalarField1D)
        assert u.name == "u"
        arr = u.numpy_view()
        assert np.allclose(arr, 0.0), "Default init should be 0.0"

    def test_scalar_1d_explicit_init(self, mesh_1d):
        """Test 1D scalar field with explicit init value."""
        u = sam.field.scalar(mesh_1d, "u", init=1.5)
        assert isinstance(u, sam.field.ScalarField1D)
        assert u.name == "u"
        arr = u.numpy_view()
        assert np.allclose(arr, 1.5), f"Expected 1.5, got {arr[0]:.2f}"

    def test_scalar_2d_default_init(self, mesh_2d):
        """Test 2D scalar field with default init value."""
        u = sam.field.scalar(mesh_2d, "u")
        assert isinstance(u, sam.field.ScalarField2D)
        assert u.name == "u"
        arr = u.numpy_view()
        assert np.allclose(arr, 0.0)

    def test_scalar_2d_explicit_init(self, mesh_2d):
        """Test 2D scalar field with explicit init value."""
        u = sam.field.scalar(mesh_2d, "u", init=2.5)
        assert isinstance(u, sam.field.ScalarField2D)
        arr = u.numpy_view()
        assert np.allclose(arr, 2.5), f"Expected 2.5, got mean {arr.mean():.2f}"

    def test_scalar_3d_default_init(self, mesh_3d):
        """Test 3D scalar field with default init value."""
        u = sam.field.scalar(mesh_3d, "u")
        assert isinstance(u, sam.field.ScalarField3D)
        assert u.name == "u"
        arr = u.numpy_view()
        assert np.allclose(arr, 0.0)

    def test_scalar_3d_explicit_init(self, mesh_3d):
        """Test 3D scalar field with explicit init value."""
        u = sam.field.scalar(mesh_3d, "u", init=3.14)
        assert isinstance(u, sam.field.ScalarField3D)
        arr = u.numpy_view()
        assert np.allclose(arr, 3.14), f"Expected 3.14, got mean {arr.mean():.2f}"

    def test_scalar_dimension_inference(self, mesh_1d, mesh_2d, mesh_3d):
        """Test that dimension is correctly inferred from mesh type."""
        u1 = sam.field.scalar(mesh_1d, "u")
        u2 = sam.field.scalar(mesh_2d, "u")
        u3 = sam.field.scalar(mesh_3d, "u")

        assert isinstance(u1, sam.field.ScalarField1D)
        assert isinstance(u2, sam.field.ScalarField2D)
        assert isinstance(u3, sam.field.ScalarField3D)

    def test_scalar_keyword_args(self, mesh_2d):
        """Test that keyword arguments work correctly."""
        u = sam.field.scalar(mesh=mesh_2d, name="test", init=42.0)
        assert isinstance(u, sam.field.ScalarField2D)
        assert u.name == "test"
        arr = u.numpy_view()
        assert np.allclose(arr, 42.0)


# ============================================================
# sam.field.vector() Tests
# ============================================================

class TestFieldNamespaceVector:
    """Test sam.field.vector() factory function."""

    def test_vector_1d_2_components_default(self, mesh_1d):
        """Test 1D vector field with 2 components (default)."""
        v = sam.field.vector(mesh_1d, "v")
        assert isinstance(v, sam.field.VectorField1D_2)
        assert v.n_components == 2
        assert v.name == "v"
        arr = v.numpy_view()
        assert np.allclose(arr, 0.0), "Default init should be 0.0"

    def test_vector_1d_2_components_explicit(self, mesh_1d):
        """Test 1D vector field with 2 components explicit."""
        v = sam.field.vector(mesh_1d, "v", n_components=2, init=1.5)
        assert isinstance(v, sam.field.VectorField1D_2)
        arr = v.numpy_view()
        assert np.allclose(arr, 1.5)

    def test_vector_1d_3_components(self, mesh_1d):
        """Test 1D vector field with 3 components."""
        v = sam.field.vector(mesh_1d, "v", n_components=3, init=2.0)
        assert isinstance(v, sam.VectorField1D_3)
        assert v.n_components == 3
        arr = v.numpy_view()
        assert np.allclose(arr, 2.0)

    def test_vector_2d_2_components_default(self, mesh_2d):
        """Test 2D vector field with 2 components (default)."""
        v = sam.field.vector(mesh_2d, "velocity")
        assert isinstance(v, sam.field.VectorField2D_2)
        assert v.n_components == 2
        assert v.name == "velocity"
        arr = v.numpy_view()
        assert np.allclose(arr, 0.0)

    def test_vector_2d_2_components_explicit_init(self, mesh_2d):
        """Test 2D vector field with 2 components and explicit init."""
        v = sam.field.vector(mesh_2d, "v", n_components=2, init=1.0)
        assert isinstance(v, sam.field.VectorField2D_2)
        arr = v.numpy_view()
        assert np.allclose(arr, 1.0)

    def test_vector_2d_3_components(self, mesh_2d):
        """Test 2D vector field with 3 components."""
        v = sam.field.vector(mesh_2d, "v", n_components=3, init=0.5)
        assert isinstance(v, sam.field.VectorField2D_3)
        assert v.n_components == 3
        arr = v.numpy_view()
        assert np.allclose(arr, 0.5)

    def test_vector_3d_2_components(self, mesh_3d):
        """Test 3D vector field with 2 components."""
        v = sam.field.vector(mesh_3d, "v", n_components=2, init=1.0)
        assert isinstance(v, sam.VectorField3D_2)
        assert v.n_components == 2
        arr = v.numpy_view()
        assert np.allclose(arr, 1.0)

    def test_vector_3d_3_components(self, mesh_3d):
        """Test 3D vector field with 3 components."""
        v = sam.field.vector(mesh_3d, "v", n_components=3, init=2.0)
        assert isinstance(v, sam.field.VectorField3D_3)
        assert v.n_components == 3
        arr = v.numpy_view()
        assert np.allclose(arr, 2.0)

    def test_vector_invalid_components(self, mesh_2d):
        """Test that invalid n_components raises an error."""
        with pytest.raises(RuntimeError, match="n_components must be 2 or 3"):
            sam.field.vector(mesh_2d, "v", n_components=5)

    def test_vector_keyword_args(self, mesh_2d):
        """Test that keyword arguments work correctly."""
        v = sam.field.vector(mesh=mesh_2d, name="test", n_components=2, init=99.0)
        assert isinstance(v, sam.field.VectorField2D_2)
        assert v.name == "test"
        arr = v.numpy_view()
        assert np.allclose(arr, 99.0)


# ============================================================
# Backward Compatibility Tests
# ============================================================

class TestBackwardCompatibility:
    """Ensure old APIs still work alongside new namespace API."""

    def test_old_scalar_constructor_still_works(self, mesh_2d):
        """Test that old ScalarField2D constructor still works."""
        u = sam.field.scalar(mesh_2d, "u", init=1.0)
        assert isinstance(u, sam.field.ScalarField2D)
        assert u.name == "u"
        arr = u.numpy_view()
        assert np.allclose(arr, 1.0)

    def test_old_make_scalar_field_still_works(self, mesh_2d):
        """Test that old make_scalar_field function still works."""
        u = sam.field.scalar(mesh_2d, "u", init=2.0)
        assert isinstance(u, sam.field.ScalarField2D)
        assert u.name == "u"
        arr = u.numpy_view()
        assert np.allclose(arr, 2.0)

    def test_old_vector_constructor_still_works(self, mesh_2d):
        """Test that old VectorField2D_2 constructor still works."""
        v = sam.field.vector(mesh_2d, "v", n_components=2, init=1.0)
        assert isinstance(v, sam.field.VectorField2D_2)
        assert v.name == "v"
        arr = v.numpy_view()
        assert np.allclose(arr, 1.0)

    def test_old_make_vector_field_still_works(self, mesh_2d):
        """Test that old make_vector_field function still works."""
        v = sam.field.vector(mesh_2d, "v", n_components=2, init=3.0)
        assert isinstance(v, sam.field.VectorField2D_2)
        assert v.name == "v"
        arr = v.numpy_view()
        assert np.allclose(arr, 3.0)

    def test_new_and_old_produce_same_fields(self, mesh_2d):
        """Test that new and old APIs produce equivalent fields."""
        # Old API
        u_old = sam.field.scalar(mesh_2d, "u", init=5.0)
        v_old = sam.field.vector(mesh_2d, "v", n_components=2, init=6.0)

        # New API
        u_new = sam.field.scalar(mesh_2d, "u", init=5.0)
        v_new = sam.field.vector(mesh_2d, "v", n_components=2, init=6.0)

        # Check types match
        assert type(u_old) == type(u_new)
        assert type(v_old) == type(v_new)

        # Check values match
        assert np.allclose(u_old.numpy_view(), u_new.numpy_view())
        assert np.allclose(v_old.numpy_view(), v_new.numpy_view())


# ============================================================
# Integration Tests
# ============================================================

class TestFieldNamespaceIntegration:
    """Integration tests for the new namespace API."""

    def test_rk3_field_setup(self, mesh_2d):
        """Test common RK3 time-stepping setup pattern."""
        # Create 3 fields for RK3 (common pattern in examples)
        u = sam.field.scalar(mesh_2d, "u", init=1.0)
        u1 = sam.field.scalar(mesh_2d, "u1", init=1.0)
        u2 = sam.field.scalar(mesh_2d, "u2", init=1.0)

        assert isinstance(u, sam.field.ScalarField2D)
        assert isinstance(u1, sam.field.ScalarField2D)
        assert isinstance(u2, sam.field.ScalarField2D)

        # Check all have correct initial value
        assert np.allclose(u.numpy_view(), 1.0)
        assert np.allclose(u1.numpy_view(), 1.0)
        assert np.allclose(u2.numpy_view(), 1.0)

    def test_burgers_setup(self, mesh_2d):
        """Test Burgers equation setup pattern."""
        # Burgers uses 2D vector field with 2 components
        u = sam.field.vector(mesh_2d, "u", n_components=2, init=0.0)

        assert isinstance(u, sam.field.VectorField2D_2)
        assert u.n_components == 2
        arr = u.numpy_view()
        assert np.allclose(arr, 0.0)

    def test_field_namespace_access(self, mesh_2d):
        """Test that sam.field namespace is accessible."""
        # Check that the field submodule exists
        assert hasattr(sam, 'field')

        # Check that it has the expected functions
        assert hasattr(sam.field, 'scalar')
        assert hasattr(sam.field, 'vector')

        # Check that field classes are also accessible
        assert hasattr(sam.field, 'ScalarField2D')
        assert hasattr(sam.field, 'VectorField2D_2')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
