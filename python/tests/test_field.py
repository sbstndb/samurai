"""
Tests for samurai Python bindings - ScalarField and VectorField classes

Tests the samurai::ScalarField and samurai::VectorField class bindings.
"""

import os
import sys

import numpy as np
import pytest

# Add the build directory to Python path for development
build_dir = os.path.join(os.path.dirname(__file__), "..", "..", "build", "python")
if os.path.exists(build_dir):
    sys.path.insert(0, build_dir)

try:
    import samurai_python as sam
except ImportError:
    pytest.skip("samurai_python module not built", allow_module_level=True)


class TestScalarField1D:
    """Tests for ScalarField1D class."""

    def test_creation(self):
        """Test creating ScalarField1D from mesh."""
        box = sam.geometry.box([0.], [1.])
        config = sam.config.make(1)
        config.min_level = 0
        config.max_level = 1

        mesh = sam.mesh.make(box, config)
        field = sam.field.scalar(mesh, "u")

        assert field.name == "u"
        assert field.dim == 1
        assert field.mesh is mesh
        assert field.size > 0

    def test_creation_with_init_value(self):
        """Test creating ScalarField with initial value."""
        box = sam.geometry.box([0.], [1.])
        config = sam.config.make(1)
        config.min_level = 0
        config.max_level = 1

        mesh = sam.mesh.make(box, config)
        field = sam.field.scalar(mesh, "u", init=3.14)

        # Check a few cells have the init value
        for i in range(min(5, field.size)):
            assert abs(field[i] - 3.14) < 1e-10

    def test_name_property(self):
        """Test field name getter/setter."""
        box = sam.geometry.box([0.], [1.])
        config = sam.config.make(1)
        config.min_level = 0
        config.max_level = 1

        mesh = sam.mesh.make(box, config)
        field = sam.field.scalar(mesh, "u")

        assert field.name == "u"

        field.name = "v"
        assert field.name == "v"

    def test_mesh_property(self):
        """Test mesh property returns correct mesh."""
        box = sam.geometry.box([0.], [1.])
        config = sam.config.make(1)
        config.min_level = 0
        config.max_level = 1

        mesh = sam.mesh.make(box, config)
        field = sam.field.scalar(mesh, "u")

        assert field.mesh is mesh

    def test_fill(self):
        """Test filling field with constant value."""
        box = sam.geometry.box([0.], [1.])
        config = sam.config.make(1)
        config.min_level = 0
        config.max_level = 1

        mesh = sam.mesh.make(box, config)
        field = sam.field.scalar(mesh, "u")
        field.fill(2.5)

        for i in range(field.size):
            assert abs(field[i] - 2.5) < 1e-10

    def test_numpy_view(self):
        """Test zero-copy NumPy view."""
        box = sam.geometry.box([0.], [1.])
        config = sam.config.make(1)
        config.min_level = 0
        config.max_level = 1

        mesh = sam.mesh.make(box, config)
        field = sam.field.scalar(mesh, "u")
        field.fill(42.0)

        arr = field.numpy_view()

        # Verify it's a NumPy array
        assert isinstance(arr, np.ndarray)

        # Verify shape
        assert arr.shape[0] == field.size

        # Verify values
        assert np.allclose(arr, 42.0)

    def test_numpy_memory_sharing(self):
        """Test that NumPy view shares memory with field."""
        box = sam.geometry.box([0.], [1.])
        config = sam.config.make(1)
        config.min_level = 0
        config.max_level = 1

        mesh = sam.mesh.make(box, config)
        field = sam.field.scalar(mesh, "u")
        field.fill(1.0)

        arr1 = field.numpy_view()
        arr2 = field.numpy_view()

        # Verify memory sharing
        assert np.shares_memory(arr1, arr2)

        # Modify through arr1
        arr1[0] = 99.0

        # Check arr2 sees the change (zero-copy)
        assert abs(arr2[0] - 99.0) < 1e-10

    def test_integer_indexing(self):
        """Test indexing field by integer index."""
        box = sam.geometry.box([0.], [1.])
        config = sam.config.make(1)
        config.min_level = 0
        config.max_level = 1

        mesh = sam.mesh.make(box, config)
        field = sam.field.scalar(mesh, "u")

        field[0] = 123.0
        assert abs(field[0] - 123.0) < 1e-10

        field[5] = 456.0
        assert abs(field[5] - 456.0) < 1e-10

    def test_ghosts_updated_flag(self):
        """Test ghosts_updated property."""
        box = sam.geometry.box([0.], [1.])
        config = sam.config.make(1)
        config.min_level = 0
        config.max_level = 1

        mesh = sam.mesh.make(box, config)
        field = sam.field.scalar(mesh, "u")

        # Initially False
        assert field.ghosts_updated is False

        # Set to True
        field.ghosts_updated = True
        assert field.ghosts_updated is True

    def test_string_representation(self):
        """Test __repr__ and __str__."""
        box = sam.geometry.box([0.], [1.])
        config = sam.config.make(1)
        config.min_level = 0
        config.max_level = 1

        mesh = sam.mesh.make(box, config)
        field = sam.field.scalar(mesh, "u")

        repr_str = repr(field)
        str_str = str(field)

        assert "ScalarField1D" in repr_str
        assert "u" in repr_str
        assert "ScalarField1D" in str_str
        assert "cells" in str_str


class TestScalarField2D:
    """Tests for ScalarField2D class."""

    def test_creation(self):
        """Test creating ScalarField2D from mesh."""
        box = sam.geometry.box([0., 0.], [1., 1.])
        config = sam.config.make(2)
        config.min_level = 0
        config.max_level = 1

        mesh = sam.mesh.make(box, config)
        field = sam.field.scalar(mesh, "u")

        assert field.name == "u"
        assert field.dim == 2
        assert field.mesh is mesh
        assert field.size > 0

    def test_fill(self):
        """Test filling 2D field with constant value."""
        box = sam.geometry.box([0., 0.], [1., 1.])
        config = sam.config.make(2)
        config.min_level = 0
        config.max_level = 1

        mesh = sam.mesh.make(box, config)
        field = sam.field.scalar(mesh, "u")
        field.fill(3.14)

        arr = field.numpy_view()
        assert np.allclose(arr, 3.14)

    def test_numpy_vectorized_operations(self):
        """Test vectorized NumPy operations on field."""
        box = sam.geometry.box([0., 0.], [1., 1.])
        config = sam.config.make(2)
        config.min_level = 0
        config.max_level = 1

        mesh = sam.mesh.make(box, config)
        field = sam.field.scalar(mesh, "u")
        field.fill(1.0)

        arr = field.numpy_view()

        # Vectorized operation
        arr[:] = np.sin(arr * np.pi)

        # Verify field was modified
        assert not np.allclose(arr, 1.0)


class TestVectorField2D_2:  # noqa: N801
    """Tests for VectorField2D_2 class (2 components)."""

    def test_creation(self):
        """Test creating VectorField2D_2 from mesh."""
        box = sam.geometry.box([0., 0.], [1., 1.])
        config = sam.config.make(2)
        config.min_level = 0
        config.max_level = 1

        mesh = sam.mesh.make(box, config)
        field = sam.field.vector(mesh, "vel", 2)

        assert field.name == "vel"
        assert field.dim == 2
        assert field.n_components == 2
        assert field.is_soa is False  # AOS layout
        assert field.mesh is mesh

    def test_fill(self):
        """Test filling vector field with scalar value."""
        box = sam.geometry.box([0., 0.], [1., 1.])
        config = sam.config.make(2)
        config.min_level = 0
        config.max_level = 1

        mesh = sam.mesh.make(box, config)
        field = sam.field.vector(mesh, "vel", 2)
        field.fill(5.0)

        arr = field.numpy_view()
        # AOS layout: (n_cells, 2)
        assert np.allclose(arr, 5.0)

    def test_numpy_view_shape(self):
        """Test NumPy view has correct shape."""
        box = sam.geometry.box([0., 0.], [1., 1.])
        config = sam.config.make(2)
        config.min_level = 0
        config.max_level = 1

        mesh = sam.mesh.make(box, config)
        field = sam.field.vector(mesh, "vel", 2)
        field.fill(0.0)

        arr = field.numpy_view()

        # Should be 2D: (n_cells, n_components) for AOS
        assert arr.ndim == 2
        assert arr.shape[0] == field.size
        assert arr.shape[1] == 2

    def test_get_component(self):
        """Test extracting individual components."""
        box = sam.geometry.box([0., 0.], [1., 1.])
        config = sam.config.make(2)
        config.min_level = 0
        config.max_level = 1

        mesh = sam.mesh.make(box, config)
        field = sam.field.vector(mesh, "vel", 2)
        field.fill([1.0, 2.0])

        comp0 = field.get_component(0)
        comp1 = field.get_component(1)

        # Component 0 should have 1.0
        arr0 = comp0.numpy_view()
        assert np.allclose(arr0, 1.0)

        # Component 1 should have 2.0
        arr1 = comp1.numpy_view()
        assert np.allclose(arr1, 2.0)

    def test_string_representation(self):
        """Test __repr__ and __str__ for VectorField."""
        box = sam.geometry.box([0., 0.], [1., 1.])
        config = sam.config.make(2)
        config.min_level = 0
        config.max_level = 1

        mesh = sam.mesh.make(box, config)
        field = sam.field.vector(mesh, "vel", 2)

        repr_str = repr(field)
        str_str = str(field)

        assert "VectorField2D" in repr_str
        assert "vel" in repr_str
        assert "2 components" in str_str


class TestVectorField2D_3:  # noqa: N801
    """Tests for VectorField2D_3 class (3 components)."""

    def test_creation(self):
        """Test creating VectorField2D_3 from mesh."""
        box = sam.geometry.box([0., 0.], [1., 1.])
        config = sam.config.make(2)
        config.min_level = 0
        config.max_level = 1

        mesh = sam.mesh.make(box, config)
        field = sam.field.vector(mesh, "f", 3)

        assert field.n_components == 3
        assert field.dim == 2


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_make_scalar_field_1d(self):
        """Test make_scalar_field for 1D mesh."""
        box = sam.geometry.box([0.], [1.])
        config = sam.config.make(1)
        config.min_level = 0
        config.max_level = 1

        mesh = sam.mesh.make(box, config)
        field = sam.field.scalar(mesh, "u", 2.5)

        assert field.name == "u"
        # Check init value
        assert abs(field[0] - 2.5) < 1e-10

    def test_make_scalar_field_2d(self):
        """Test make_scalar_field for 2D mesh."""
        box = sam.geometry.box([0., 0.], [1., 1.])
        config = sam.config.make(2)
        config.min_level = 0
        config.max_level = 1

        mesh = sam.mesh.make(box, config)
        field = sam.field.scalar(mesh, "u")

        assert field.name == "u"
        assert field.dim == 2

    def test_make_vector_field_2_components(self):
        """Test make_vector_field with 2 components."""
        box = sam.geometry.box([0., 0.], [1., 1.])
        config = sam.config.make(2)
        config.min_level = 0
        config.max_level = 1

        mesh = sam.mesh.make(box, config)
        field = sam.field.vector(mesh, "vel", 2)

        assert field.n_components == 2

    def test_make_vector_field_3_components(self):
        """Test make_vector_field with 3 components."""
        box = sam.geometry.box([0., 0.], [1., 1.])
        config = sam.config.make(2)
        config.min_level = 0
        config.max_level = 1

        mesh = sam.mesh.make(box, config)
        field = sam.field.vector(mesh, "f", 3)

        assert field.n_components == 3


class TestFieldSubmodule:
    """Tests for field submodule."""

    def test_field_submodule_exists(self):
        """Test that field submodule exists."""
        assert hasattr(sam, 'field')

    def test_scalar_field_in_submodule(self):
        """Test that ScalarField factory is accessible from field submodule."""
        fs = sam.field
        assert hasattr(fs, 'scalar')

    def test_vector_field_in_submodule(self):
        """Test that VectorField factory is accessible from field submodule."""
        fs = sam.field
        assert hasattr(fs, 'vector')

    def test_field_from_submodule(self):
        """Test creating field from submodule."""
        scalar = sam.field.scalar
        box = sam.geometry.box([0.], [1.])
        config = sam.config.make(1)
        config.min_level = 0
        config.max_level = 1

        mesh = sam.mesh.make(box, config)
        field = scalar(mesh, "u")
        assert field.name == "u"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
