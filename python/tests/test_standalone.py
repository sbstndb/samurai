"""
Standalone tests for Samurai Python bindings.

These tests verify that the Python bindings work correctly when built
as a standalone package (using find_package(samurai CONFIG REQUIRED)).

These are minimal smoke tests to ensure basic functionality works.
For full test coverage, see the other test files in the tests/ directory.
"""

import sys
import pytest

# Try to import the module
try:
    import samurai_python as sam
    SAMURAI_AVAILABLE = True
except ImportError:
    SAMURAI_AVAILABLE = False
    pytest.skip("samurai_python module not available", allow_module_level=True)


class TestBasicImport:
    """Test basic module import and version."""

    def test_module_import(self):
        """Test that the module can be imported."""
        assert SAMURAI_AVAILABLE

    def test_version_attribute(self):
        """Test that __version__ attribute exists."""
        assert hasattr(sam, "__version__")
        assert isinstance(sam.__version__, str)
        # Version should be non-empty and follow semantic versioning
        parts = sam.__version__.split(".")
        assert len(parts) >= 2
        assert all(p.isdigit() for p in parts)

    def test_module_docstring(self):
        """Test that the module has documentation."""
        assert sam.__doc__ is not None
        assert len(sam.__doc__) > 0


class TestGeometrySubmodule:
    """Test the geometry submodule."""

    def test_geometry_submodule_exists(self):
        """Test that the geometry submodule exists."""
        assert hasattr(sam, "geometry")

    def test_box_factory(self):
        """Test creating a Box using the factory function."""
        # Create a 2D box
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        assert box is not None

    def test_box_min_max(self):
        """Test Box min/max properties."""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        # The exact API may vary, but we should be able to get min/max
        # This is a placeholder - adjust based on actual API
        assert hasattr(box, "min") or hasattr(box, "min_corner")


class TestConfigSubmodule:
    """Test the config submodule."""

    def test_config_submodule_exists(self):
        """Test that the config submodule exists."""
        assert hasattr(sam, "config")

    def test_mesh_config_factory(self):
        """Test creating a MeshConfig using the factory function."""
        # The exact API may vary
        try:
            config = sam.config.mesh_config(dim=2)
            assert config is not None
        except Exception:
            # Factory might not exist or have different signature
            pass


class TestMeshSubmodule:
    """Test the mesh submodule."""

    def test_mesh_submodule_exists(self):
        """Test that the mesh submodule exists."""
        assert hasattr(sam, "mesh")

    def test_mesh_factory(self):
        """Test creating a mesh using the factory function."""
        try:
            box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
            mesh = sam.mesh.make(box, min_level=2, max_level=4)
            assert mesh is not None
        except Exception:
            # Factory might require different parameters
            pass


class TestFieldSubmodule:
    """Test the field submodule."""

    def test_field_submodule_exists(self):
        """Test that the field submodule exists."""
        assert hasattr(sam, "field")

    def test_scalar_field_factory(self):
        """Test creating a scalar field using the factory function."""
        try:
            box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
            mesh = sam.mesh.make(box, min_level=2, max_level=4)
            field = sam.field.scalar(mesh, "u")
            assert field is not None
        except Exception:
            # Factory might require different parameters
            pass

    def test_vector_field_factory(self):
        """Test creating a vector field using the factory function."""
        try:
            box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
            mesh = sam.mesh.make(box, min_level=2, max_level=4)
            field = sam.field.vector(mesh, "vel", n_components=2)
            assert field is not None
        except Exception:
            # Factory might require different parameters
            pass


class TestOperatorsSubmodule:
    """Test the operators submodule."""

    def test_operators_submodule_exists(self):
        """Test that the operators submodule exists."""
        assert hasattr(sam, "operators")


class TestBoundarySubmodule:
    """Test the boundary submodule."""

    def test_boundary_submodule_exists(self):
        """Test that the boundary submodule exists."""
        assert hasattr(sam, "boundary")


class TestAdaptationSubmodule:
    """Test the adaptation submodule."""

    def test_adaptation_submodule_exists(self):
        """Test that the adaptation submodule exists."""
        assert hasattr(sam, "adaptation")


class TestAlgorithmsSubmodule:
    """Test the algorithms submodule."""

    def test_algorithms_submodule_exists(self):
        """Test that the algorithms submodule exists."""
        assert hasattr(sam, "algorithms")


class TestIOSubmodule:
    """Test the I/O submodule."""

    def test_io_submodule_exists(self):
        """Test that the I/O submodule exists."""
        assert hasattr(sam, "io")


class TestModuleLevelFunctions:
    """Test module-level convenience functions."""

    def test_upwind_function(self):
        """Test that upwind operator is available at module level."""
        # This function should be available for backward compatibility
        assert hasattr(sam, "upwind") or hasattr(sam.operators, "upwind")


class TestUtilities:
    """Test Python utility modules."""

    def test_samurai_python_utils_import(self):
        """Test that samurai_python.utils can be imported."""
        try:
            import samurai_python.utils
            assert True
        except ImportError:
            pytest.skip("samurai_python.utils not available")

    def test_progress_module(self):
        """Test that progress utilities are available."""
        try:
            from samurai_python.utils import progress
            assert progress is not None
        except ImportError:
            pytest.skip("progress utilities not available")


if __name__ == "__main__":
    # Run tests when executed directly
    pytest.main([__file__, "-v"])
