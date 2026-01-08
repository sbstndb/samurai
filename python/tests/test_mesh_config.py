"""
Tests for samurai Python bindings - MeshConfig class

Tests the samurai::mesh_config class bindings for 1D, 2D, and 3D.
"""

import os
import sys

import pytest

# Add the build directory to Python path for development
build_dir = os.path.join(os.path.dirname(__file__), "..", "..", "build", "python")
if os.path.exists(build_dir):
    sys.path.insert(0, build_dir)

try:
    import samurai_python as sam
except ImportError:
    pytest.skip("samurai_python module not built", allow_module_level=True)


class TestMeshConfig1D:
    """Tests for MeshConfig1D class."""

    def test_creation(self):
        """Test creating MeshConfig1D."""
        config = sam.config.make(1)
        assert config.dim == 1

    def test_default_values(self):
        """Test default values."""
        config = sam.config.make(1)
        assert config.min_level == 0
        assert config.max_level == 6
        assert config.start_level == 6

    def test_min_level_property(self):
        """Test min_level property setter/getter."""
        config = sam.config.make(1)
        config.min_level = 2
        assert config.min_level == 2

    def test_max_level_property(self):
        """Test max_level property setter/getter."""
        config = sam.config.make(1)
        config.max_level = 8
        assert config.max_level == 8

    def test_start_level_property(self):
        """Test start_level property setter/getter."""
        config = sam.config.make(1)
        config.start_level = 4
        assert config.start_level == 4

    def test_graduation_width_property(self):
        """Test graduation_width property setter/getter."""
        config = sam.config.make(1)
        config.graduation_width = 1
        assert config.graduation_width == 1

    def test_stencil_configuration(self):
        """Test stencil radius and size."""
        config = sam.config.make(1)
        config.max_stencil_radius = 2
        assert config.max_stencil_radius == 2
        assert config.max_stencil_size == 4

    def test_scaling_factor(self):
        """Test scaling_factor property setter/getter."""
        config = sam.config.make(1)
        config.scaling_factor = 0.5
        assert abs(config.scaling_factor - 0.5) < 1e-10

    def test_approx_box_tol(self):
        """Test approx_box_tol property setter/getter."""
        config = sam.config.make(1)
        config.approx_box_tol = 0.01
        assert abs(config.approx_box_tol - 0.01) < 1e-10

    def test_ghost_width_readonly(self):
        """Test ghost_width is accessible."""
        config = sam.config.make(1)
        ghost = config.ghost_width
        assert ghost >= 0

    def test_periodic_scalar(self):
        """Test setting periodic with scalar value."""
        config = sam.config.make(1)
        config.set_periodic(True)
        assert config.get_periodic(0) == True

        config.set_periodic(False)
        assert config.get_periodic(0) == False

    def test_periodic_per_direction(self):
        """Test setting periodic per direction."""
        config = sam.config.make(1)
        config.set_periodic_per_direction([True])
        assert config.get_periodic(0) == True

    def test_repr(self):
        """Test string representation."""
        config = sam.config.make(1)
        s = repr(config)
        assert "_MeshConfig1D" in s
        assert "min_level=" in s


class TestMeshConfig2D:
    """Tests for MeshConfig2D class."""

    def test_creation(self):
        """Test creating MeshConfig2D."""
        config = sam.config.make(2)
        assert config.dim == 2

    def test_default_values(self):
        """Test default values."""
        config = sam.config.make(2)
        assert config.min_level == 0
        assert config.max_level == 6
        assert config.start_level == 6

    def test_level_properties(self):
        """Test level properties."""
        config = sam.config.make(2)
        config.min_level = 2
        config.max_level = 6
        config.start_level = 4
        assert config.min_level == 2
        assert config.max_level == 6
        assert config.start_level == 4

    def test_graduation_width(self):
        """Test graduation_width property."""
        config = sam.config.make(2)
        config.graduation_width = 2
        assert config.graduation_width == 2

    def test_stencil_configuration(self):
        """Test stencil radius and size."""
        config = sam.config.make(2)
        config.max_stencil_size = 6  # Will set radius to 3
        assert config.max_stencil_radius == 3
        assert config.max_stencil_size == 6

    def test_scaling_factor(self):
        """Test scaling_factor property."""
        config = sam.config.make(2)
        config.scaling_factor = 1.5
        assert abs(config.scaling_factor - 1.5) < 1e-10

    def test_periodic_scalar(self):
        """Test setting periodic with scalar value."""
        config = sam.config.make(2)
        config.set_periodic(True)
        assert config.get_periodic(0) == True
        assert config.get_periodic(1) == True

        config.set_periodic(False)
        assert config.get_periodic(0) == False
        assert config.get_periodic(1) == False

    def test_periodic_per_direction(self):
        """Test setting periodic per direction."""
        config = sam.config.make(2)
        config.set_periodic_per_direction([True, False])
        assert config.get_periodic(0) == True
        assert config.get_periodic(1) == False

    def test_periodic_index_out_of_range(self):
        """Test that out of range index raises error."""
        config = sam.config.make(2)
        with pytest.raises(Exception):  # RuntimeError or similar
            config.get_periodic(2)


class TestMeshConfig3D:
    """Tests for MeshConfig3D class."""

    def test_creation(self):
        """Test creating MeshConfig3D."""
        config = sam.config.make(3)
        assert config.dim == 3

    def test_default_values(self):
        """Test default values."""
        config = sam.config.make(3)
        assert config.min_level == 0
        assert config.max_level == 6
        assert config.start_level == 6

    def test_level_properties(self):
        """Test level properties."""
        config = sam.config.make(3)
        config.min_level = 1
        config.max_level = 7
        config.start_level = 5
        assert config.min_level == 1
        assert config.max_level == 7
        assert config.start_level == 5

    def test_periodic_per_direction(self):
        """Test setting periodic per direction in 3D."""
        config = sam.config.make(3)
        config.set_periodic_per_direction([True, False, True])
        assert config.get_periodic(0) == True
        assert config.get_periodic(1) == False
        assert config.get_periodic(2) == True


class TestMeshConfigStringRepresentation:
    """Tests for string representation."""

    def test_repr_1d(self):
        """Test __repr__ for 1D config."""
        config = sam.config.make(1)
        s = repr(config)
        assert "_MeshConfig1D" in s

    def test_repr_2d(self):
        """Test __repr__ for 2D config."""
        config = sam.config.make(2)
        config.min_level = 2
        config.max_level = 6
        s = repr(config)
        assert "_MeshConfig2D" in s
        assert "min_level=" in s

    def test_repr_3d(self):
        """Test __repr__ for 3D config."""
        config = sam.config.make(3)
        s = repr(config)
        assert "_MeshConfig3D" in s

    def test_str_1d(self):
        """Test __str__ for 1D config."""
        config = sam.config.make(1)
        s = str(config)
        assert "_MeshConfig1D" in s

    def test_str_2d(self):
        """Test __str__ for 2D config."""
        config = sam.config.make(2)
        s = str(config)
        assert "_MeshConfig2D" in s

    def test_str_3d(self):
        """Test __str__ for 3D config."""
        config = sam.config.make(3)
        s = str(config)
        assert "_MeshConfig3D" in s


class TestMeshConfigSubmodule:
    """Tests for config submodule."""

    def test_config_submodule_exists(self):
        """Test that config submodule exists."""
        assert hasattr(sam, 'config')

    def test_make_function_exists(self):
        """Test that make() function is available."""
        cfg = sam.config
        assert hasattr(cfg, 'make')
        assert callable(cfg.make)

    def test_make_function_for_dimensions(self):
        """Test that make() function works for all dimensions."""
        config_1d = sam.config.make(1)
        assert config_1d.dim == 1

        config_2d = sam.config.make(2)
        assert config_2d.dim == 2

        config_3d = sam.config.make(3)
        assert config_3d.dim == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
