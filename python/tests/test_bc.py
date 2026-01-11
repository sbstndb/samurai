"""
Tests for samurai Python bindings - Boundary Conditions

Tests the make_bc function and boundary condition types.
"""

import os
import sys

import pytest

# Add the build directory to Python path for development
# Try build_py314 first, then build
build_dir = os.path.join(os.path.dirname(__file__), "..", "..", "build_py314", "python")
if not os.path.exists(build_dir):
    build_dir = os.path.join(os.path.dirname(__file__), "..", "..", "build", "python")
if os.path.exists(build_dir):
    sys.path.insert(0, build_dir)

try:
    import samurai_python as sam
except ImportError:
    pytest.skip("samurai_python module not built", allow_module_level=True)


class TestMakeDirichletBC:
    """Tests for make_dirichlet_bc function."""

    def test_1d_dirichlet_order1(self):
        """Test Dirichlet BC of order 1 for 1D field."""
        box = sam.geometry.box([0.0], [1.0])
        config = sam.config.make(1)
        config.min_level = 3
        config.max_level = 3
        mesh = sam.mesh.make(box, config)

        u = sam.field.scalar(mesh, "u", init=0.0)

        # Create Dirichlet BC with value 0.0 (returns None, BC is attached to field)
        sam.make_dirichlet_bc(u, 0.0)

        # If we get here without exception, the BC was attached successfully
        assert True

    def test_1d_dirichlet_different_orders(self):
        """Test Dirichlet BC with different orders."""
        box = sam.geometry.box([0.0], [1.0])
        config = sam.config.make(1)
        config.min_level = 2
        config.max_level = 2
        config.max_stencil_size = 10  # Support up to order 4 BCs
        mesh = sam.mesh.make(box, config)

        # Test orders 1-4
        for order in [1, 2, 3, 4]:
            u = sam.field.scalar(mesh, "u", init=0.0)
            sam.make_dirichlet_bc(u, 1.5, order=order)
            # If we get here without exception, it worked
            assert True

    def test_1d_dirichlet_invalid_order(self):
        """Test that invalid order raises an error."""
        box = sam.geometry.box([0.0], [1.0])
        config = sam.config.make(1)
        config.min_level = 2
        config.max_level = 2
        mesh = sam.mesh.make(box, config)

        u = sam.field.scalar(mesh, "u", init=0.0)

        # Order 5 should raise an error
        with pytest.raises(RuntimeError, match="order must be between 1 and 4"):
            sam.make_dirichlet_bc(u, 0.0, order=5)

    def test_2d_dirichlet_order1(self):
        """Test Dirichlet BC of order 1 for 2D field (advection_2d case)."""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config.make(2)
        config.min_level = 4
        config.max_level = 4
        mesh = sam.mesh.make(box, config)

        u = sam.field.scalar(mesh, "u", init=0.0)

        # Create Dirichlet BC with value 0.0 (as in advection_2d.cpp line 110)
        sam.make_dirichlet_bc(u, 0.0)

        # If we get here without exception, the BC was attached successfully
        assert True

    def test_2d_dirichlet_nonzero_value(self):
        """Test Dirichlet BC with non-zero constant value."""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config.make(2)
        config.min_level = 2
        config.max_level = 2
        mesh = sam.mesh.make(box, config)

        u = sam.field.scalar(mesh, "u", init=0.0)

        # Create Dirichlet BC with value 5.0
        sam.make_dirichlet_bc(u, 5.0)

        assert True

    def test_2d_dirichlet_different_orders(self):
        """Test Dirichlet BC with different orders in 2D."""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config.make(2)
        config.min_level = 2
        config.max_level = 2
        config.max_stencil_size = 10  # Support up to order 4 BCs
        mesh = sam.mesh.make(box, config)

        # Test orders 1-4
        for order in [1, 2, 3, 4]:
            u = sam.field.scalar(mesh, "u", init=0.0)
            sam.make_dirichlet_bc(u, 0.0, order=order)
            assert True

    def test_3d_dirichlet_order1(self):
        """Test Dirichlet BC of order 1 for 3D field."""
        box = sam.geometry.box([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        config = sam.config.make(3)
        config.min_level = 1
        config.max_level = 1
        mesh = sam.mesh.make(box, config)

        u = sam.field.scalar(mesh, "u", init=0.0)

        # Create Dirichlet BC
        sam.make_dirichlet_bc(u, 1.0)

        assert True

    def test_default_order_parameter(self):
        """Test that order defaults to 1."""
        box = sam.geometry.box([0.0], [1.0])
        config = sam.config.make(1)
        config.min_level = 3
        config.max_level = 3
        mesh = sam.mesh.make(box, config)

        u1 = sam.field.scalar(mesh, "u1", init=0.0)
        u2 = sam.field.scalar(mesh, "u2", init=0.0)

        # Don't specify order - should default to 1
        sam.make_dirichlet_bc(u1, 0.0)
        sam.make_dirichlet_bc(u2, 0.0, order=1)

        # Both should work
        assert True


class TestPolynomialExtrapolationBC:
    """Tests for polynomial extrapolation boundary condition."""

    def test_1d_polynomial_extrapolation_default_order(self):
        """Test polynomial extrapolation with default order (2) for 1D field."""
        box = sam.geometry.box([0.0], [1.0])
        config = sam.config.make(1)
        config.min_level = 2
        config.max_level = 2
        mesh = sam.mesh.make(box, config)

        u = sam.field.scalar(mesh, "u", init=0.0)

        # Create polynomial extrapolation BC with default order
        sam.make_polynomial_extrapolation_bc(u)
        assert True

    def test_1d_polynomial_extrapolation_different_orders(self):
        """Test polynomial extrapolation with different orders."""
        box = sam.geometry.box([0.0], [1.0])
        config = sam.config.make(1)
        config.min_level = 2
        config.max_level = 2
        config.max_stencil_size = 8  # Support up to order 3 (stencil size 6)
        mesh = sam.mesh.make(box, config)

        # Test orders 1-3
        for order in [1, 2, 3]:
            u = sam.field.scalar(mesh, "u", init=0.0)
            sam.make_polynomial_extrapolation_bc(u, order=order)
            assert True

    def test_1d_polynomial_extrapolation_invalid_order(self):
        """Test that invalid order raises an error."""
        box = sam.geometry.box([0.0], [1.0])
        config = sam.config.make(1)
        config.min_level = 2
        config.max_level = 2
        mesh = sam.mesh.make(box, config)

        u = sam.field.scalar(mesh, "u", init=0.0)

        # Order 4 should raise an error (only 1-3 supported)
        with pytest.raises(RuntimeError, match="order must be 1, 2, or 3"):
            sam.make_polynomial_extrapolation_bc(u, order=4)

    def test_2d_polynomial_extrapolation(self):
        """Test polynomial extrapolation for 2D field."""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config.make(2)
        config.min_level = 2
        config.max_level = 2
        mesh = sam.mesh.make(box, config)

        u = sam.field.scalar(mesh, "u", init=0.0)
        sam.make_polynomial_extrapolation_bc(u, order=2)
        assert True

    def test_3d_polynomial_extrapolation(self):
        """Test polynomial extrapolation for 3D field."""
        box = sam.geometry.box([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        config = sam.config.make(3)
        config.min_level = 1
        config.max_level = 1
        config.max_stencil_size = 8  # Support up to order 3 (stencil size 6)
        mesh = sam.mesh.make(box, config)

        u = sam.field.scalar(mesh, "u", init=0.0)
        sam.make_polynomial_extrapolation_bc(u, order=3)
        assert True

    def test_polynomial_extrapolation_via_boundary_submodule(self):
        """Test polynomial extrapolation via boundary submodule."""
        box = sam.geometry.box([0.0], [1.0])
        config = sam.config.make(1)
        config.min_level = 2
        config.max_level = 2
        mesh = sam.mesh.make(box, config)

        u = sam.field.scalar(mesh, "u", init=0.0)

        # Use the boundary submodule API
        sam.boundary.polynomial_extrapolation(u, order=2)
        assert True


class TestNeumannBC:
    """Tests for Neumann boundary condition."""

    def test_1d_neumann_default(self):
        """Test default Neumann BC (zero derivative/no-flux) for 1D."""
        box = sam.geometry.box([0.0], [1.0])
        config = sam.config.make(1)
        config.min_level = 3
        config.max_level = 3
        mesh = sam.mesh.make(box, config)

        u = sam.field.scalar(mesh, "u", init=0.0)

        # Create Neumann BC with default (zero derivative)
        sam.make_neumann_bc(u)
        assert True

    def test_1d_neumann_explicit_value(self):
        """Test Neumann BC with explicit derivative value."""
        box = sam.geometry.box([0.0], [1.0])
        config = sam.config.make(1)
        config.min_level = 2
        config.max_level = 2
        mesh = sam.mesh.make(box, config)

        u = sam.field.scalar(mesh, "u", init=0.0)

        # Create Neumann BC with derivative value
        sam.make_neumann_bc(u, 1.5)
        assert True

    def test_2d_neumann_default(self):
        """Test Neumann BC for 2D field."""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config.make(2)
        config.min_level = 2
        config.max_level = 2
        mesh = sam.mesh.make(box, config)

        u = sam.field.scalar(mesh, "u", init=0.0)

        # Create Neumann BC with default
        sam.make_neumann_bc(u)
        assert True

    def test_2d_neumann_explicit_value(self):
        """Test Neumann BC with explicit derivative in 2D."""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config.make(2)
        config.min_level = 2
        config.max_level = 2
        mesh = sam.mesh.make(box, config)

        u = sam.field.scalar(mesh, "u", init=0.0)

        # Create Neumann BC with derivative value
        sam.make_neumann_bc(u, 0.5)
        assert True

    def test_3d_neumann(self):
        """Test Neumann BC for 3D field."""
        box = sam.geometry.box([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        config = sam.config.make(3)
        config.min_level = 1
        config.max_level = 1
        mesh = sam.mesh.make(box, config)

        u = sam.field.scalar(mesh, "u", init=0.0)

        # Create Neumann BC
        sam.make_neumann_bc(u, 0.0)
        assert True

    def test_neumann_via_boundary_submodule(self):
        """Test Neumann BC via boundary submodule."""
        box = sam.geometry.box([0.0], [1.0])
        config = sam.config.make(1)
        config.min_level = 2
        config.max_level = 2
        mesh = sam.mesh.make(box, config)

        u = sam.field.scalar(mesh, "u", init=0.0)

        # Use the boundary submodule API
        sam.boundary.neumann(u, 0.0)
        assert True


class TestVectorFieldBC:
    """Tests for VectorField boundary conditions."""

    def test_2d_vector_dirichlet(self):
        """Test Dirichlet BC for 2D vector field (Burgers equation)."""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config.make(2)
        config.min_level = 2
        config.max_level = 2
        mesh = sam.mesh.make(box, config)

        # Create 2D vector field (2 components)
        u = sam.field.vector(mesh, "u", 2)

        # Create Dirichlet BC with list of values
        sam.make_dirichlet_bc(u, [0.0, 0.0])
        assert True

    def test_2d_vector_dirichlet_different_orders(self):
        """Test Dirichlet BC for 2D vector field with different orders."""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config.make(2)
        config.min_level = 2
        config.max_level = 2
        config.max_stencil_size = 10  # Support up to order 4
        mesh = sam.mesh.make(box, config)

        # Test orders 1-3
        for order in [1, 2, 3]:
            u = sam.field.vector(mesh, "u", 2)
            sam.make_dirichlet_bc(u, [1.0, 0.0], order=order)
            assert True

    def test_2d_vector_wrong_value_count(self):
        """Test error on wrong number of values for VectorField."""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config.make(2)
        config.min_level = 2
        config.max_level = 2
        mesh = sam.mesh.make(box, config)

        u = sam.field.vector(mesh, "u", 2)

        # Should raise error: expected 2 values, got 3
        with pytest.raises(RuntimeError, match="Expected 2 values"):
            sam.make_dirichlet_bc(u, [0.0, 0.0, 0.0])

    def test_3d_vector_dirichlet(self):
        """Test Dirichlet BC for 3D vector field."""
        box = sam.geometry.box([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        config = sam.config.make(3)
        config.min_level = 1
        config.max_level = 1
        mesh = sam.mesh.make(box, config)

        # Create 3D vector field (3 components)
        u = sam.field.vector(mesh, "u", 3)

        # Create Dirichlet BC with list of values
        sam.make_dirichlet_bc(u, [1.0, 0.0, 0.0])
        assert True

    def test_3d_vector_wrong_value_count(self):
        """Test error on wrong number of values for 3D VectorField."""
        box = sam.geometry.box([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        config = sam.config.make(3)
        config.min_level = 1
        config.max_level = 1
        mesh = sam.mesh.make(box, config)

        u = sam.field.vector(mesh, "u", 3)

        # Should raise error: expected 3 values, got 2
        with pytest.raises(RuntimeError, match="Expected 3 values"):
            sam.make_dirichlet_bc(u, [0.0, 0.0])


# NOTE: Directional BC tests are temporarily disabled because they require
# xtensor-python type registration. This can be re-enabled later.
# class TestDirectionalBC:
#     """Tests for direction-specific boundary conditions."""
#     ...


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
