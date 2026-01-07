"""
Tests for NumPy-style field creation helpers.
"""

import sys
import os

# Add the build directory to Python path for development
build_dir = os.path.join(os.path.dirname(__file__), "..", "..", "build", "python")
if os.path.exists(build_dir):
    sys.path.insert(0, build_dir)

try:
    import samurai_python as sam
except ImportError:
    import pytest
    pytest.skip("samurai_python module not built", allow_module_level=True)

import numpy as np


class TestFieldZeros:
    def test_zeros_1d(self):
        config = sam.config.make(1)
        config.min_level = 2
        box = sam.geometry.box([0.0], [1.0])
        mesh = sam.mesh.MRMesh1D(box, config)

        u = sam.field.zeros(mesh, "u")
        assert np.allclose(u.numpy_view(), 0.0)

    def test_zeros_2d(self):
        config = sam.config.make(2)
        config.min_level = 2
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        mesh = sam.mesh.MRMesh2D(box, config)

        u = sam.field.zeros(mesh, "u")
        assert np.allclose(u.numpy_view(), 0.0)

    def test_zeros_3d(self):
        config = sam.config.make(3)
        config.min_level = 2
        box = sam.geometry.box([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        mesh = sam.mesh.MRMesh3D(box, config)

        u = sam.field.zeros(mesh, "u")
        assert np.allclose(u.numpy_view(), 0.0)


class TestFieldOnes:
    def test_ones_2d(self):
        config = sam.config.make(2)
        config.min_level = 2
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        mesh = sam.mesh.MRMesh2D(box, config)

        u = sam.field.ones(mesh, "u")
        assert np.allclose(u.numpy_view(), 1.0)


class TestFieldFull:
    def test_full_2d(self):
        config = sam.config.make(2)
        config.min_level = 2
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        mesh = sam.mesh.MRMesh2D(box, config)

        u = sam.field.full(mesh, 3.14, "pi")
        assert np.allclose(u.numpy_view(), 3.14)


class TestFieldLike:
    def test_zeros_like_2d(self):
        config = sam.config.make(2)
        config.min_level = 2
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        mesh = sam.mesh.MRMesh2D(box, config)

        u = sam.field.scalar(mesh, "u", init=5.0)
        v = sam.field.zeros_like(u, "v")

        # Same mesh type
        assert isinstance(v, type(u))
        # But different values
        assert np.allclose(u.numpy_view(), 5.0)
        assert np.allclose(v.numpy_view(), 0.0)

    def test_ones_like_2d(self):
        config = sam.config.make(2)
        config.min_level = 2
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        mesh = sam.mesh.MRMesh2D(box, config)

        u = sam.field.scalar(mesh, "u", init=5.0)
        v = sam.field.ones_like(u, "v")

        assert np.allclose(v.numpy_view(), 1.0)

    def test_full_like_2d(self):
        config = sam.config.make(2)
        config.min_level = 2
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        mesh = sam.mesh.MRMesh2D(box, config)

        u = sam.field.scalar(mesh, "u", init=5.0)
        v = sam.field.full_like(u, 2.71, "v")

        assert np.allclose(v.numpy_view(), 2.71)


class TestVectorFieldHelpers:
    def test_zeros_vector_2d(self):
        config = sam.config.make(2)
        config.min_level = 2
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        mesh = sam.mesh.MRMesh2D(box, config)

        vel = sam.field.zeros_vector(mesh, n_components=2)
        # Check it's a vector field
        assert hasattr(vel, 'name')
        assert vel.name == "vel"

    def test_zeros_vector_3d(self):
        config = sam.config.make(2)
        config.min_level = 2
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        mesh = sam.mesh.MRMesh2D(box, config)

        B = sam.field.zeros_vector(mesh, "B", n_components=3)
        assert B.name == "B"

    def test_ones_vector_2d(self):
        config = sam.config.make(2)
        config.min_level = 2
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        mesh = sam.mesh.MRMesh2D(box, config)

        vel = sam.field.ones_vector(mesh, "vel", n_components=2)
        assert vel.name == "vel"

    def test_full_vector_2d(self):
        config = sam.config.make(2)
        config.min_level = 2
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        mesh = sam.mesh.MRMesh2D(box, config)

        vel = sam.field.full_vector(mesh, 2.5, "vel", n_components=2)
        assert vel.name == "vel"

    def test_zeros_like_vector(self):
        config = sam.config.make(2)
        config.min_level = 2
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        mesh = sam.mesh.MRMesh2D(box, config)

        vel = sam.field.zeros_vector(mesh, "vel", n_components=2)
        vel2 = sam.field.zeros_like_vector(vel, "vel2")

        # Same type
        assert type(vel2) == type(vel)
        # Different name
        assert vel2.name == "vel2"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
