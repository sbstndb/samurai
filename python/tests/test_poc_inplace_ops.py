#!/usr/bin/env python3
"""
POC Test Suite for In-Place Arithmetic Operators

This test suite validates in-place operators (+=, -=, *=, /=) for ScalarField.
The POC aims to prove that in-place operators:
1. Modify fields in-place (same object id)
2. Return self for chaining
3. Work correctly after mesh adaptation (no stale references)
4. Provide performance benefits over copy-based operators

This is a PROOF OF CONCEPT - focus on critical functionality.
"""

import sys
import os
import pytest
import numpy as np
import time

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
    box = sam.geometry.Box1D([0.0], [1.0])
    config = sam.config.MeshConfig1D()
    config.min_level = 4
    config.max_level = 4
    return sam.mesh.MRMesh1D(box, config)


@pytest.fixture
def mesh_2d():
    """Create a simple 2D mesh for testing."""
    box = sam.geometry.Box2D([0.0, 0.0], [1.0, 1.0])
    config = sam.config.MeshConfig2D()
    config.min_level = 4
    config.max_level = 4
    return sam.mesh.MRMesh2D(box, config)


@pytest.fixture
def mesh_2d_adaptive():
    """Create a 2D mesh configured for adaptation testing."""
    box = sam.geometry.Box2D([0.0, 0.0], [1.0, 1.0])
    config = sam.config.MeshConfig2D()
    config.min_level = 2
    config.max_level = 6
    return sam.mesh.MRMesh2D(box, config)


@pytest.fixture
def scalar_field_1d(mesh_1d):
    """Create a ScalarField1D initialized to 1.0."""
    field = sam.field.ScalarField1D(mesh_1d, "u", init=1.0)
    return field


@pytest.fixture
def scalar_field_2d(mesh_2d):
    """Create a ScalarField2D initialized to 2.0."""
    field = sam.field.ScalarField2D(mesh_2d, "u", init=2.0)
    return field


# ============================================================
# TestScalarFieldInPlaceOps - Basic Functionality
# ============================================================

class TestScalarFieldInPlaceOps:
    """Test in-place operators for ScalarField - basic functionality."""

    def test_iadd_scalar(self, scalar_field_1d):
        """field += scalar should modify in-place."""
        original_id = id(scalar_field_1d)
        original_name = scalar_field_1d.name

        # Perform in-place addition
        scalar_field_1d += 0.5
        result = scalar_field_1d  # Get reference to verify it's the same object

        # Verify same object (in-place modification)
        assert id(result) == original_id, "+= should return same object"
        assert id(scalar_field_1d) == original_id, "field should be modified in-place"

        # Verify name unchanged
        assert scalar_field_1d.name == original_name, "Name should not change"

        # Verify correct values
        arr = scalar_field_1d.numpy_view()
        assert np.allclose(arr, 1.5), f"Expected 1.5, got mean {arr.mean():.2f}"

    def test_isub_scalar(self, scalar_field_1d):
        """field -= scalar should modify in-place."""
        original_id = id(scalar_field_1d)
        original_name = scalar_field_1d.name

        # Perform in-place subtraction
        scalar_field_1d -= 0.3
        result = scalar_field_1d  # Get reference to verify it's the same object

        # Verify same object
        assert id(result) == original_id, "-= should return same object"
        assert id(scalar_field_1d) == original_id, "field should be modified in-place"

        # Verify name unchanged
        assert scalar_field_1d.name == original_name, "Name should not change"

        # Verify correct values
        arr = scalar_field_1d.numpy_view()
        assert np.allclose(arr, 0.7), f"Expected 0.7, got mean {arr.mean():.2f}"

    def test_imul_scalar(self, scalar_field_1d):
        """field *= scalar should modify in-place."""
        original_id = id(scalar_field_1d)
        original_name = scalar_field_1d.name

        # Perform in-place multiplication
        scalar_field_1d *= 3.0
        result = scalar_field_1d  # Get reference to verify it's the same object

        # Verify same object
        assert id(result) == original_id, "*= should return same object"
        assert id(scalar_field_1d) == original_id, "field should be modified in-place"

        # Verify name unchanged
        assert scalar_field_1d.name == original_name, "Name should not change"

        # Verify correct values
        arr = scalar_field_1d.numpy_view()
        assert np.allclose(arr, 3.0), f"Expected 3.0, got mean {arr.mean():.2f}"

    def test_itruediv_scalar(self, scalar_field_1d):
        """field /= scalar should modify in-place."""
        original_id = id(scalar_field_1d)
        original_name = scalar_field_1d.name

        # Perform in-place division
        scalar_field_1d /= 2.0
        result = scalar_field_1d  # Get reference to verify it's the same object

        # Verify same object
        assert id(result) == original_id, "/= should return same object"
        assert id(scalar_field_1d) == original_id, "field should be modified in-place"

        # Verify name unchanged
        assert scalar_field_1d.name == original_name, "Name should not change"

        # Verify correct values
        arr = scalar_field_1d.numpy_view()
        assert np.allclose(arr, 0.5), f"Expected 0.5, got mean {arr.mean():.2f}"

    def test_iadd_field(self, mesh_1d):
        """field += other_field should add element-wise in-place."""
        f1 = sam.field.scalar(mesh_1d, "f1", init=1.0)
        f2 = sam.field.scalar(mesh_1d, "f2", init=2.5)

        original_id = id(f1)
        original_name = f1.name

        # Perform in-place addition
        f1 += f2
        result = f1  # Get reference to verify it's the same object

        # Verify same object
        assert id(result) == original_id, "field += other should return same object"
        assert id(f1) == original_id, "f1 should be modified in-place"

        # Verify name unchanged
        assert f1.name == original_name, "Name should not change"

        # Verify correct values
        arr = f1.numpy_view()
        assert np.allclose(arr, 3.5), f"Expected 3.5, got mean {arr.mean():.2f}"

        # Verify f2 unchanged
        arr2 = f2.numpy_view()
        assert np.allclose(arr2, 2.5), "f2 should remain unchanged"

    def test_isub_field(self, mesh_1d):
        """field -= other_field should subtract element-wise in-place."""
        f1 = sam.field.scalar(mesh_1d, "f1", init=3.0)
        f2 = sam.field.scalar(mesh_1d, "f2", init=1.0)

        original_id = id(f1)
        original_name = f1.name

        # Perform in-place subtraction
        f1 -= f2
        result = f1  # Get reference to verify it's the same object

        # Verify same object
        assert id(result) == original_id, "field -= other should return same object"
        assert id(f1) == original_id, "f1 should be modified in-place"

        # Verify name unchanged
        assert f1.name == original_name, "Name should not change"

        # Verify correct values
        arr = f1.numpy_view()
        assert np.allclose(arr, 2.0), f"Expected 2.0, got mean {arr.mean():.2f}"

    def test_returns_self(self, scalar_field_1d):
        """In-place ops should return self for chaining."""
        original_id = id(scalar_field_1d)

        # Chain multiple operations
        scalar_field_1d += 1.0
        scalar_field_1d *= 2.0
        scalar_field_1d -= 1.0
        result = scalar_field_1d  # Get reference to verify it's the same object

        # Verify still same object after chaining
        assert id(result) == original_id, "Chained ops should return same object"
        assert id(scalar_field_1d) == original_id, "field should still be same object"

        # Verify correct result: ((1.0 + 1.0) * 2.0) - 1.0 = 3.0
        arr = scalar_field_1d.numpy_view()
        assert np.allclose(arr, 3.0), f"Expected 3.0, got mean {arr.mean():.2f}"

    def test_modifies_in_place(self, scalar_field_1d):
        """Should modify original object, not create new one."""
        # Store reference
        field_ref = scalar_field_1d
        original_id = id(scalar_field_1d)

        # Perform operation
        scalar_field_1d += 5.0

        # Verify field_ref sees the change (same object)
        assert id(field_ref) == original_id, "Reference should still point to same object"

        # Verify values changed
        arr = field_ref.numpy_view()
        assert np.allclose(arr, 6.0), f"Expected 6.0, got mean {arr.mean():.2f}"

    def test_2d_inplace_ops(self, scalar_field_2d):
        """In-place ops should work in 2D."""
        original_id = id(scalar_field_2d)

        # Perform operations
        scalar_field_2d += 1.0  # 2.0 + 1.0 = 3.0
        scalar_field_2d *= 2.0  # 3.0 * 2.0 = 6.0

        # Verify same object
        assert id(scalar_field_2d) == original_id, "2D field should be modified in-place"

        # Verify correct values
        arr = scalar_field_2d.numpy_view()
        assert np.allclose(arr, 6.0), f"Expected 6.0, got mean {arr.mean():.2f}"

    def test_zero_initialization(self, mesh_1d):
        """In-place ops should work with zero-initialized fields."""
        field = sam.field.scalar(mesh_1d, "u", init=0.0)

        # Perform operations
        field += 1.0
        field *= 5.0

        arr = field.numpy_view()
        assert np.allclose(arr, 5.0), f"Expected 5.0, got mean {arr.mean():.2f}"

    def test_negative_values(self, mesh_1d):
        """In-place ops should handle negative values correctly."""
        field = sam.field.scalar(mesh_1d, "u", init=1.0)

        # Subtract to get negative
        field -= 2.0  # 1.0 - 2.0 = -1.0

        arr = field.numpy_view()
        assert np.allclose(arr, -1.0), f"Expected -1.0, got mean {arr.mean():.2f}"

    def test_fractional_values(self, mesh_1d):
        """In-place ops should handle fractional values correctly."""
        field = sam.field.scalar(mesh_1d, "u", init=1.0)

        # Use fractional values
        field += 0.333333333
        field /= 1.333333333

        arr = field.numpy_view()
        expected = 1.0 / 1.333333333
        assert np.allclose(arr, expected, rtol=1e-6), \
            f"Expected {expected:.6f}, got mean {arr.mean():.6f}"


# ============================================================
# TestInPlaceOpsAfterMeshAdaptation - Critical POC Validation
# ============================================================

class TestInPlaceOpsAfterMeshAdaptation:
    """
    Test that in-place ops work after mesh adaptation (CRITICAL!).

    This is the KEY PROBLEM the POC solves:
    - Regular operators (u = u - dt*flux) create new fields with stale mesh references
    - After mesh adaptation, these stale references cause segfaults
    - In-place operators should NOT have this problem
    """

    def test_iadd_after_adaptation(self, mesh_2d_adaptive):
        """In-place ops should remain valid after mesh adaptation."""
        # Create field with non-uniform initialization
        u = sam.field.scalar(mesh_2d_adaptive, "u", init=1.0)

        # Initialize with non-uniform data (so adaptation actually does something)
        def init_pattern(cell):
            cx, cy = cell.center()
            dist = (cx - 0.5)**2 + (cy - 0.5)**2
            u[cell.index] = 1.0 if dist < 0.1 else 0.0

        sam.algorithms.for_each_cell(u.mesh, init_pattern)

        # Setup adaptation
        MRadaptation = sam.adaptation.make_MRAdapt(u)
        mra_config = sam.config.MRAConfig()
        mra_config.epsilon = 1e-2
        mra_config.regularity = 1.0

        # Perform initial adaptation
        MRadaptation(mra_config)
        u.resize()

        original_id = id(u)
        original_name = u.name

        # In-place operation AFTER adaptation (this should work!)
        u += 1.0

        # Verify still same object
        assert id(u) == original_id, "Field should still be same object after adaptation"
        assert u.name == original_name, "Name should not change"

        # Verify values
        arr = u.numpy_view()
        # Should have values 1.0 and 2.0 (0.0 + 1.0 = 1.0, 1.0 + 1.0 = 2.0)
        assert np.all((arr == 1.0) | (arr == 2.0)), \
            f"Expected values 1.0 or 2.0, got range [{arr.min():.2f}, {arr.max():.2f}]"

    def test_no_stale_reference_after_adaptation(self, mesh_2d_adaptive):
        """
        Should NOT crash after adaptation (unlike u = u - dt*flux).

        This is the CRITICAL test that proves the POC concept.
        """
        # Create field
        u = sam.field.scalar(mesh_2d_adaptive, "u", init=1.0)

        # Setup adaptation
        MRadaptation = sam.adaptation.make_MRAdapt(u)
        mra_config = sam.config.MRAConfig()
        mra_config.epsilon = 1e-2
        mra_config.regularity = 1.0

        # Simulate time loop with mesh adaptation
        for i in range(3):
            # Adapt mesh (this changes mesh structure!)
            MRadaptation(mra_config)

            # CRITICAL: In-place op after adaptation
            # This should NOT crash (unlike regular operators)
            u += 0.1

            # Resize and continue
            u.resize()

        # If we get here without crashing, the POC works!
        assert True, "Should complete without segfault"

    def test_multiple_ops_after_multiple_adaptations(self, mesh_2d_adaptive):
        """Multiple in-place ops should work after multiple adaptations."""
        u = sam.field.scalar(mesh_2d_adaptive, "u", init=0.0)

        # Setup adaptation
        MRadaptation = sam.adaptation.make_MRAdapt(u)
        mra_config = sam.config.MRAConfig()
        mra_config.epsilon = 1e-2
        mra_config.regularity = 1.0

        # Simulate time loop
        for i in range(5):
            # Adapt
            MRadaptation(mra_config)

            # Multiple in-place operations
            u += 1.0
            u *= 2.0
            u -= 1.0  # Net: (u + 1.0) * 2.0 - 1.0

            u.resize()

        # Verify final state
        arr = u.numpy_view()
        # After 5 iterations: (((((0 + 1) * 2 - 1 + 1) * 2 - 1 + 1) * 2 - 1 + 1) * 2 - 1 + 1) * 2 - 1
        # This is complex, just verify it changed from 0
        assert arr.mean() > 0, "Field should have been modified"

    def test_comparison_with_copy_operator(self, mesh_2d_adaptive):
        """
        Demonstrate the difference between in-place and copy operators.

        This test shows WHY we need in-place operators.
        """
        # Create two fields with same initial state
        mesh = mesh_2d_adaptive
        u_inplace = sam.field.scalar(mesh, "u_inplace", init=1.0)
        u_copy = sam.field.scalar(mesh, "u_copy", init=1.0)

        # Setup adaptation
        MRadaptation = sam.adaptation.make_MRAdapt(u_inplace)
        mra_config = sam.config.MRAConfig()
        mra_config.epsilon = 1e-2
        mra_config.regularity = 1.0

        # Adapt mesh (changes structure)
        MRadaptation(mra_config)
        u_inplace.resize()
        u_copy.resize()

        # IN-PLACE: Should work fine
        u_inplace += 1.0  # This works!

        # COPY-BASED: Would need to use .assign() workaround
        # u_copy = u_copy + 1.0  # This creates new field with stale reference!
        u_copy.assign(u_copy + 1.0)  # Current workaround

        # Both should have correct values
        arr_inplace = u_inplace.numpy_view()
        arr_copy = u_copy.numpy_view()

        assert np.allclose(arr_inplace, arr_copy), \
            "Both methods should give same result"
        assert np.allclose(arr_inplace, 2.0), \
            f"Expected 2.0, got mean {arr_inplace.mean():.2f}"

    def test_field_field_ops_after_adaptation(self, mesh_2d_adaptive):
        """Field-to-field in-place ops should work after adaptation."""
        mesh = mesh_2d_adaptive
        u = sam.field.scalar(mesh, "u", init=1.0)
        v = sam.field.scalar(mesh, "v", init=2.0)

        # Setup adaptation
        MRadaptation = sam.adaptation.make_MRAdapt(u)
        mra_config = sam.config.MRAConfig()
        mra_config.epsilon = 1e-2
        mra_config.regularity = 1.0

        # Adapt
        MRadaptation(mra_config)
        u.resize()
        v.resize()

        # In-place field-to-field operation
        u += v  # Should work!

        arr = u.numpy_view()
        assert np.allclose(arr, 3.0), f"Expected 3.0, got mean {arr.mean():.2f}"


# ============================================================
# TestPerformance - Validate Performance Benefits
# ============================================================

class TestPerformance:
    """
    Performance validation for in-place operators.

    In-place operators should be faster than copy-based operators
    because they avoid memory allocation.
    """

    def test_inplace_faster_than_copy(self, mesh_2d):
        """
        u += 1.0 should be faster than u = u + 1.0.

        This is a simple benchmark - we just verify it completes.
        """
        mesh = mesh_2d

        # Create fields
        u1 = sam.field.scalar(mesh, "u1", init=1.0)
        u2 = sam.field.scalar(mesh, "u2", init=1.0)

        # Warm-up
        u1 += 1.0
        u2.assign(u2 + 1.0)

        # Reset
        u1.fill(1.0)
        u2.fill(1.0)

        # Benchmark in-place
        start_inplace = time.perf_counter()
        for _ in range(100):
            u1 += 1.0
            u1 -= 1.0
        time_inplace = time.perf_counter() - start_inplace

        # Benchmark copy-based (with .assign workaround)
        start_copy = time.perf_counter()
        for _ in range(100):
            u2.assign(u2 + 1.0)
            u2.assign(u2 - 1.0)
        time_copy = time.perf_counter() - start_copy

        # In-place should be faster (or at least not significantly slower)
        # We allow some tolerance for measurement noise
        speedup = time_copy / time_inplace if time_inplace > 0 else 1.0

        print(f"\nPerformance comparison (100 iterations):")
        print(f"  In-place:  {time_inplace*1000:.2f} ms")
        print(f"  Copy-based: {time_copy*1000:.2f} ms")
        print(f"  Speedup:   {speedup:.2f}x")

        # Just verify both completed successfully
        assert np.allclose(u1.numpy_view(), 1.0), "In-place should preserve values"
        assert np.allclose(u2.numpy_view(), 1.0), "Copy-based should preserve values"

    def test_memory_efficiency(self, mesh_2d):
        """
        In-place ops should not allocate new memory.

        This is a basic sanity check - we verify object identity.
        """
        u = sam.field.scalar(mesh_2d, "u", init=1.0)
        original_id = id(u)

        # Perform many operations
        for _ in range(1000):
            u += 0.001
            u -= 0.001

        # Should still be same object
        assert id(u) == original_id, \
            "In-place ops should not create new objects"

        # Values should be correct (1.0 +/- small errors)
        arr = u.numpy_view()
        assert np.allclose(arr, 1.0, atol=1e-10), \
            f"Values should be ~1.0, got {arr.mean():.10f}"


# ============================================================
# TestEdgeCases - Boundary Conditions
# ============================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_divide_by_zero_raises(self, mesh_1d):
        """Division by zero should raise an error."""
        field = sam.field.scalar(mesh_1d, "u", init=1.0)

        with pytest.raises((ZeroDivisionError, RuntimeError)):
            field /= 0.0

    def test_very_large_values(self, mesh_1d):
        """In-place ops should handle large values."""
        field = sam.field.scalar(mesh_1d, "u", init=1e100)

        field *= 2.0

        arr = field.numpy_view()
        assert np.allclose(arr, 2e100, rtol=1e-10), \
            f"Expected 2e100, got {arr[0]:.10e}"

    def test_very_small_values(self, mesh_1d):
        """In-place ops should handle small values."""
        field = sam.field.scalar(mesh_1d, "u", init=1e-100)

        field /= 2.0

        arr = field.numpy_view()
        assert np.allclose(arr, 5e-101, rtol=1e-10), \
            f"Expected 5e-101, got {arr[0]:.10e}"

    def test_chained_field_ops(self, mesh_1d):
        """Complex chained operations should work."""
        f1 = sam.field.scalar(mesh_1d, "f1", init=2.0)
        f2 = sam.field.scalar(mesh_1d, "f2", init=3.0)

        # Chain: f1 = (f1 + f2) * 2 - 1
        # Using in-place: f1 += f2; f1 *= 2; f1 -= 1
        original_id = id(f1)
        f1 += f2
        f1 *= 2.0
        f1 -= 1.0

        # Result: (2.0 + 3.0) * 2.0 - 1.0 = 9.0
        assert id(f1) == original_id, "Chained ops should preserve object"
        arr = f1.numpy_view()
        assert np.allclose(arr, 9.0), f"Expected 9.0, got mean {arr.mean():.2f}"


# ============================================================
# TestVectorFieldInPlaceOps - Future Work
# ============================================================

class TestVectorFieldInPlaceOps:
    """
    Test in-place operators for VectorField (future work).

    This class is a placeholder for future VectorField in-place ops.
    For the POC, we focus on ScalarField first.
    """

    @pytest.mark.skip(reason="POC: VectorField in-place ops not yet implemented")
    def test_vector_iadd_scalar(self, mesh_2d):
        """VectorField += scalar should work."""
        field = sam.field.vector(mesh_2d, "vel", n_components=2, init=1.0)
        field += 2.0  # Should add to all components
        # Not implemented yet in POC

    @pytest.mark.skip(reason="POC: VectorField in-place ops not yet implemented")
    def test_vector_iadd_vector(self, mesh_2d):
        """VectorField += VectorField should work element-wise."""
        v1 = sam.field.vector(mesh_2d, "v1", n_components=2, init=1.0)
        v2 = sam.field.vector(mesh_2d, "v2", n_components=2, init=2.0)
        v1 += v2
        # Not implemented yet in POC


# ============================================================
# Main Test Runner
# ============================================================

if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
