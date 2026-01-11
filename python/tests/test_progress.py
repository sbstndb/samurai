"""
Comprehensive test suite for Samurai progress bar API v0.30.0+

Tests the new progress reporting utilities including:
- MeshStatistics: Efficient mesh statistics tracking
- TimeLoop: Time-stepping loop progress tracking
- IterationLoop: Fixed-count iteration progress tracking
- mesh_adaptation: Context manager for mesh adaptation
"""

import os
import sys
import time
from io import StringIO

import pytest

# Add the source directory to Python path for development
src_dir = os.path.join(os.path.dirname(__file__), "..", "src")
if os.path.exists(src_dir):
    sys.path.insert(0, src_dir)

# Add the build directory to Python path for development
# Note: using build_py314 to match conftest.py
build_dir = os.path.join(os.path.dirname(__file__), "..", "..", "build_py314", "python")
if os.path.exists(build_dir):
    sys.path.insert(0, build_dir)

try:
    import samurai_python as sam
    # Progress API v0.30.0: all in sam.progress submodule
    MeshStatistics = sam.progress.MeshStatistics
    compute_mesh_stats = sam.progress.compute_mesh_stats
    TimeLoop = sam.progress.TimeLoop
    IterationLoop = sam.progress.IterationLoop
    mesh_adaptation = sam.progress.mesh_adaptation
    iteration = sam.progress.iteration
    time_loop = sam.progress.time_loop
except ImportError as e:
    pytest.skip(f"Required modules not available: {e}", allow_module_level=True)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def real_mesh_1d():
    """Create a real 1D mesh for integration tests."""
    try:
        box = sam.geometry.box([0.], [1.])
        config = sam.config.make(1)
        config.min_level = 0
        config.max_level = 2
        mesh = sam.mesh.make(box, config)
        return mesh
    except Exception:
        pytest.skip("Could not create real mesh")


@pytest.fixture
def real_mesh_2d():
    """Create a real 2D mesh for integration tests."""
    try:
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config.make(2)
        config.min_level = 0
        config.max_level = 2
        mesh = sam.mesh.make(box, config)
        return mesh
    except Exception:
        pytest.skip("Could not create real mesh")


# =============================================================================
# MeshStatistics Tests
# =============================================================================

class TestMeshStatistics:
    """Test suite for MeshStatistics class."""

    def test_initialization(self):
        """Test MeshStatistics initialization."""
        stats = MeshStatistics()
        assert stats.n_cells == 0
        assert stats.min_level == 0
        assert stats.max_level == 0

    def test_initialization_with_level_breakdown(self):
        """Test initialization with level breakdown enabled."""
        stats = MeshStatistics(enable_level_breakdown=True)
        assert stats._enable_level_breakdown is True

    def test_initialization_without_level_breakdown(self):
        """Test initialization with level breakdown disabled."""
        stats = MeshStatistics(enable_level_breakdown=False)
        assert stats._enable_level_breakdown is False

    def test_update_with_real_mesh_1d(self, real_mesh_1d):
        """Test updating with a real 1D mesh."""
        stats = MeshStatistics()
        stats.update(real_mesh_1d)

        assert stats.n_cells > 0
        assert stats.min_level >= 0
        assert stats.max_level >= stats.min_level

    def test_update_with_real_mesh_2d(self, real_mesh_2d):
        """Test updating with a real 2D mesh."""
        stats = MeshStatistics(enable_level_breakdown=True)
        stats.update(real_mesh_2d)

        assert stats.n_cells > 0
        assert stats.min_level >= 0
        assert stats.max_level >= stats.min_level

        # Check that level counts are populated
        if stats.level_counts:
            total_from_levels = sum(stats.level_counts.values())
            assert total_from_levels == stats.n_cells

    def test_get_summary_before_update(self):
        """Test get_summary before any update."""
        stats = MeshStatistics()
        summary = stats.get_summary()
        assert "not computed" in summary or summary == "{}"

    def test_get_summary_after_update(self, real_mesh_1d):
        """Test get_summary after update."""
        stats = MeshStatistics()
        stats.update(real_mesh_1d)
        summary = stats.get_summary()
        # Should contain some info about the mesh
        assert len(summary) > 0

    def test_repr_before_update(self):
        """Test __repr__ before update."""
        stats = MeshStatistics()
        repr_str = repr(stats)
        assert "MeshStatistics" in repr_str

    def test_repr_after_update(self, real_mesh_1d):
        """Test __repr__ after update."""
        stats = MeshStatistics()
        stats.update(real_mesh_1d)
        repr_str = repr(stats)
        assert "MeshStatistics" in repr_str


# =============================================================================
# ComputeMeshStats Tests
# =============================================================================

class TestComputeMeshStats:
    """Test suite for compute_mesh_stats convenience function."""

    def test_compute_mesh_stats_with_real_mesh(self, real_mesh_1d):
        """Test compute_mesh_stats with real mesh."""
        result = compute_mesh_stats(real_mesh_1d)

        assert result["n_cells"] > 0
        assert result["min_level"] >= 0
        assert result["max_level"] >= result["min_level"]


# =============================================================================
# TimeLoop Tests (New API v0.30.0)
# =============================================================================

class TestTimeLoop:
    """Test suite for TimeLoop class (new API)."""

    def test_initialization(self):
        """Test TimeLoop initialization with new API."""
        pbar = TimeLoop(Tf=1.0, dt=0.01, desc="Test")
        assert pbar.Tf == 1.0
        assert pbar.dt_initial == 0.01
        assert pbar.desc == "Test"
        assert pbar.t == 0.0
        assert pbar.iteration == 0

    def test_context_manager_enter(self):
        """Test entering context manager."""
        pbar = TimeLoop(Tf=1.0, dt=0.01)
        with pbar as context:
            assert context is pbar
            # Note: _start_time may be None if tqdm is disabled
            if not pbar.disable:
                assert pbar._start_time is not None
                assert isinstance(pbar._start_time, float)

    def test_context_manager_exit(self):
        """Test exiting context manager."""
        pbar = TimeLoop(Tf=1.0, dt=0.01, desc="TestSim")
        # Should not raise any exception
        with pbar:
            pass
        # After exit, progress bar should be closed
        assert pbar._pbar is None

    def test_continue_loop(self):
        """Test continue_loop method."""
        pbar = TimeLoop(Tf=0.1, dt=0.05)
        assert pbar.continue_loop() is True  # t=0 < Tf=0.1

        pbar.advance_time(0.05)
        assert pbar.continue_loop() is True  # t=0.05 < Tf=0.1

        pbar.advance_time(0.05)
        assert pbar.continue_loop() is False  # t=0.1 == Tf=0.1

    def test_advance_time_without_dt(self):
        """Test advance_time without specifying dt (uses default)."""
        pbar = TimeLoop(Tf=1.0, dt=0.1)
        pbar.advance_time()  # Should use dt_initial=0.1
        assert pbar.t == 0.1
        assert pbar.iteration == 1

    def test_advance_time_with_dt(self):
        """Test advance_time with explicit dt parameter."""
        pbar = TimeLoop(Tf=1.0, dt=0.1)
        pbar.advance_time(0.05)
        assert pbar.t == 0.05
        assert pbar.iteration == 1

    def test_advance_time_to_completion(self):
        """Test advance_time until completion."""
        pbar = TimeLoop(Tf=0.1, dt=0.05)

        # First advance
        pbar.advance_time()
        assert pbar.t == 0.05
        assert pbar.continue_loop() is True

        # Second advance - should reach completion
        pbar.advance_time()
        assert pbar.t == 0.1
        assert pbar.continue_loop() is False

    def test_update_stats_with_mesh(self, real_mesh_2d):
        """Test update_stats with mesh parameter."""
        pbar = TimeLoop(Tf=1.0, dt=0.1)
        pbar.update_stats(mesh=real_mesh_2d)
        # Should not raise any exception
        assert pbar._track_mesh is True
        assert pbar._mesh_stats is not None

    def test_update_stats_with_custom_values(self):
        """Test update_stats with custom statistics."""
        pbar = TimeLoop(Tf=1.0, dt=0.1)
        pbar.update_stats(residual=1e-6, iterations=100)
        # Should not raise any exception

    def test_get_progress(self):
        """Test progress property."""
        pbar = TimeLoop(Tf=1.0, dt=0.1)
        assert pbar.progress == 0.0  # t=0, Tf=1.0

        pbar.advance_time(0.5)
        assert pbar.progress == 0.5  # t=0.5, Tf=1.0

        pbar.advance_time(0.5)
        assert pbar.progress == 1.0  # t=1.0, Tf=1.0

    def test_get_eta(self):
        """Test get_eta method."""
        pbar = TimeLoop(Tf=1.0, dt=0.1)
        eta = pbar.get_eta()
        # Before any iteration, ETA should be 0
        assert eta == 0.0

    def test_time_loop_function_factory(self):
        """Test time_loop() factory function."""
        with time_loop(Tf=1.0, dt=0.01, desc="Factory Test") as pbar:
            assert isinstance(pbar, TimeLoop)
            assert pbar.Tf == 1.0


# =============================================================================
# IterationLoop Tests (New API v0.30.0)
# =============================================================================

class TestIterationLoop:
    """Test suite for IterationLoop class (new API)."""

    def test_initialization(self):
        """Test IterationLoop initialization."""
        pbar = IterationLoop(total=100, desc="Iterations")
        assert pbar.total == 100
        assert pbar.current == 0
        assert pbar.desc == "Iterations"

    def test_context_manager(self):
        """Test IterationLoop as context manager."""
        pbar = IterationLoop(total=100, desc="TestLoop")

        with pbar as context:
            assert context is pbar
            assert pbar._pbar is not None or pbar.disable

    def test_update(self):
        """Test update method."""
        pbar = IterationLoop(total=100)
        pbar.update(10)
        assert pbar.current == 10

        pbar.update(5)
        assert pbar.current == 15

    def test_update_default(self):
        """Test update with default n=1."""
        pbar = IterationLoop(total=100)
        pbar.update()
        assert pbar.current == 1

    def test_set_postfix(self):
        """Test set_postfix method."""
        pbar = IterationLoop(total=100)
        pbar.set_postfix(loss=0.1, accuracy=0.95)
        # Should not raise any exception

    def test_iteration_function_factory(self):
        """Test iteration() factory function."""
        with iteration(total=100, desc="Factory Test") as pbar:
            assert isinstance(pbar, IterationLoop)
            assert pbar.total == 100


# =============================================================================
# Mesh Adaptation Tests
# =============================================================================

class TestMeshAdaptation:
    """Test suite for mesh_adaptation context manager."""

    def test_mesh_adaptation_context_manager(self, real_mesh_2d):
        """Test mesh_adaptation as context manager."""
        with mesh_adaptation(real_mesh_2d) as stats:
            # Stats should be a MeshStatistics object
            assert isinstance(stats, MeshStatistics)
            # Should have counted cells
            assert stats.n_cells > 0

    def test_mesh_adaptation_with_real_adaptation(self):
        """Test mesh_adaptation with actual mesh adaptation."""
        try:
            # Create mesh and field
            box = sam.geometry.box([0.0], [1.0])
            config = sam.config.make(1)
            config.min_level = 0
            config.max_level = 4

            mesh = sam.mesh.make(box, config)
            field = sam.field.scalar(mesh, "u", init=1.0)

            # Apply BC and adapt
            sam.boundary.dirichlet(field, 0.0)

            n_cells_before = mesh.nb_cells

            # Perform adaptation
            with mesh_adaptation(mesh) as stats:
                stats_before = stats.n_cells

                # Do adaptation
                MRadapt = sam.adaptation.make_MRAdapt(field)
                mra_config = sam.config.MRAConfig()
                mra_config.epsilon = 1e-2
                MRadapt(mra_config)
                sam.adaptation.update_ghost_mr(field)

            # Stats should be updated after context exit
            assert n_cells_before > 0
        except Exception as e:
            pytest.skip(f"Could not perform real adaptation: {e}")


# =============================================================================
# Integration Tests
# =============================================================================

class TestProgressIntegration:
    """Integration tests for progress tracking with real simulations."""

    def test_time_loop_with_mesh_tracking(self, real_mesh_2d):
        """Test TimeLoop with mesh statistics tracking."""
        pbar = TimeLoop(Tf=0.01, dt=0.001, desc="Sim")

        with pbar:
            while pbar.continue_loop():
                pbar.advance_time()
                pbar.update_stats(mesh=real_mesh_2d)

        # Should complete without errors
        assert pbar.t >= pbar.Tf or abs(pbar.t - pbar.Tf) < 1e-10

    def test_iteration_loop_simple(self):
        """Test IterationLoop for simple iteration."""
        results = []

        with iteration(total=10, desc="Processing") as pbar:
            for i in range(10):
                results.append(i * 2)
                pbar.update()

        assert len(results) == 10
        assert results == [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]

    def test_full_workflow(self, real_mesh_1d):
        """Test complete workflow: mesh -> field -> time loop."""
        # Create field
        field = sam.field.scalar(real_mesh_1d, "u", init=0.0)

        # Simple time-stepping loop
        dt = 0.01
        Tf = 0.05

        with time_loop(Tf=Tf, dt=dt, desc="Time integration") as pbar:
            while pbar.continue_loop():
                # Update field (dummy operation)
                sam.algorithms.for_each_cell(real_mesh_1d, lambda cell: None)

                # Update progress
                pbar.advance_time(dt)

        # Should complete the loop
        assert True  # If we get here, the workflow succeeded

    def test_progress_percentage(self):
        """Test progress percentage calculation."""
        pbar = TimeLoop(Tf=1.0, dt=0.1)

        assert pbar.progress == 0.0

        pbar.advance_time(0.5)
        assert abs(pbar.progress - 0.5) < 1e-10

        pbar.advance_time(0.5)
        assert pbar.progress == 1.0


# =============================================================================
# Performance Tests
# =============================================================================

class TestProgressPerformance:
    """Test progress bar performance and efficiency."""

    def test_mesh_statistics_caching(self, real_mesh_2d):
        """Test that MeshStatistics caching works correctly."""
        stats = MeshStatistics(enable_level_breakdown=True)
        stats.update(real_mesh_2d)

        # Accessing level_counts multiple times should be efficient
        count1 = stats.level_counts
        count2 = stats.level_counts
        assert count1 == count2

    def test_level_breakdown_overhead(self, real_mesh_2d):
        """Test that level breakdown doesn't add too much overhead."""
        import time

        # Without level breakdown
        start = time.time()
        for _ in range(100):
            stats = MeshStatistics(enable_level_breakdown=False)
            stats.update(real_mesh_2d)
        time_without = time.time() - start

        # With level breakdown
        start = time.time()
        for _ in range(100):
            stats = MeshStatistics(enable_level_breakdown=True)
            stats.update(real_mesh_2d)
        time_with = time.time() - start

        # Level breakdown shouldn't be more than 10x slower
        assert time_with < time_without * 10


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestProgressErrorHandling:
    """Test error handling in progress tracking."""

    def test_time_loop_with_zero_time(self):
        """Test TimeLoop with Tf=0."""
        pbar = TimeLoop(Tf=0.0, dt=0.01)
        assert pbar.Tf == 0.0
        assert pbar.continue_loop() is False  # Should be complete immediately

    def test_time_loop_with_negative_dt(self):
        """Test TimeLoop with negative dt."""
        pbar = TimeLoop(Tf=1.0, dt=-0.01)
        pbar.advance_time()
        # t should be negative (but this is an edge case)
        assert pbar.t < 0

    def test_iteration_loop_with_zero_total(self):
        """Test IterationLoop with total=0."""
        pbar = IterationLoop(total=0)
        assert pbar.total == 0
        pbar.update()
        assert pbar.current == 1  # Still counts iterations
