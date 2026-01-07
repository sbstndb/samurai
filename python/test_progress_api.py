#!/usr/bin/env python3
"""
Test script for the progress bar API.

This script demonstrates and tests the progress bar functionality with
a simple mesh-based simulation.
"""

import sys
import os
from pathlib import Path

# Add build directory to path
build_dir = os.path.join(os.path.dirname(__file__), "..", "build", "python")
if os.path.exists(build_dir):
    sys.path.insert(0, build_dir)

import samurai_python as sam
from samurai_python.utils import progress


def test_time_loop_basic():
    """Test basic time loop functionality."""
    print("\n" + "=" * 60)
    print("Test 1: Basic time loop without mesh")
    print("=" * 60)

    with progress.time_loop(Tf=1.0, dt=0.01) as pbar:
        while pbar.continue_loop():
            # Simulate some work
            import time
            time.sleep(0.01)
            pbar.advance_time(0.01)

    print("✓ Test 1 passed")


def test_time_loop_with_stats():
    """Test time loop with mesh statistics."""
    print("\n" + "=" * 60)
    print("Test 2: Time loop with mesh statistics")
    print("=" * 60)

    # Create a simple mesh
    box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
    config = sam.MeshConfig2D()
    config.min_level = 2
    config.max_level = 5

    mesh = sam.MRMesh2D(box, config)
    u = sam.field.zeros(mesh, "u")

    with progress.time_loop(Tf=0.1, dt=0.01, desc="Advection") as pbar:
        iteration = 0
        while pbar.continue_loop():
            # Simulate mesh adaptation
            if iteration % 5 == 0:
                MRadaptation = sam.make_MRAdapt(u)
                mra_config = sam.MRAConfig()
                mra_config.epsilon = 1e-3
                mra_config.regularity = 1
                MRadaptation(mra_config)

            # Update progress with mesh statistics
            pbar.advance_time(0.01)
            pbar.update_stats(mesh=u.mesh)

            iteration += 1

    print("✓ Test 2 passed")


def test_mesh_adaptation():
    """Test mesh adaptation context manager."""
    print("\n" + "=" * 60)
    print("Test 3: Mesh adaptation tracking")
    print("=" * 60)

    # Create mesh
    box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
    config = sam.MeshConfig2D()
    config.min_level = 2
    config.max_level = 6

    mesh = sam.MRMesh2D(box, config)
    u = sam.field.zeros(mesh, "u")

    # Initialize with some data
    def init_fn(cell):
        cx, cy = cell.center()
        if (cx - 0.5)**2 + (cy - 0.5)**2 < 0.1**2:
            u[cell.index] = 1.0

    sam.for_each_cell(mesh, init_fn)

    # Test mesh adaptation with progress tracking
    with progress.mesh_adaptation(mesh, desc="Adapting mesh") as stats:
        MRadaptation = sam.make_MRAdapt(u)
        mra_config = sam.MRAConfig()
        mra_config.epsilon = 1e-3
        mra_config.regularity = 1
        MRadaptation(mra_config)

    print("✓ Test 3 passed")


def test_iteration_loop():
    """Test simple iteration loop."""
    print("\n" + "=" * 60)
    print("Test 4: Iteration loop")
    print("=" * 60)

    with progress.iteration(total=50, desc="Processing items") as pbar:
        for i in range(50):
            # Simulate work
            import time
            time.sleep(0.01)
            pbar.update()

            # Update with custom stats every 10 iterations
            if i % 10 == 0:
                pbar.set_postfix(item=i, value=i*i)

    print("✓ Test 4 passed")


def test_mesh_statistics():
    """Test mesh statistics computation."""
    print("\n" + "=" * 60)
    print("Test 5: Mesh statistics")
    print("=" * 60)

    # Create mesh
    box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
    config = sam.MeshConfig2D()
    config.min_level = 3
    config.max_level = 5

    mesh = sam.MRMesh2D(box, config)

    # Compute statistics
    stats = progress.compute_mesh_stats(mesh)
    print(f"Computed stats: {stats}")

    # Test MeshStatistics class
    mesh_stats = progress.MeshStatistics(enable_level_breakdown=True)
    mesh_stats.update(mesh)

    print(f"Summary: {mesh_stats.get_summary()}")
    print(f"Level breakdown: {mesh_stats.get_level_breakdown()}")
    print(f"Repr: {repr(mesh_stats)}")

    print("✓ Test 5 passed")


def test_disabled_progress():
    """Test progress bar with display disabled."""
    print("\n" + "=" * 60)
    print("Test 6: Disabled progress bar")
    print("=" * 60)

    with progress.time_loop(Tf=0.1, dt=0.01, disable=True) as pbar:
        iteration = 0
        while pbar.continue_loop():
            import time
            time.sleep(0.01)
            pbar.advance_time(0.01)
            iteration += 1

    print(f"Completed {iteration} iterations (progress bar disabled)")
    print("✓ Test 6 passed")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Samurai Python Progress Bar API Test Suite")
    print("=" * 60)

    tests = [
        test_time_loop_basic,
        test_time_loop_with_stats,
        test_mesh_adaptation,
        test_iteration_loop,
        test_mesh_statistics,
        test_disabled_progress,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ Test failed: {test.__name__}")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
