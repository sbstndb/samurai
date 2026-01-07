#!/usr/bin/env python3
"""
Basic test for progress bar API (without requiring full C++ bindings).

This tests the progress bar functionality in isolation.
"""

import sys
import os
import time

# Add build directory to path
build_dir = os.path.join(os.path.dirname(__file__), "..", "build", "python")
if os.path.exists(build_dir):
    sys.path.insert(0, build_dir)

from samurai_python.utils import progress


def test_basic_time_loop():
    """Test basic time loop."""
    print("\n" + "=" * 60)
    print("Test 1: Basic time loop")
    print("=" * 60)

    with progress.time_loop(Tf=1.0, dt=0.1) as pbar:
        while pbar.continue_loop():
            time.sleep(0.05)
            pbar.advance_time(0.1)

    print("✓ Test 1 passed")


def test_iteration_loop():
    """Test iteration loop."""
    print("\n" + "=" * 60)
    print("Test 2: Iteration loop")
    print("=" * 60)

    with progress.iteration(total=20, desc="Processing") as pbar:
        for i in range(20):
            time.sleep(0.02)
            pbar.update()
            if i % 5 == 0:
                pbar.set_postfix(step=i)

    print("✓ Test 2 passed")


def test_disabled_progress():
    """Test with progress disabled."""
    print("\n" + "=" * 60)
    print("Test 3: Disabled progress bar")
    print("=" * 60)

    with progress.time_loop(Tf=0.5, dt=0.1, disable=True) as pbar:
        count = 0
        while pbar.continue_loop():
            time.sleep(0.05)
            pbar.advance_time(0.1)
            count += 1

    print(f"Completed {count} iterations (no progress bar shown)")
    print("✓ Test 3 passed")


def test_time_loop_custom_stats():
    """Test time loop with custom statistics."""
    print("\n" + "=" * 60)
    print("Test 4: Time loop with custom statistics")
    print("=" * 60)

    with progress.time_loop(Tf=0.5, dt=0.05) as pbar:
        iteration = 0
        while pbar.continue_loop():
            time.sleep(0.02)
            pbar.advance_time(0.05)

            # Update with custom stats
            if iteration % 5 == 0:
                pbar.update_stats(value=iteration**2, rate=iteration/10.0)

            iteration += 1

    print("✓ Test 4 passed")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Samurai Progress Bar API - Basic Tests")
    print("=" * 60)

    tests = [
        test_basic_time_loop,
        test_iteration_loop,
        test_disabled_progress,
        test_time_loop_custom_stats,
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
    print(f"Results: {passed}/{len(tests)} tests passed")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
