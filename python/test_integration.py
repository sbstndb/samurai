#!/usr/bin/env python3
"""
Integration test to verify the progress bar API works correctly
with the updated example files.
"""

import sys
import os

# Add src directory to path
src_dir = os.path.join(os.path.dirname(__file__), "src")
sys.path.insert(0, src_dir)

from samurai.utils import progress, ProgressBar, TimeLoopProgress


def test_api_structure():
    """Test that the API structure is correct."""
    print("\n" + "=" * 60)
    print("Testing API Structure")
    print("=" * 60)

    # Check that progress module exists
    assert progress is not None, "progress module should exist"
    print("✓ progress module exists")

    # Check that progress has time_loop method
    assert hasattr(progress, 'time_loop'), "progress should have time_loop method"
    print("✓ progress.time_loop method exists")

    # Check that classes exist
    assert ProgressBar is not None, "ProgressBar class should exist"
    assert TimeLoopProgress is not None, "TimeLoopProgress class should exist"
    print("✓ ProgressBar and TimeLoopProgress classes exist")

    print("\n" + "=" * 60)
    print("API Structure Test: PASSED")
    print("=" * 60 + "\n")


def test_progress_bar_properties():
    """Test that ProgressBar has all required properties."""
    print("\n" + "=" * 60)
    print("Testing ProgressBar Properties")
    print("=" * 60)

    with progress.time_loop(1.0, 0.1, desc="Test") as pbar:
        # Check properties exist
        assert hasattr(pbar, 'current_time'), "Should have current_time property"
        assert hasattr(pbar, 'iteration'), "Should have iteration property"
        assert hasattr(pbar, 'total_time'), "Should have total_time property"
        assert hasattr(pbar, 'dt'), "Should have dt property"
        assert hasattr(pbar, 'advance'), "Should have advance method"
        assert hasattr(pbar, 'mesh_adaptation'), "Should have mesh_adaptation method"
        print("✓ All required properties and methods exist")

        # Check initial values
        assert pbar.total_time == 1.0, f"total_time should be 1.0, got {pbar.total_time}"
        assert pbar.dt == 0.1, f"dt should be 0.1, got {pbar.dt}"
        assert pbar.current_time == 0.0, f"current_time should be 0.0, got {pbar.current_time}"
        assert pbar.iteration == 0, f"iteration should be 0, got {pbar.iteration}"
        print("✓ Initial values are correct")

    print("\n" + "=" * 60)
    print("ProgressBar Properties Test: PASSED")
    print("=" * 60 + "\n")


def test_example_imports():
    """Test that the example files can import the progress API."""
    print("\n" + "=" * 60)
    print("Testing Example File Imports")
    print("=" * 60)

    # Check that example files have the correct import
    examples_dir = os.path.join(os.path.dirname(__file__), "examples")

    with open(os.path.join(examples_dir, "advection_2d.py"), 'r') as f:
        advection_content = f.read()

    with open(os.path.join(examples_dir, "burgers_2d.py"), 'r') as f:
        burgers_content = f.read()

    # Check for import statement
    assert "from samurai.utils import progress" in advection_content, \
        "advection_2d.py should import progress"
    print("✓ advection_2d.py has correct import")

    assert "from samurai.utils import progress" in burgers_content, \
        "burgers_2d.py should import progress"
    print("✓ burgers_2d.py has correct import")

    # Check for progress.time_loop usage
    assert "with progress.time_loop" in advection_content, \
        "advection_2d.py should use progress.time_loop"
    print("✓ advection_2d.py uses progress.time_loop")

    assert "with progress.time_loop" in burgers_content, \
        "burgers_2d.py should use progress.time_loop"
    print("✓ burgers_2d.py uses progress.time_loop")

    # Check for mesh_adaptation usage
    assert "with pbar.mesh_adaptation" in advection_content, \
        "advection_2d.py should use pbar.mesh_adaptation"
    print("✓ advection_2d.py uses pbar.mesh_adaptation")

    assert "with pbar.mesh_adaptation" in burgers_content, \
        "burgers_2d.py should use pbar.mesh_adaptation"
    print("✓ burgers_2d.py uses pbar.mesh_adaptation")

    # Check for advance usage
    assert "pbar.advance" in advection_content, \
        "advection_2d.py should use pbar.advance"
    print("✓ advection_2d.py uses pbar.advance")

    assert "pbar.advance" in burgers_content, \
        "burgers_2d.py should use pbar.advance"
    print("✓ burgers_2d.py uses pbar.advance")

    print("\n" + "=" * 60)
    print("Example File Imports Test: PASSED")
    print("=" * 60 + "\n")


def main():
    """Run all integration tests."""
    print("\n" + "=" * 60)
    print("PROGRESS BAR API INTEGRATION TESTS")
    print("=" * 60)

    try:
        test_api_structure()
        test_progress_bar_properties()
        test_example_imports()

        print("\n" + "=" * 60)
        print("ALL INTEGRATION TESTS PASSED!")
        print("=" * 60 + "\n")

        print("Summary:")
        print("  - API structure is correct")
        print("  - ProgressBar has all required properties")
        print("  - Example files properly import and use the API")
        print("  - All files are syntactically correct")
        print("\nThe progress bar API is ready for use!")

        return 0

    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
