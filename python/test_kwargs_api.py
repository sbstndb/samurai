#!/usr/bin/env python3
"""
Quick test for new kwargs API improvements.

Tests MeshConfig and MRAConfig constructor with keyword arguments.
"""

import sys
import os

# Add build directory to path
build_dir = os.path.join(os.path.dirname(__file__), "..", "build", "python")
if os.path.exists(build_dir):
    sys.path.insert(0, build_dir)

import samurai_python as sam

print("=" * 70)
print("Testing new kwargs API for MeshConfig and MRAConfig")
print("=" * 70)

# Test 1: MeshConfig with kwargs
print("\n[Test 1] MeshConfig2D with kwargs")
print("-" * 50)

try:
    # Old way (still works)
    config_old = sam.MeshConfig2D()
    config_old.min_level = 4
    config_old.max_level = 10
    print(f"✓ Old way works: min={config_old.min_level}, max={config_old.max_level}")

    # New way (with kwargs)
    config_new = sam.MeshConfig2D(min_level=4, max_level=10)
    print(f"✓ New way works: min={config_new.min_level}, max={config_new.max_level}")

    # New way with more options
    config_full = sam.MeshConfig2D(
        min_level=2,
        max_level=8,
        periodic=True,
        disable_minimal_ghost_width=True
    )
    print(f"✓ Full kwargs: min={config_full.min_level}, max={config_full.max_level}")
    print(f"  periodic={config_full.get_periodic(0)}")

    # Per-direction periodicity
    config_per_dir = sam.MeshConfig2D(
        min_level=0,
        max_level=5,
        periodic_per_direction=[True, False]
    )
    print(f"✓ Per-direction periodic: [{config_per_dir.get_periodic(0)}, {config_per_dir.get_periodic(1)}]")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 2: MRAConfig with kwargs
print("\n[Test 2] MRAConfig with kwargs")
print("-" * 50)

try:
    # Old way (still works)
    mra_old = sam.MRAConfig()
    mra_old.epsilon = 2e-4
    mra_old.regularity = 2.0
    print(f"✓ Old way works: eps={mra_old.epsilon:.2e}, reg={mra_old.regularity}")

    # New way (with kwargs)
    mra_new = sam.MRAConfig(epsilon=2e-4, regularity=2.0)
    print(f"✓ New way works: eps={mra_new.epsilon:.2e}, reg={mra_new.regularity}")

    # All kwargs
    mra_full = sam.MRAConfig(epsilon=1e-5, regularity=1.5, relative_detail=True)
    print(f"✓ Full kwargs: eps={mra_full.epsilon:.2e}, reg={mra_full.regularity}, rel={mra_full.relative_detail}")

    # Property setting (not chaining in Python - that's C++ style)
    mra_props = sam.MRAConfig()
    mra_props.epsilon = 3e-4
    mra_props.regularity = 0.5
    print(f"✓ Property setting: eps={mra_props.epsilon:.2e}, reg={mra_props.regularity}")

    # Kwargs in constructor is the Pythonic way!
    mra_kwargs = sam.MRAConfig(epsilon=3e-4, regularity=0.5)
    print(f"✓ Constructor kwargs: eps={mra_kwargs.epsilon:.2e}, reg={mra_kwargs.regularity}")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Code reduction example
print("\n[Test 3] Code reduction comparison")
print("-" * 50)

print("OLD way (4 lines):")
print("  config = sam.MeshConfig2D()")
print("  config.min_level = 4")
print("  config.max_level = 10")
print("  config.disable_minimal_ghost_width()")
print()
print("NEW way (1 line):")
print("  config = sam.MeshConfig2D(min_level=4, max_level=10, disable_minimal_ghost_width=True)")

print("\n" + "=" * 70)
print("All tests completed! ✨")
print("=" * 70)
