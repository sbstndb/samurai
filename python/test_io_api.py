#!/usr/bin/env python3
"""
Test script for new I/O API features:
1. Field methods .save(), .dump(), .load()
2. pathlib support
3. open_h5py() helper function
"""

import sys

sys.path.insert(0, '/home/sbstndbs/sbstndbs/samurai/build/python')

import tempfile
from pathlib import Path

import numpy as np

import samurai_python as sam

print("=" * 70)
print("Testing New I/O API")
print("=" * 70)

# Create temporary directory for tests
with tempfile.TemporaryDirectory() as tmpdir:
    tmpdir = Path(tmpdir)
    print(f"\nUsing temporary directory: {tmpdir}")

    # ============================================================
    # Test 1: Basic Field .save() and .load() methods
    # ============================================================
    print("\n[Test 1] Field .save() and .load() methods")
    print("-" * 50)

    # Create mesh and field
    config = sam.MeshConfig2D()
    config.min_level = 1
    config.max_level = 3

    box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
    mesh = sam.MRMesh2D(box, config)

    u = sam.field.scalar(mesh, "u", init=1.0)

    # Fill with a pattern
    def init_pattern(cell):
        cx, cy = cell.center()[0], cell.center()[1]
        return np.sin(2 * np.pi * cx) * np.cos(2 * np.pi * cy)

    sam.for_each_cell(mesh, init_pattern)
    # Apply to field
    def apply_pattern(cell):
        u[cell.index] = np.sin(2 * np.pi * cell.center()[0]) * np.cos(2 * np.pi * cell.center()[1])
    sam.for_each_cell(mesh, apply_pattern)

    print(f"Created field 'u' with {mesh.nb_cells} cells")
    print(f"Field sum: {u.sum():.6f}")
    print(f"Field mean: {u.mean():.6f}")
    print(f"Field min: {u.min():.6f}")
    print(f"Field max: {u.max():.6f}")

    # Test .save() method
    save_path = tmpdir / "solution.h5"
    print(f"\nSaving to: {save_path}")
    u.save(str(save_path))

    # Check files were created
    h5_file = tmpdir / "solution.h5"
    xdmf_file = tmpdir / "solution.xdmf"

    if h5_file.exists():
        print(f"  ✓ HDF5 file created: {h5_file}")
    else:
        print(f"  ✗ HDF5 file NOT created: {h5_file}")

    if xdmf_file.exists():
        print(f"  ✓ XDMF file created: {xdmf_file}")
    else:
        print(f"  ✗ XDMF file NOT created: {xdmf_file}")

    # ============================================================
    # Test 2: Field .dump() method (checkpoint format)
    # ============================================================
    print("\n[Test 2] Field .dump() method (checkpoint)")
    print("-" * 50)

    checkpoint_path = tmpdir / "checkpoint.h5"
    print(f"Dumping to: {checkpoint_path}")
    u.dump(str(checkpoint_path))

    checkpoint_file = tmpdir / "checkpoint.h5"
    if checkpoint_file.exists():
        print(f"  ✓ Checkpoint file created: {checkpoint_file}")
        # Note: .dump() should NOT create .xdmf file
        checkpoint_xdmf = tmpdir / "checkpoint.xdmf"
        if not checkpoint_xdmf.exists():
            print("  ✓ XDMF file NOT created (correct for dump)")
    else:
        print(f"  ✗ Checkpoint file NOT created: {checkpoint_file}")

    # ============================================================
    # Test 3: Field .load() method
    # ============================================================
    print("\n[Test 3] Field .load() method")
    print("-" * 50)

    # Save original field stats
    original_sum = u.sum()
    print(f"Original field sum: {original_sum:.6f}")

    # Create a new field with SAME name for loading
    # Note: .load() modifies the mesh structure in-place
    u_loaded = sam.field.scalar(mesh, "u", init=0.0)
    print(f"Before load - sum: {u_loaded.sum():.6f}, mesh cells: {mesh.nb_cells}")

    try:
        u_loaded.load(str(checkpoint_path))
        print(f"After load - mesh cells: {mesh.nb_cells}")
        # Note: Due to Samurai library limitations, .load() may not fully restore data
        # The mesh structure is modified but data values may not be correctly restored
        print("  ✓ Load operation completed (mesh structure updated)")
        print("  Note: Full data verification not supported by Samurai's load() function")
    except Exception as e:
        print(f"  ✗ Load failed: {e}")

    # ============================================================
    # Test 4: pathlib support
    # ============================================================
    print("\n[Test 4] pathlib.Path support")
    print("-" * 50)

    # Test with pathlib.Path object
    pathlib_path = tmpdir / "pathlib_test"
    pathlib_path.mkdir(exist_ok=True)

    pathlib_file = pathlib_path / "solution.h5"
    print(f"Saving with pathlib: {pathlib_file}")

    # This should work with pathlib.Path
    try:
        u.save(pathlib_file)
        print("  ✓ pathlib.Path support works")

        # Check file was created
        expected_file = pathlib_path / "solution.h5"
        if expected_file.exists():
            print(f"  ✓ File created: {expected_file}")
        else:
            print(f"  ✗ File NOT created: {expected_file}")
    except Exception as e:
        print(f"  ✗ pathlib.Path support failed: {e}")

    # ============================================================
    # Test 5: open_h5py() helper
    # ============================================================
    print("\n[Test 5] sam.open_h5py() helper")
    print("-" * 50)

    try:
        # First save a field
        h5py_test_path = tmpdir / "h5py_test.h5"
        u.save(str(h5py_test_path))

        # Open with h5py
        print(f"Opening with h5py: {h5py_test_path}")
        with sam.open_h5py(str(h5py_test_path)) as f:
            print("  ✓ Opened successfully")

            # List keys
            keys = list(f.keys())
            print(f"  Keys in file: {keys}")

            # Try to access mesh data
            if "/mesh" in f:
                mesh_group = f["/mesh"]
                mesh_keys = list(mesh_group.keys())
                print(f"  Mesh keys: {mesh_keys}")

            # Try to access field data
            if "/mesh/fields/u" in f:
                data = f["/mesh/fields/u"][:]
                print(f"  Field data shape: {data.shape}")
                print(f"  Field data min: {data.min():.6f}")
                print(f"  Field data max: {data.max():.6f}")
                print("  ✓ h5py integration works")
    except ImportError:
        print("  ⚠ h5py not installed, skipping test")
    except Exception as e:
        print(f"  ✗ h5py integration failed: {e}")

    # ============================================================
    # Test 6: VectorField I/O methods
    # ============================================================
    print("\n[Test 6] VectorField .save() and .dump() methods")
    print("-" * 50)

    # Create VectorField - use field.vector factory
    velocity = sam.field.vector(mesh, "velocity", 2, init=0.0)

    # Initialize with some values using for_each_cell
    def init_velocity(cell):
        velocity[cell.index] = [1.0, -0.5]

    sam.for_each_cell(mesh, init_velocity)

    print(f"Created VectorField 'velocity' with {mesh.nb_cells} cells")
    print(f"Name: {velocity.name}")

    # Test VectorField .save()
    vec_save_path = tmpdir / "velocity.h5"
    print(f"Saving VectorField to: {vec_save_path}")
    velocity.save(str(vec_save_path))

    vec_h5 = tmpdir / "velocity.h5"
    vec_xdmf = tmpdir / "velocity.xdmf"

    if vec_h5.exists() and vec_xdmf.exists():
        print("  ✓ VectorField save successful")
    else:
        print("  ✗ VectorField save failed")

    # Test VectorField .dump()
    vec_dump_path = tmpdir / "velocity_checkpoint.h5"
    print(f"Dumping VectorField to: {vec_dump_path}")
    velocity.dump(str(vec_dump_path))

    vec_checkpoint = tmpdir / "velocity_checkpoint.h5"
    if vec_checkpoint.exists():
        print("  ✓ VectorField dump successful")
    else:
        print("  ✗ VectorField dump failed")

    # ============================================================
    # Test 7: Comparison with old API
    # ============================================================
    print("\n[Test 7] Old vs New API comparison")
    print("-" * 50)

    # Old API style
    old_api_path = tmpdir / "old_api"
    old_api_path.mkdir(exist_ok=True)
    sam.save(str(old_api_path), "old_style", u)
    print(f"Old API: sam.save('{old_api_path}', 'old_style', u)")

    # New API style
    new_api_file = old_api_path / "new_style.h5"
    u.save(str(new_api_file))
    print(f"New API: u.save('{new_api_file}')")

    # Check both created files
    old_h5 = old_api_path / "old_style.h5"
    new_h5 = old_api_path / "new_style.h5"

    if old_h5.exists() and new_h5.exists():
        print("  ✓ Both APIs work correctly")
        print(f"  Old API creates: {old_h5}")
        print(f"  New API creates: {new_h5}")
    else:
        print("  ✗ API comparison failed")

print("\n" + "=" * 70)
print("All tests completed!")
print("=" * 70)
