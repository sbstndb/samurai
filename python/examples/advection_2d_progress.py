#!/usr/bin/env python3
"""
Finite Volume example for the advection equation in 2D with progress bars.

This demo demonstrates:
- 2D adaptive mesh refinement (AMR)
- Upwind operator for advection
- Mesh adaptation based on multiresolution analysis
- Time stepping with Euler method
- Progress bars for time stepping and mesh adaptation
- HDF5 output for Paraview visualization

The advection equation: du/dt + a·∇u = 0
with velocity a = (1, 1) and a circular initial condition.

Equivalent to: demos/FiniteVolume/advection_2d.cpp

Usage:
    python advection_2d_progress.py

Features:
    - Progress bar during time stepping
    - Live metrics display (time, cells, min/max level)
    - Progress indicator for mesh adaptation
    - Real-time matplotlib visualization (optional)
"""

import sys
import os
from pathlib import Path

# Add build directory to path for development
build_dir = os.path.join(os.path.dirname(__file__), "..", "..", "build", "python")
if os.path.exists(build_dir):
    sys.path.insert(0, build_dir)

# Add viz directory to path for matplotlib visualization
viz_dir = os.path.join(os.path.dirname(__file__), "..", "viz")
if os.path.exists(viz_dir):
    sys.path.insert(0, viz_dir)

import matplotlib.pyplot as plt
import samurai_python as sam
import samplotlib as svmpl  # matplotlib visualization

# Import progress bar from demo_progress
try:
    from demo_progress import ProgressBar, TQDMProgressBar
    HAS_PROGRESS = True
except ImportError:
    HAS_PROGRESS = False
    print("Warning: demo_progress.py not found. Progress bars disabled.")


def init_circular(u, center=(0.3, 0.3), radius=0.2):
    """Initialize field with a circular condition.

    Args:
        u: ScalarField to initialize
        center: Center of the circle (x, y)
        radius: Radius of the circle
    """
    def init_cell(cell):
        cx, cy = cell.center()
        dist_sq = (cx - center[0])**2 + (cy - center[1])**2
        if dist_sq < radius**2:
            u[cell.index] = 1.0
        else:
            u[cell.index] = 0.0

    sam.algorithms.for_each_cell(u.mesh, init_cell)


def get_mesh_stats(mesh):
    """Get mesh statistics for progress display.

    Args:
        mesh: MRMesh object

    Returns:
        dict: Dictionary with mesh statistics
    """
    stats = {}

    # Count cells by level
    level_counts = {}

    def count_by_level(cell):
        level = cell.level
        if level not in level_counts:
            level_counts[level] = 0
        level_counts[level] += 1

    sam.algorithms.for_each_cell(mesh, count_by_level)

    stats['cells'] = sum(level_counts.values())
    stats['min_level'] = min(level_counts.keys()) if level_counts else 0
    stats['max_level'] = max(level_counts.keys()) if level_counts else 0

    return stats


def main():
    """Main simulation function."""
    # Visualization option
    enable_realtime_viz = True  # Set to False to disable matplotlib visualization

    # ============================================================
    # Simulation parameters
    # ============================================================

    # Domain: [0, 1] x [0, 1]
    box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])

    # Velocity: a = (1, 1)
    velocity = [1.0, 1.0]

    # Time parameters
    Tf = 0.1        # Final time
    cfl = 0.5       # CFL condition

    # Output parameters
    output_path = Path("./results")
    filename = "FV_advection_2d_progress"

    print(f"=== Advection 2D Python Demo with Progress Bars ===")
    print(f"Domain: [0, 1] x [0, 1]")
    print(f"Velocity: ({velocity[0]}, {velocity[1]})")
    print(f"CFL: {cfl}")
    print(f"Final time: {Tf}")
    print(f"Output: {output_path}/{filename}_*.h5")
    if HAS_PROGRESS:
        print(f"Progress bars: ENABLED")
    if enable_realtime_viz:
        print(f"Real-time visualization: ENABLED")
    print(f"==============================\n")

    # ============================================================
    # Mesh configuration
    # ============================================================

    config = sam.config.MeshConfig2D()
    config.min_level = 4      # Minimum refinement level
    config.max_level = 10     # Maximum refinement level
    config.disable_minimal_ghost_width()  # Required for proper ghost cell handling

    # Create mesh and fields
    mesh = sam.mesh.MRMesh2D(box, config)
    u = sam.field.zeros(mesh, "u")      # Current solution
    unp1 = sam.field.zeros(mesh, "unp1")  # Next time step

    # ============================================================
    # Initialize with circular condition
    # ============================================================

    print("Initializing field with circular condition...")
    init_circular(u, center=(0.3, 0.3), radius=0.2)

    # Apply boundary conditions
    sam.boundary.dirichlet(u, 0.0)

    # ============================================================
    # Initial mesh adaptation
    # ============================================================

    MRadaptation = sam.adaptation.make_MRAdapt(u)
    mra_config = sam.config.MRAConfig()
    mra_config.epsilon = 2e-4
    mra_config.regularity = 1.0

    print("Performing initial mesh adaptation...")
    MRadaptation(mra_config)
    # Note: No ghost update needed here - will be done in loop before first use

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Save initial state
    it = 0
    print(f"Saving initial state to {output_path}/{filename}_init.h5")
    sam.save(str(output_path), f"{filename}_{it:05d}", u)

    # ============================================================
    # Time stepping
    # ============================================================

    # dt based on CFL condition
    min_cell_length = mesh.min_cell_length  # Property, not a method
    max_velocity = max(abs(v) for v in velocity)
    dt = cfl * min_cell_length / max_velocity

    print(f"Min cell length: {min_cell_length:.6e}")
    print(f"Time step: {dt:.6e}")

    t = 0.0
    nt = 0
    save_interval = int(Tf / (dt * 10))  # Save ~10 times
    if save_interval < 1:
        save_interval = 1

    # Estimate total steps
    estimated_steps = int(Tf / dt)

    # Setup real-time visualization if enabled
    plotter = None
    if enable_realtime_viz:
        plt.ion()  # Interactive mode
        fig, ax = plt.subplots(figsize=(8, 8))
        plotter = svmpl.FieldPlotter(u, ax=ax, cmap='RdBu_r', vmin=0.0, vmax=1.0, show_mesh=True)
        plt.pause(0.01)

    print(f"Starting time stepping...")
    print(f"Estimated {estimated_steps} time steps\n")

    # ============================================================
    # MAIN TIME LOOP WITH PROGRESS BAR
    # ============================================================

    pbar = None
    if HAS_PROGRESS:
        try:
            pbar = ProgressBar(
                total=estimated_steps,
                desc="Time stepping"
            )
        except Exception as e:
            print(f"Warning: Could not create progress bar: {e}")
            pbar = None

    while t < Tf:
        # 1. Adapt mesh FIRST (as in C++ version)
        MRadaptation(mra_config)

        # 2. Resize unp1 field after mesh adaptation (CRITICAL!)
        unp1.resize()

        # 3. Update BCs and ghost cells BEFORE computing fluxes
        sam.adaptation.update_ghost_mr(u)

        # 4. Update time
        t += dt
        nt += 1

        # 5. Apply upwind operator with FRESH ghost values
        upwind_result = sam.operators.upwind(u, velocity)

        # 6. Euler time step: unp1 = u - dt * upwind(a, u)
        unp1.assign(u - dt * upwind_result)  # In-place to avoid stale mesh references

        # 7. Swap arrays (efficient: no memory allocation)
        sam.swap_field_arrays_2d(u, unp1)

        # Update progress bar with live metrics
        if pbar is not None:
            mesh_stats = get_mesh_stats(mesh)
            pbar.update(1, metrics={
                "time": f"{t:.4f}",
                "cells": mesh_stats['cells'],
                "min_lvl": mesh_stats['min_level'],
                "max_lvl": mesh_stats['max_level']
            })

        # Print progress and save
        if nt % save_interval == 0 or t >= Tf:
            # Count cells by level
            level_counts = {}
            def count_by_level(cell):
                level = cell.level
                if level not in level_counts:
                    level_counts[level] = 0
                level_counts[level] += 1
            sam.algorithms.for_each_cell(mesh, count_by_level)

            min_level = min(level_counts.keys()) if level_counts else 0
            max_level = max(level_counts.keys()) if level_counts else 0
            n_cells = sum(level_counts.values())

            # Also print to console for record keeping
            print(f"  Step {nt:6d}: t = {t:12.6e}, cells = {n_cells:6d}, "
                  f"min_level = {min_level:2d}, max_level = {max_level:2d}")

            # Update real-time visualization
            if enable_realtime_viz and plotter is not None:
                plotter.update(u, title=f"Advection 2D - t={t:.3f}, cells={n_cells}")
                plt.pause(0.001)  # Small pause to allow GUI update

            # Save state
            sam.save(str(output_path), f"{filename}_{nt:05d}", u)

    # Close progress bar
    if pbar is not None:
        pbar.close()

    # ============================================================
    # Summary
    # ============================================================

    print("\n" + "=" * 70)
    print(f"Simulation complete!")
    print(f"\nStatistics:")
    print(f"  Final time: {t:.6e}")
    print(f"  Time steps: {nt}")
    print(f"  Output files: {nt // save_interval + 2}")
    print(f"\nGenerated files in {output_path}:")
    print(f"  - {filename}_*.h5/.xdmf     (time series)")
    print(f"\nTo visualize in Paraview:")
    print(f"  paraview {output_path}/{filename}_00000.xdmf")
    print(f"\nThis demo is equivalent to demos/FiniteVolume/advection_2d.cpp")

    # Keep matplotlib figure open if visualization was enabled
    if enable_realtime_viz and plotter is not None:
        plt.ioff()  # Turn off interactive mode
        print(f"\nReal-time visualization complete.")
        print(f"Close the plot window to exit...")
        plt.show()


if __name__ == "__main__":
    main()
