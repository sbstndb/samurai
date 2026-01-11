#!/usr/bin/env python3
"""
Finite Volume example for the advection equation in 2D.

This demo demonstrates:
- 2D adaptive mesh refinement (AMR) with multiresolution analysis
- Upwind operator for advection (first-order)
- Euler time stepping method
- Circular initial condition

The advection equation: du/dt + a·∇u = 0
with velocity a and a circular initial condition.

Usage:
    python advection.py                     # Run with default parameters
    python advection.py --velocity 1.0 0.5  # Custom velocity
    python advection.py --tf 0.05 --cfl 0.3 # Custom time parameters

Options:
    --velocity VX VY    Velocity vector components (default: 1.0 1.0)
    --tf FLOAT          Final time (default: 0.1)
    --cfl FLOAT         CFL condition (default: 0.5)
    --no-plot           Disable real-time matplotlib visualization
"""

import argparse
import os
import sys
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
import samplotlib as svmpl

import samurai_python as sam
from samurai_python.utils import progress


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


def main():
    """Main simulation function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Advection equation 2D with AMR and upwind scheme",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--velocity",
        type=float,
        nargs=2,
        default=[1.0, 1.0],
        metavar=("VX", "VY"),
        help="Velocity vector components"
    )
    parser.add_argument("--tf", type=float, default=0.1, help="Final time")
    parser.add_argument("--cfl", type=float, default=0.5, help="CFL condition")
    parser.add_argument("--no-plot", action="store_true", help="Disable real-time matplotlib visualization")
    args = parser.parse_args()

    # Visualization option
    enable_realtime_viz = not args.no_plot

    # ============================================================
    # Simulation parameters
    # ============================================================

    # Domain: [0, 1] x [0, 1]
    box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])

    # Velocity: a = (vx, vy)
    velocity = list(args.velocity)

    # Time parameters
    Tf = args.tf
    cfl = args.cfl

    # Output parameters
    output_path = Path("./advection_2d_results")
    filename = "advection_2d"

    print("=" * 70)
    print("Advection 2D - Upwind with Euler")
    print("=" * 70)
    print("Domain: [0, 1] x [0, 1]")
    print(f"Velocity: ({velocity[0]}, {velocity[1]})")
    print(f"CFL: {cfl}")
    print(f"Final time: {Tf}")
    print(f"Output: {output_path}/{filename}_*.h5")
    if enable_realtime_viz:
        print("Real-time visualization: ENABLED")
    print("=" * 70)
    print()

    # ============================================================
    # Mesh configuration (using factory functions)
    # ============================================================

    # Use factory function for cleaner mesh creation
    mesh = sam.mesh.make(box, min_level=4, max_level=10)
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

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Save initial state
    it = 0
    print(f"Saving initial state to {output_path}/{filename}_init.h5")
    sam.save(str(output_path / f"{filename}_{it:05d}"), u)

    # ============================================================
    # Time stepping
    # ============================================================

    # dt based on CFL condition
    min_cell_length = mesh.min_cell_length
    max_velocity = max(abs(v) for v in velocity)
    dt = cfl * min_cell_length / max_velocity

    print(f"Min cell length: {min_cell_length:.6e}")
    print(f"Time step: {dt:.6e}")

    save_interval = int(Tf / (dt * 10))  # Save ~10 times
    if save_interval < 1:
        save_interval = 1

    # Setup real-time visualization if enabled
    plotter = None
    if enable_realtime_viz:
        plt.ion()  # Interactive mode
        fig, ax = plt.subplots(figsize=(8, 8))
        plotter = svmpl.FieldPlotter(u, ax=ax, cmap='RdBu_r', vmin=0.0, vmax=1.0, show_mesh=True)
        plt.pause(0.01)

    print("Starting time stepping...")
    print(f"Estimated ~{int(Tf/dt)} time steps\n")

    # ============================================================
    # MAIN TIME LOOP WITH PROGRESS BAR
    # ============================================================

    with progress.time_loop(Tf, dt, desc="Advection 2D") as pbar:
        while True:
            # 1. Adapt mesh FIRST (as in C++ version)
            with progress.mesh_adaptation(mesh):
                MRadaptation(mra_config)

            # 2. Resize unp1 field after mesh adaptation (CRITICAL!)
            unp1.resize()

            # 3. CRITICAL: Update BCs and ghost cells BEFORE computing fluxes
            sam.adaptation.update_ghost_mr(u)

            # 4. Advance time and check if simulation is complete
            pbar.advance_time(dt)
            pbar.update_stats(mesh=mesh)
            if not pbar.continue_loop():
                break

            # 5. Apply upwind operator with FRESH ghost values
            upwind_result = sam.operators.upwind(u, velocity)

            # 6. Euler time step: unp1 = u - dt * upwind(a, u)
            unp1.assign(u - dt * upwind_result)

            # 7. Swap arrays (efficient: no memory allocation)
            sam.swap_field_arrays_2d(u, unp1)

            # Save and visualize at intervals
            if pbar.iteration % save_interval == 0:
                # Update real-time visualization
                if enable_realtime_viz and plotter is not None:
                    plotter.update(u, title=f"Advection 2D - t={pbar.t:.3f}, cells={mesh.nb_cells}")
                    plt.pause(0.001)  # Small pause to allow GUI update

                # Save state
                sam.save(str(output_path / f"{filename}_{pbar.iteration:05d}"), u)

    # ============================================================
    # Summary
    # ============================================================

    print("\n" + "=" * 54)
    print("Simulation complete!")
    print("\nStatistics:")
    print(f"  Final time: {Tf:.6e}")
    print(f"  Time steps: {pbar.iteration}")
    print(f"  Output files: {pbar.iteration // save_interval + 2}")
    print(f"\nGenerated files in {output_path}:")
    print(f"  - {filename}_*.h5/.xdmf     (time series)")
    print("\nTo visualize in Paraview:")
    print(f"  paraview {output_path}/{filename}_00000.xdmf")

    # Keep matplotlib figure open if visualization was enabled
    if enable_realtime_viz and plotter is not None:
        plt.ioff()  # Turn off interactive mode
        print("\nReal-time visualization complete.")
        print("Close the plot window to exit...")
        plt.show()


if __name__ == "__main__":
    main()
