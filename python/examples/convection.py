#!/usr/bin/env python3
"""
Finite Volume example for the linear convection equation in 2D.

This demo demonstrates:
- 2D adaptive mesh refinement (AMR) with multiresolution analysis
- WENO5 convection operator for high-order accuracy
- TVD-RK3 time stepping scheme
- Optional obstacle using DomainBuilder
- Rectangle initial condition
- Real-time matplotlib visualization

The linear convection equation: du/dt + a·∇u = 0
with constant velocity a and a rectangular initial condition.

Usage:
    python convection.py                      # Run with default parameters
    python convection.py --obstacle           # Enable obstacle in domain
    python convection.py --tf 1.0 --cfl 0.5   # Custom time parameters
    python convection.py --velocity 1.0 0.5   # Custom velocity vector
    python convection.py --no-plot            # Disable real-time matplotlib visualization

Options:
    --obstacle            Enable rectangular obstacle in domain
    --velocity VX VY      Velocity vector components (default: 1.0 -1.0)
    --tf FLOAT            Final time (default: 2.0)
    --cfl FLOAT           CFL condition (default: 0.95)
    --no-plot             Disable real-time matplotlib visualization
"""

import sys
import os
import argparse
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
import samplotlib as svmpl
from samurai_python.utils import progress


def init_rectangle(u):
    """Initialize field with a rectangle condition.

    Rectangle: [-0.8, -0.3] x [0.3, 0.8]

    Args:
        u: ScalarField to initialize
    """
    def init_cell(cell):
        cx, cy = cell.center()
        # Rectangle condition
        if -0.8 <= cx <= -0.3 and 0.3 <= cy <= 0.8:
            u[cell.index] = 1.0
        else:
            u[cell.index] = 0.0

    sam.algorithms.for_each_cell(u.mesh, init_cell)


def main():
    """Main simulation function."""

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Linear convection equation 2D with AMR and WENO5",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--obstacle",
        action="store_true",
        help="Enable rectangular obstacle in the domain"
    )
    parser.add_argument(
        "--velocity",
        type=float,
        nargs=2,
        default=[1.0, -1.0],
        metavar=("VX", "VY"),
        help="Velocity vector components"
    )
    parser.add_argument("--tf", type=float, default=2.0, help="Final time")
    parser.add_argument("--cfl", type=float, default=0.95, help="CFL condition")
    parser.add_argument("--no-plot", action="store_true", help="Disable real-time matplotlib visualization")
    args = parser.parse_args()

    # ============================================================
    # Simulation parameters
    # ============================================================
    velocity = list(args.velocity)
    use_obstacle = args.obstacle
    Tf = args.tf
    cfl = args.cfl
    enable_realtime_viz = not args.no_plot

    # Mesh parameters
    min_level = 1
    max_level = 4
    max_stencil_size = 6  # Required for WENO5

    # Multiresolution parameters
    epsilon = 1e-3
    regularity = 1.0

    # Output parameters
    output_path = Path("./convection_2d_results")
    filename = "convection_2d"
    nfiles = 10

    print("=" * 70)
    print("Linear Convection 2D - WENO5 with TVD-RK3")
    print("=" * 70)
    if use_obstacle:
        print(f"Domain: [-1, 1] x [-1, 1] with obstacle [0, 0.4] x [0, 0.4]")
    else:
        print(f"Domain: [-1, 1] x [-1, 1]")
    print(f"Velocity: ({velocity[0]}, {velocity[1]})")
    print(f"Initial condition: rectangle [-0.8, -0.3] x [0.3, 0.8]")
    print(f"CFL: {cfl}")
    print(f"Final time: {Tf}")
    print(f"Mesh levels: min={min_level}, max={max_level}")
    print(f"Max stencil size: {max_stencil_size} (for WENO5)")
    print(f"MR adaptation: epsilon={epsilon}, regularity={regularity}")
    print(f"Output: {output_path}/{filename}_*.h5")
    print("=" * 70)
    print()

    # ============================================================
    # Domain with optional obstacle using DomainBuilder
    # ============================================================

    # ============================================================
    # Mesh configuration
    # ============================================================

    # Create mesh configuration (needed for max_stencil_size with WENO5)
    config = sam.config.make(2)
    config.min_level = min_level
    config.max_level = max_level
    config.max_stencil_size = max_stencil_size

    if use_obstacle:
        # Create a domain builder with an initial box
        domain = sam.geometry.DomainBuilder2D([-1.0, -1.0], [1.0, 1.0])

        # Remove a rectangular region to create an obstacle
        # Obstacle: [0.0, 0.4] x [0.0, 0.4] (in the flow path)
        domain.remove([0.0, 0.0], [0.4, 0.4])

        print(f"Domain created with DomainBuilder:")
        print(f"  Added boxes: {len(domain.added_boxes)}")
        print(f"  Removed boxes (obstacles): {len(domain.removed_boxes)}")
        print()

        # Create mesh from DomainBuilder
        mesh = sam.mesh.make(domain, config)
    else:
        # Simple box domain
        box = sam.geometry.box([-1.0, -1.0], [1.0, 1.0])
        mesh = sam.mesh.make(box, config)

    # Create fields for TVD-RK3 time stepping
    u = sam.field.zeros(mesh, "u")
    u1 = sam.field.zeros(mesh, "u1")
    u2 = sam.field.zeros(mesh, "u2")
    unp1 = sam.field.zeros(mesh, "unp1")

    # ============================================================
    # Initialize with rectangle condition
    # ============================================================
    print("Initializing field with rectangle condition...")
    init_rectangle(u)

    # ============================================================
    # Mesh adaptation setup
    # ============================================================
    MRadaptation = sam.adaptation.make_MRAdapt(u)
    mra_config = sam.config.MRAConfig()
    mra_config.epsilon = epsilon
    mra_config.regularity = regularity

    print("Performing initial mesh adaptation...")
    MRadaptation(mra_config)

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Save initial condition
    it = 0
    save_filename = f"{filename}_init"
    print(f"Saving initial condition to {output_path / save_filename}")
    sam.save(f"{output_path}/{save_filename}", u)

    # ============================================================
    # Time stepping setup
    # ============================================================
    min_cell_length = mesh.min_cell_length
    sum_velocities = sum(abs(v) for v in velocity)
    dt = cfl * min_cell_length / sum_velocities

    print(f"\nTime stepping parameters:")
    print(f"  Min cell length: {min_cell_length:.6e}")
    print(f"  Time step: {dt:.6e}")
    print(f"  Expected iterations: ~{int(Tf/dt)}")
    print()

    dt_save = Tf / nfiles
    nsave = 0

    # ============================================================
    # Real-time visualization setup
    # ============================================================
    plotter = None
    plot_interval = 1  # Update plot every iteration
    if enable_realtime_viz:
        plt.ion()  # Interactive mode
        fig, ax = plt.subplots(figsize=(8, 8))
        plotter = svmpl.FieldPlotter(u, ax=ax, cmap='RdBu_r',
                                     vmin=0.0, vmax=1.0, show_mesh=True)
        plt.pause(0.01)

    # ============================================================
    # Main time loop with TVD-RK3
    # ============================================================
    t = 0.0
    nt = 0

    print(f"{'Iter':>6} {'Time':>12} {'dt':>12} {'Cells':>10}")
    print("-" * 42)

    with progress.time_loop(Tf, dt, desc="Convection 2D") as pbar:
        while True:
            # 1. Adapt mesh FIRST (before computing fluxes)
            with progress.mesh_adaptation(mesh):
                MRadaptation(mra_config)

            # 2. Resize ALL fields after mesh adaptation
            u1.resize()
            u2.resize()
            unp1.resize()

            # 3. CRITICAL: Update ghost cells BEFORE computing fluxes
            # This is required for correct WENO5 stencil operations
            sam.adaptation.update_ghost_mr(u)

            # 4. Advance time and check if simulation is complete
            pbar.advance_time(dt)
            pbar.update_stats(mesh=mesh)
            if not pbar.continue_loop():
                break

            # 5. NOW safe to compute fluxes with valid ghost values
            # ========================================================
            # TVD-RK3 (SSPRK3) time stepping scheme
            # ========================================================
            # Scheme:
            #   u1   = u - dt * conv(u)
            #   u2   = 3/4*u + 1/4*(u1 - dt*conv(u1))
            #   unp1 = 1/3*u + 2/3*(u2 - dt*conv(u2))

            # Stage 1: u1 = u - dt * conv(u)
            flux1 = sam.operators.convection_weno5(u, velocity)
            u1.assign(u - dt * flux1)

            # Stage 2: u2 = 3/4*u + 1/4*(u1 - dt*conv(u1))
            flux2 = sam.operators.convection_weno5(u1, velocity)
            u2.assign((3.0 / 4.0) * u + (1.0 / 4.0) * (u1 - dt * flux2))

            # Stage 3: unp1 = 1/3*u + 2/3*(u2 - dt*conv(u2))
            flux3 = sam.operators.convection_weno5(u2, velocity)
            unp1.assign((1.0 / 3.0) * u + (2.0 / 3.0) * (u2 - dt * flux3))

            # Swap u and unp1 (u becomes the new solution)
            u, unp1 = unp1, u
            u.name = "u"

            # ========================================================
            # Update visualization
            # ========================================================
            if enable_realtime_viz and plotter is not None and pbar.iteration % plot_interval == 0:
                plotter.update(u, title=f"Convection 2D - t={pbar.t:.3f}, cells={mesh.nb_cells}")
                plt.pause(0.001)  # Small pause to allow GUI update

            # ========================================================
            # Save output
            # ========================================================
            if pbar.t >= (nsave + 1) * dt_save:
                suffix = f"_ite_{nsave}" if nfiles > 1 else ""
                save_filename = f"{filename}{suffix}"
                sam.save(f"{output_path}/{save_filename}", u)
                nsave += 1

                # Print progress
                print(f"{pbar.iteration:6d} {pbar.t:12.6f} {dt:12.6f} {mesh.nb_cells:10d}")

    print()
    print("=" * 70)
    print("Simulation complete!")
    print(f"  Final time: {pbar.t:.6f}")
    print(f"  Time steps: {pbar.iteration}")
    print(f"  Output files: {nsave + 1} (including initial)")
    print()
    print(f"Generated files in {output_path.absolute()}:")
    print(f"  - {filename}_init.h5/.xdmf    (initial condition)")
    print(f"  - {filename}_ite_*.h5/.xdmf   (time series)")
    print()
    print(f"To visualize in Paraview:")
    print(f"  paraview {output_path.absolute() / filename}_ite_0.xdmf")
    print()

    if use_obstacle:
        print(f"Note: The obstacle region [0, 0.4] x [0, 0.4] is excluded from the mesh.")
        print(f"      The flow will go around this obstacle.")
        print()

    # Keep matplotlib figure open if visualization was enabled
    if enable_realtime_viz and plotter is not None:
        plt.ioff()  # Turn off interactive mode
        print(f"\nReal-time visualization complete.")
        print(f"Close the plot window to exit...")
        plt.show()

    print("=" * 70)


if __name__ == "__main__":
    main()
