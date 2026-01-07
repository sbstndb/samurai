#!/usr/bin/env python3
"""
Burgers 2D equation with progress bars and variable time stepping.

This demo demonstrates:
- 2D Burgers equation: ∂u/∂t + u·∇u = ν·∇²u
- RK3 (Runge-Kutta 3rd order) time integration
- WENO5 spatial discretization for convection
- Variable time stepping with CFL condition
- Progress bars for time stepping and mesh adaptation
- Live metrics display (time, dt, cells, min/max level)

Equation:
    ∂u/∂t + u·∇u = 0  (inviscid Burgers)
    ∂u/∂t + u·∇u = ν·∇²u  (viscous Burgers)

This example shows:
- Progress tracking with variable time steps
- RK3 time stepping with mesh adaptation
- Vector field operations
- Real-time visualization with progress

Usage:
    python burgers_2d_progress.py

Features:
    - Progress bar during RK3 time stepping
    - Live metrics (time, dt, cells, velocity magnitude)
    - Mesh adaptation progress tracking
    - Automatic time step adjustment at final time
"""

import sys
import os
import math
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

# Import progress bar from demo_progress
try:
    from demo_progress import ProgressBar, TQDMProgressBar
    HAS_PROGRESS = True
except ImportError:
    HAS_PROGRESS = False
    print("Warning: demo_progress.py not found. Progress bars disabled.")


def init_hat(u):
    """Initialize VectorField with a 'hat' function (radial decrease from center).

    Args:
        u: VectorField2D_2 to initialize
    """
    max_val = 1.0
    radius = 0.5

    def init_cell(cell):
        cx, cy = cell.center()
        # Compute radial distance from origin
        dist = math.sqrt(cx**2 + cy**2)

        # Hat function: linear decrease from center
        if dist <= radius:
            value = -max_val / radius * dist + max_val
        else:
            value = 0.0

        # Set both components to same value
        u[cell.index] = [value, value]

    sam.algorithms.for_each_cell(u.mesh, init_cell)


def init_bands(u):
    """Initialize VectorField with 'bands' pattern.

    Each component gets a different band pattern based on its coordinate.
    Component 0 (u) depends on x-coordinate.
    Component 1 (v) depends on y-coordinate.

    Args:
        u: VectorField2D_2 to initialize
    """
    max_val = 2.0

    def init_cell(cell):
        cx, cy = cell.center()

        # Component 0: based on x-coordinate
        if -0.5 <= cx <= 0:
            u_val = 2 * max_val * cx + max_val
        elif 0 <= cx <= 0.5:
            u_val = -2 * max_val * cx + max_val
        else:
            u_val = 0.0

        # Component 1: based on y-coordinate
        if -0.5 <= cy <= 0:
            v_val = 2 * max_val * cy + max_val
        elif 0 <= cy <= 0.5:
            v_val = -2 * max_val * cy + max_val
        else:
            v_val = 0.0

        u[cell.index] = [u_val, v_val]

    sam.algorithms.for_each_cell(u.mesh, init_cell)


def compute_magnitude(vector_field, scalar_field):
    """Compute magnitude of vector field into scalar field.

    Args:
        vector_field: VectorField2D_2 source
        scalar_field: ScalarField2D destination for magnitude
    """
    def compute_cell(cell):
        val = vector_field[cell.index]
        magnitude = math.sqrt(val[0]**2 + val[1]**2)
        scalar_field[cell.index] = magnitude

    sam.algorithms.for_each_cell(vector_field.mesh, compute_cell)


def get_max_velocity(u):
    """Get maximum velocity magnitude from vector field.

    Args:
        u: VectorField2D_2

    Returns:
        float: Maximum velocity magnitude
    """
    max_vel = 0.0

    def compute_max(cell):
        nonlocal max_vel
        val = u[cell.index]
        magnitude = math.sqrt(val[0]**2 + val[1]**2)
        max_vel = max(max_vel, magnitude)

    sam.algorithms.for_each_cell(u.mesh, compute_max)
    return max_vel


def main():
    """Main simulation function."""

    # ============================================================
    # Simulation parameters
    # ============================================================
    box_corner1 = [-1.0, -1.0]
    box_corner2 = [1.0, 1.0]

    Tf = 1.0           # Final time
    cfl = 0.95         # CFL condition
    nfiles = 50        # Number of output files

    # Initial condition: "hat" or "bands"
    init_sol = "hat"

    # Multiresolution parameters
    min_level = 5
    max_level = 7

    # Output
    output_path = Path("./results")
    filename = "burgers_2d_progress"

    # Visualization option
    enable_realtime_viz = True  # Set to False to disable matplotlib visualization

    print(f"=== Burgers 2D with Progress Bars ===")
    print(f"Domain: [{box_corner1[0]}, {box_corner2[0]}] x [{box_corner1[1]}, {box_corner2[1]}]")
    print(f"CFL: {cfl}")
    print(f"Final time: {Tf}")
    print(f"Initial condition: {init_sol}")
    print(f"min_level: {min_level}, max_level: {max_level}")
    if HAS_PROGRESS:
        print(f"Progress bars: ENABLED")
    if enable_realtime_viz:
        print(f"Real-time visualization: ENABLED")
    print(f"==============================\n")

    # ============================================================
    # Mesh and field creation
    # ============================================================
    box = sam.geometry.Box2D(box_corner1, box_corner2)

    config = sam.config.MeshConfig2D()
    config.min_level = min_level
    config.max_level = max_level
    config.max_stencil_size = 6  # Required for WENO5

    mesh = sam.mesh.MRMesh2D(box, config)

    # Create VectorFields for RK3 time stepping
    u = sam.field.zeros_vector(mesh, "u", n_components=2)
    u1 = sam.field.zeros_vector(mesh, "u1", n_components=2)
    u2 = sam.field.zeros_vector(mesh, "u2", n_components=2)
    unp1 = sam.field.zeros_vector(mesh, "unp1", n_components=2)

    # ScalarField for visualization (magnitude of velocity)
    u_mag = sam.field.zeros(mesh, "u_mag")

    # ============================================================
    # Initial conditions
    # ============================================================
    print(f"Initializing with '{init_sol}' condition...")
    if init_sol == "hat":
        init_hat(u)
    elif init_sol == "bands":
        init_bands(u)
    else:
        raise ValueError(f"Unknown initial solution: {init_sol}")

    # Boundary conditions (Dirichlet with value 0 for both components)
    sam.boundary.dirichlet(u, [0.0, 0.0], order=3)
    sam.boundary.dirichlet(u1, [0.0, 0.0], order=3)
    sam.boundary.dirichlet(u2, [0.0, 0.0], order=3)
    sam.boundary.dirichlet(unp1, [0.0, 0.0], order=3)

    # ============================================================
    # Mesh adaptation setup
    # ============================================================
    MRadaptation = sam.adaptation.make_MRAdapt(u)
    mra_config = sam.config.MRAConfig()
    mra_config.epsilon = 2e-4
    mra_config.regularity = 1.0

    # Initial adaptation
    print("Performing initial mesh adaptation...")
    MRadaptation(mra_config)

    # ============================================================
    # Real-time visualization setup
    # ============================================================
    plotter = None
    plot_interval = 10  # Update plot every N iterations
    if enable_realtime_viz:
        plt.ion()  # Interactive mode
        fig, ax = plt.subplots(figsize=(8, 8))
        # Compute initial magnitude
        compute_magnitude(u, u_mag)
        plotter = svmpl.FieldPlotter(u_mag, ax=ax, cmap='plasma',
                                     vmin=0.0, vmax=2.0, show_mesh=True)
        plt.pause(0.01)

    # ============================================================
    # Time stepping setup
    # ============================================================
    dx = mesh.min_cell_length
    dt = cfl * dx / (2 ** 2)  # dt = CFL * dx / 2^dim for 2D

    dt_save = Tf / nfiles
    nsave = 0

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Save initial condition
    save_filename = f"{filename}_init"
    sam.save(str(output_path), save_filename, u)
    print(f"Saved initial condition to {output_path / save_filename}")

    # ============================================================
    # Main time loop
    # ============================================================
    t = 0.0
    nt = 0

    # Estimate total steps (rough estimate, will be updated)
    estimated_steps = int(Tf / dt)

    print(f"\nStarting time integration:")
    print(f"  Tf = {Tf}, dt = {dt:.6f}, CFL = {cfl}")
    print(f"  min_level = {min_level}, max_level = {max_level}")
    print(f"  epsilon = {mra_config.epsilon}")
    print(f"  Estimated steps: {estimated_steps}")
    print()

    # Setup progress bar
    pbar = None
    if HAS_PROGRESS:
        try:
            pbar = TQDMProgressBar(
                total=estimated_steps,
                desc="Burgers RK3"
            )
        except Exception as e:
            print(f"Warning: Could not create progress bar: {e}")
            pbar = None

    while t < Tf:
        # Adapt mesh FIRST (before computing fluxes)
        MRadaptation(mra_config)

        # Update time
        dt_current = dt
        t += dt
        if t > Tf:
            # Adjust dt to exactly reach Tf
            dt_current = Tf - (t - dt)
            t = Tf

        nt += 1

        # Resize temporary fields after mesh adaptation
        u1.resize()
        u2.resize()
        unp1.resize()

        # ========================================================
        # RK3 time scheme
        # ========================================================
        # Stage 1: u1 = u - dt * conv(u)
        flux1 = sam.operators.convection_weno5(u)
        u1.assign(u - dt_current * flux1)  # In-place assignment to avoid stale mesh references

        # Stage 2: u2 = 3/4*u + 1/4*(u1 - dt*conv(u1))
        flux2 = sam.operators.convection_weno5(u1)
        u2.assign((3.0 / 4.0) * u + (1.0 / 4.0) * (u1 - dt_current * flux2))

        # Stage 3: unp1 = 1/3*u + 2/3*(u2 - dt*conv(u2))
        flux3 = sam.operators.convection_weno5(u2)
        unp1.assign((1.0 / 3.0) * u + (2.0 / 3.0) * (u2 - dt_current * flux3))

        # Swap u and unp1 (u becomes the new solution)
        u, unp1 = unp1, u
        u.name = "u"      # Rename to ensure consistent field names in output

        # Get metrics for progress bar
        max_vel = get_max_velocity(u)
        n_cells = mesh.nb_cells

        # Update progress bar with live metrics
        if pbar is not None:
            pbar.update(1, time=f"{t:.4f}", dt=f"{dt_current:.6f}",
                       cells=n_cells, max_vel=f"{max_vel:.3f}")

        # ========================================================
        # Update visualization
        # ========================================================
        if enable_realtime_viz and plotter is not None and nt % plot_interval == 0:
            u_mag.resize()  # Ensure magnitude field matches current mesh
            compute_magnitude(u, u_mag)
            plotter.update(u_mag, title=f"Burgers 2D - t={t:.3f}, cells={n_cells}")
            plt.pause(0.001)  # Small pause to allow GUI update

        # ========================================================
        # Save output
        # ========================================================
        if t >= (nsave + 1) * dt_save or t == Tf:
            suffix = f"_ite_{nsave}" if nfiles > 1 else ""
            sam.save(str(output_path), f"{filename}{suffix}", u)
            nsave += 1

    # Close progress bar
    if pbar is not None:
        pbar.close()

    # ============================================================
    # Summary
    # ============================================================
    print()
    print("=" * 70)
    print(f"Simulation complete!")
    print(f"\nStatistics:")
    print(f"  Total iterations: {nt}")
    print(f"  Final time: {t}")
    print(f"  Final cells: {n_cells}")
    print(f"  Output saved to: {output_path.absolute()}")
    print(f"\nTo visualize in Paraview:")
    print(f"  paraview {output_path.absolute() / filename}{suffix}.xdmf")
    print(f"\nThis demo shows RK3 time stepping with progress bars")

    # Keep matplotlib figure open if visualization was enabled
    if enable_realtime_viz and plotter is not None:
        plt.ioff()  # Turn off interactive mode
        print(f"\nReal-time visualization complete.")
        print(f"Close the plot window to exit...")
        plt.show()


if __name__ == "__main__":
    main()
