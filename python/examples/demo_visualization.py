#!/usr/bin/env python3
"""
Demonstration of Samurai AMR visualization with matplotlib.

This script demonstrates the new matplotlib-based visualization capabilities
for adaptive mesh refinement (AMR) fields in Samurai Python.

Features demonstrated:
- Static scalar field visualization
- Vector field (quiver) visualization
- Mesh structure visualization by refinement level
- Real-time plotting during simulation
"""

import sys
import os
import math
from pathlib import Path

# Add build directory to path for development
build_dir = os.path.join(os.path.dirname(__file__), "..", "..", "build", "python")
if os.path.exists(build_dir):
    sys.path.insert(0, build_dir)

# Add viz directory to path
viz_dir = os.path.join(os.path.dirname(__file__), "..", "viz")
if os.path.exists(viz_dir):
    sys.path.insert(0, viz_dir)

import matplotlib.pyplot as plt
import samurai_python as sam
import samplotlib as svmpl


def init_circular(u, center=(0.0, 0.0), radius=0.3):
    """Initialize field with a circular condition.

    Args:
        u: ScalarField2D to initialize
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


def demo_static_scalar_field():
    """Demonstrate static scalar field visualization."""
    print("=" * 70)
    print("Demo 1: Static Scalar Field Visualization")
    print("=" * 70)

    # Create mesh
    box = sam.geometry.Box2D([-1.0, -1.0], [1.0, 1.0])
    config = sam.config.MeshConfig2D()
    config.min_level = 3
    config.max_level = 6
    config.max_stencil_size = 6

    mesh = sam.mesh.MRMesh2D(box, config)
    u = sam.field.zeros(mesh, "u")

    # Initialize with circular pattern
    init_circular(u, center=(0.0, 0.0), radius=0.5)

    # Apply mesh adaptation
    MRadaptation = sam.adaptation.make_MRAdapt(u)
    mra_config = sam.config.MRAConfig()
    mra_config.epsilon = 1e-3
    mra_config.regularity = 1.0
    MRadaptation(mra_config)

    print(f"Mesh: {mesh.nb_cells} cells, levels {mesh.min_level}-{mesh.max_level}")

    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Field with colormap
    svmpl.plot_field(u, ax=axes[0], cmap='viridis', show_mesh=True)
    axes[0].set_title('Scalar Field with Mesh')
    svmpl.set_axes_equal(axes[0])

    # Plot 2: Field colored by refinement level
    svmpl.plot_field(u, ax=axes[1], show_level=True, show_mesh=False)
    axes[1].set_title('Colored by Refinement Level')
    svmpl.set_axes_equal(axes[1])

    # Plot 3: Mesh structure only
    svmpl.plot_mesh(u, ax=axes[2], by_level=True, linewidths=1.0)
    axes[2].set_title('Mesh Structure by Level')
    svmpl.set_axes_equal(axes[2])

    plt.tight_layout()
    output_file = Path("./results/viz_demo_scalar.png")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=150)
    print(f"Saved: {output_file}")
    plt.close()


def demo_vector_field():
    """Demonstrate vector field visualization."""
    print("\n" + "=" * 70)
    print("Demo 2: Vector Field Visualization")
    print("=" * 70)

    # Create mesh
    box = sam.geometry.Box2D([-1.0, -1.0], [1.0, 1.0])
    config = sam.config.MeshConfig2D()
    config.min_level = 3
    config.max_level = 5
    config.max_stencil_size = 6

    mesh = sam.mesh.MRMesh2D(box, config)

    # Create vector field with rotational velocity
    vel = sam.field.zeros_vector(mesh, "velocity", n_components=2)

    def init_velocity(cell):
        cx, cy = cell.center()
        # Rotational velocity: v = (-y, x)
        vel[cell.index] = [-cy, cx]

    sam.algorithms.for_each_cell(mesh, init_velocity)

    # Apply mesh adaptation
    MRadaptation = sam.adaptation.make_MRAdapt(vel)
    mra_config = sam.config.MRAConfig()
    mra_config.epsilon = 1e-3
    mra_config.regularity = 1.0
    MRadaptation(mra_config)

    print(f"Mesh: {mesh.nb_cells} cells, levels {mesh.min_level}-{mesh.max_level}")

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Quiver without mesh
    svmpl.plot_vector(vel, ax=axes[0], scale=20, cmap='plasma')
    axes[0].set_title('Vector Field (Quiver)')
    svmpl.set_axes_equal(axes[0])

    # Plot 2: Quiver with mesh overlay
    svmpl.plot_vector(vel, ax=axes[1], scale=20, show_mesh=True, cmap='plasma')
    axes[1].set_title('Vector Field with Mesh Overlay')
    svmpl.set_axes_equal(axes[1])

    plt.tight_layout()
    output_file = Path("./results/viz_demo_vector.png")
    plt.savefig(output_file, dpi=150)
    print(f"Saved: {output_file}")
    plt.close()


def demo_realtime_plotting():
    """Demonstrate real-time plotting during simulation."""
    print("\n" + "=" * 70)
    print("Demo 3: Real-Time Plotting During Simulation")
    print("=" * 70)

    # Create mesh
    box = sam.geometry.Box2D([0.0, 0.0], [1.0, 1.0])
    config = sam.config.MeshConfig2D()
    config.min_level = 3
    config.max_level = 6
    config.max_stencil_size = 6
    config.disable_minimal_ghost_width()

    mesh = sam.mesh.MRMesh2D(box, config)

    # Create fields
    u = sam.field.zeros(mesh, "u")
    unp1 = sam.field.zeros(mesh, "unp1")

    # Initialize with circular condition
    init_circular(u, center=(0.3, 0.3), radius=0.2)

    # Boundary conditions
    sam.boundary.dirichlet(u, 0.0)

    # Initial adaptation
    MRadaptation = sam.adaptation.make_MRAdapt(u)
    mra_config = sam.config.MRAConfig()
    mra_config.epsilon = 2e-4
    mra_config.regularity = 1.0
    MRadaptation(mra_config)

    # Setup real-time plotter
    fig, ax = plt.subplots(figsize=(8, 8))
    plotter = svmpl.FieldPlotter(u, ax=ax, cmap='RdBu_r',
                                 vmin=0.0, vmax=1.0, show_mesh=True)

    # Time stepping parameters
    velocity = [1.0, 1.0]
    min_cell_length = mesh.min_cell_length
    max_velocity = max(abs(v) for v in velocity)
    dt = 0.5 * min_cell_length / max_velocity

    Tf = 0.05
    t = 0.0
    nt = 0
    plot_interval = 5

    print(f"Starting simulation with {mesh.nb_cells} cells...")
    print(f"dt = {dt:.6e}, Tf = {Tf}")

    # Interactive mode
    plt.ion()
    plt.show()

    while t < Tf:
        # Adapt mesh
        MRadaptation(mra_config)

        # Resize fields after adaptation
        unp1.resize()

        # Update time
        t += dt
        nt += 1

        # Update ghost cells
        sam.adaptation.update_ghost_mr(u)

        # Compute flux and update
        upwind_result = sam.operators.upwind(velocity, u)
        unp1.assign(u - dt * upwind_result)

        # Swap
        sam.swap_field_arrays_2d(u, unp1)

        # Update plot periodically
        if nt % plot_interval == 0:
            plotter.update(u, title=f"Advection - t = {t:.4f}, cells = {mesh.nb_cells}")
            plotter.pause(0.01)
            print(f"  iteration {nt}: t = {t:.4f}, cells = {mesh.nb_cells}")

    # Final plot
    plotter.update(u, title=f"Final - t = {t:.4f}, cells = {mesh.nb_cells}")

    print(f"\nSimulation complete! {nt} iterations")

    # Save final state
    output_file = Path("./results/viz_demo_realtime_final.png")
    plt.savefig(output_file, dpi=150)
    print(f"Saved: {output_file}")

    # Keep plot open
    plt.ioff()
    plt.show()


def demo_multiple_initial_conditions():
    """Demonstrate visualization of different initial conditions."""
    print("\n" + "=" * 70)
    print("Demo 4: Multiple Initial Conditions")
    print("=" * 70)

    # Create mesh
    box = sam.geometry.Box2D([-1.0, -1.0], [1.0, 1.0])
    config = sam.config.MeshConfig2D()
    config.min_level = 3
    config.max_level = 6
    config.max_stencil_size = 6

    mesh = sam.mesh.MRMesh2D(box, config)

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # Test 1: Circle at origin
    u1 = sam.field.zeros(mesh, "circle")
    init_circular(u1, center=(0.0, 0.0), radius=0.4)
    MRadapt1 = sam.make_MRAdapt(u1)
    mra_config = sam.MRAConfig()
    mra_config.epsilon = 1e-3
    mra_config.regularity = 1.0
    MRadapt1(mra_config)
    svmpl.plot_field(u1, ax=axes[0, 0], cmap='viridis')
    axes[0, 0].set_title('Circle at Origin')
    svmpl.set_axes_equal(axes[0, 0])

    # Test 2: Offset circle
    u2 = sam.field.zeros(mesh, "offset_circle")
    init_circular(u2, center=(0.3, 0.3), radius=0.3)
    MRadapt2 = sam.make_MRAdapt(u2)
    MRadapt2(mra_config)
    svmpl.plot_field(u2, ax=axes[0, 1], cmap='plasma')
    axes[0, 1].set_title('Offset Circle')
    svmpl.set_axes_equal(axes[0, 1])

    # Test 3: Two circles
    u3 = sam.field.zeros(mesh, "two_circles")
    def init_two_circles(cell):
        cx, cy = cell.center()
        # Circle 1
        dist1_sq = (cx - 0.3)**2 + (cy - 0.3)**2
        # Circle 2
        dist2_sq = (cx + 0.3)**2 + (cy + 0.3)**2
        if dist1_sq < 0.2**2 or dist2_sq < 0.2**2:
            u3[cell.index] = 1.0
        else:
            u3[cell.index] = 0.0
    sam.algorithms.for_each_cell(mesh, init_two_circles)
    MRadapt3 = sam.make_MRAdapt(u3)
    MRadapt3(mra_config)
    svmpl.plot_field(u3, ax=axes[1, 0], cmap='coolwarm')
    axes[1, 0].set_title('Two Circles')
    svmpl.set_axes_equal(axes[1, 0])

    # Test 4: Refinement levels
    u4 = sam.field.zeros(mesh, "levels")
    init_circular(u4, center=(0.0, 0.0), radius=0.5)
    MRadapt4 = sam.make_MRAdapt(u4)
    MRadapt4(mra_config)
    svmpl.plot_field(u4, ax=axes[1, 1], show_level=True)
    axes[1, 1].set_title('Refinement Levels')
    svmpl.set_axes_equal(axes[1, 1])

    plt.tight_layout()
    output_file = Path("./results/viz_demo_multiple.png")
    plt.savefig(output_file, dpi=150)
    print(f"Saved: {output_file}")
    plt.close()


def main():
    """Run all visualization demos."""
    print("\n" + "=" * 70)
    print("Samurai AMR Visualization Demos")
    print("=" * 70)
    print()

    # Create output directory
    Path("./results").mkdir(parents=True, exist_ok=True)

    # Run demos (comment out real-time demo if running non-interactively)
    demo_static_scalar_field()
    demo_vector_field()
    demo_multiple_initial_conditions()

    # Note: Real-time demo requires interactive display
    # Comment out if running in non-interactive environment
    try:
        demo_realtime_plotting()
    except Exception as e:
        print(f"\nNote: Real-time demo skipped (requires display): {e}")

    print("\n" + "=" * 70)
    print("All demos complete!")
    print("Check ./results/ for output images")
    print("=" * 70)


if __name__ == "__main__":
    main()
