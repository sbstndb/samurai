#!/usr/bin/env python3
"""
Finite Volume example for the linear convection equation in 2D using multiresolution.

This demo demonstrates:
- 2D adaptive mesh refinement (AMR) with multiresolution analysis
- WENO5 convection operator for high-order accuracy
- TVD-RK3 time stepping scheme
- Periodic boundary conditions
- Rectangle initial condition

The linear convection equation: du/dt + a·∇u = 0
with constant velocity a = (1, -1) and a rectangle initial condition.

Equivalent to: demos/FiniteVolume/linear_convection.cpp
"""

import sys
import os
from pathlib import Path

# Add build directory to path for development
build_dir = os.path.join(os.path.dirname(__file__), "..", "..", "build", "python")
if os.path.exists(build_dir):
    sys.path.insert(0, build_dir)

import samurai_python as sam


def init_rectangle(u):
    """Initialize field with a rectangle condition.

    Rectangle: [-0.8, -0.3] x [0.3, 0.8]

    Args:
        u: ScalarField2D to initialize
    """
    def init_cell(cell):
        cx, cy = cell.center()
        # Rectangle condition
        if -0.8 <= cx <= -0.3 and 0.3 <= cy <= 0.8:
            u[cell.index] = 1.0
        else:
            u[cell.index] = 0.0

    sam.for_each_cell(u.mesh, init_cell)


def main():
    """Main simulation function."""

    # ============================================================
    # Simulation parameters
    # ============================================================

    # Domain: [-1, 1] x [-1, 1]
    box = sam.Box2D([-1.0, -1.0], [1.0, 1.0])

    # Velocity: a = (1, -1)
    velocity = [1.0, -1.0]

    # Time parameters
    Tf = 3.0          # Final time
    cfl = 0.95        # CFL condition

    # Mesh parameters
    min_level = 5
    max_level = 9
    max_stencil_size = 6  # Required for WENO5

    # Multiresolution parameters
    epsilon = 1e-4      # MR adaptation tolerance
    regularity = 1.0    # Gradation parameter

    # Output parameters
    output_path = Path("/home/sbstndbs/sbstndbs/samurai/linear_convection_python")
    filename = "linear_convection_2d_python"
    nfiles = 10  # Number of output files

    print("=" * 70)
    print("Linear Convection 2D - WENO5 with TVD-RK3")
    print("=" * 70)
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
    # Mesh configuration
    # ============================================================

    config = sam.MeshConfig2D()
    config.min_level = min_level
    config.max_level = max_level
    config.max_stencil_size = max_stencil_size
    config.set_periodic(True)  # Periodic boundary conditions

    # Create mesh and fields
    mesh = sam.MRMesh2D(box, config)

    # Create fields for TVD-RK3 time stepping
    u = sam.field.scalar(mesh, "u", init=0.0)      # Current solution
    u1 = sam.field.scalar(mesh, "u1", init=0.0)    # RK stage 1
    u2 = sam.field.scalar(mesh, "u2", init=0.0)    # RK stage 2
    unp1 = sam.field.scalar(mesh, "unp1", init=0.0)  # RK stage 3

    # ============================================================
    # Initialize with rectangle condition
    # ============================================================

    print("Initializing field with rectangle condition...")
    init_rectangle(u)

    # ============================================================
    # Mesh adaptation setup
    # ============================================================

    MRadaptation = sam.make_MRAdapt(u)
    mra_config = sam.MRAConfig()
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
    sam.save(str(output_path), save_filename, u)

    # ============================================================
    # Time stepping setup
    # ============================================================

    # Compute dt based on CFL condition
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
    # Main time loop with TVD-RK3
    # ============================================================

    t = 0.0
    nt = 0

    print(f"{'Iter':>6} {'Time':>12} {'dt':>12} {'Cells':>10}")
    print("-" * 42)

    while t < Tf:
        # Adapt mesh FIRST (before computing fluxes)
        MRadaptation(mra_config)

        # Resize temporary fields after mesh adaptation
        u1.resize()
        u2.resize()
        unp1.resize()

        # Update time
        t += dt
        if t > Tf:
            # Adjust dt to exactly reach Tf
            dt += Tf - t
            t = Tf

        nt += 1

        # ========================================================
        # TVD-RK3 (SSPRK3) time stepping scheme
        # ========================================================
        # Scheme:
        #   u1   = u - dt * conv(u)
        #   u2   = 3/4*u + 1/4*(u1 - dt*conv(u1))
        #   unp1 = 1/3*u + 2/3*(u2 - dt*conv(u2))

        # Stage 1: u1 = u - dt * conv(u)
        flux1 = sam.operators.convection_weno5(velocity, u)
        u1.assign(u - dt * flux1)

        # Stage 2: u2 = 3/4*u + 1/4*(u1 - dt*conv(u1))
        flux2 = sam.operators.convection_weno5(velocity, u1)
        u2.assign((3.0 / 4.0) * u + (1.0 / 4.0) * (u1 - dt * flux2))

        # Stage 3: unp1 = 1/3*u + 2/3*(u2 - dt*conv(u2))
        flux3 = sam.operators.convection_weno5(velocity, u2)
        unp1.assign((1.0 / 3.0) * u + (2.0 / 3.0) * (u2 - dt * flux3))

        # Swap u and unp1 (u becomes the new solution)
        sam.swap_field_arrays_2d(u, unp1)

        # ========================================================
        # Save output
        # ========================================================
        if t >= (nsave + 1) * dt_save or abs(t - Tf) < 1e-12:
            suffix = f"_ite_{nsave}" if nfiles > 1 else ""
            save_filename = f"{filename}{suffix}"
            sam.save(str(output_path), save_filename, u)
            nsave += 1

            # Print progress
            print(f"{nt:6d} {t:12.6f} {dt:12.6f} {mesh.nb_cells:10d}")

    print()
    print("=" * 70)
    print("Simulation complete!")
    print(f"  Final time: {t:.6f}")
    print(f"  Time steps: {nt}")
    print(f"  Output files: {nsave + 1} (including initial)")
    print()
    print(f"Generated files in {output_path.absolute()}:")
    print(f"  - {filename}_init.h5/.xdmf    (initial condition)")
    print(f"  - {filename}_ite_*.h5/.xdmf   (time series)")
    print()
    print(f"To visualize in Paraview:")
    print(f"  paraview {output_path.absolute() / filename}_ite_0.xdmf")
    print()
    print(f"Equivalent C++ demo: demos/FiniteVolume/linear_convection.cpp")
    print("=" * 70)


if __name__ == "__main__":
    main()
