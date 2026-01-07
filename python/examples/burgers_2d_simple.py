#!/usr/bin/env python3
"""Burgers 2D equation - Simple version without mesh adaptation"""

import sys
import os
import math
from pathlib import Path

# Add build directory to path
build_dir = os.path.join(os.path.dirname(__file__), "..", "..", "build", "python")
if os.path.exists(build_dir):
    sys.path.insert(0, build_dir)

import samurai_python as sam


def init_hat(u):
    """Initialize with hat function"""
    def init_cell(cell):
        cx, cy = cell.center()
        dist = math.sqrt(cx**2 + cy**2)
        if dist <= 0.5:
            value = -1.0 / 0.5 * dist + 1.0
        else:
            value = 0.0
        u[cell.index] = [value, value]
    sam.algorithms.for_each_cell(u.mesh, init_cell)


def main():
    # Parameters - use fixed mesh (no adaptation)
    box = sam.geometry.Box2D([-1.0, -1.0], [1.0, 1.0])

    config = sam.config.MeshConfig2D()
    config.min_level = 6  # Fixed level, no adaptation
    config.max_level = 6
    config.max_stencil_size = 6

    mesh = sam.mesh.MRMesh2D(box, config)

    # Fields
    u = sam.field.zeros_vector(mesh, "u", n_components=2)
    u1 = sam.field.zeros_vector(mesh, "u1", n_components=2)
    u2 = sam.field.zeros_vector(mesh, "u2", n_components=2)

    # Initialize
    init_hat(u)
    sam.boundary.dirichlet(u, [0.0, 0.0], order=3)
    sam.boundary.dirichlet(u1, [0.0, 0.0], order=3)
    sam.boundary.dirichlet(u2, [0.0, 0.0], order=3)

    # Time stepping
    Tf = 0.2
    cfl = 0.5
    dx = mesh.min_cell_length
    dt = cfl * dx / 4

    output_path = Path("./results")
    output_path.mkdir(parents=True, exist_ok=True)

    # Save initial
    sam.save(str(output_path), "burgers_simple_init", u)
    print(f"Saved initial condition, cells = {mesh.nb_cells}")

    # Time loop
    t = 0.0
    nt = 0

    while t < Tf:
        t += dt
        if t > Tf:
            dt += Tf - t
            t = Tf

        nt += 1
        print(f"iteration {nt}: t = {t:.4f}, dt = {dt:.6f}")

        # RK3
        flux1 = sam.operators.convection_weno5(u)
        u1.assign(u - dt * flux1)

        flux2 = sam.operators.convection_weno5(u1)
        u2.assign((3.0/4.0) * u + (1.0/4.0) * (u1 - dt * flux2))

        flux3 = sam.operators.convection_weno5(u2)
        u.assign((1.0/3.0) * u + (2.0/3.0) * (u2 - dt * flux3))

        # Save every 10 iterations
        if nt % 10 == 0:
            sam.save(str(output_path), f"burgers_simple_{nt:04d}", u)

    # Save final
    sam.save(str(output_path), "burgers_simple_final", u)
    print(f"\nSimulation complete! {nt} iterations, final time = {t}")


if __name__ == "__main__":
    main()
