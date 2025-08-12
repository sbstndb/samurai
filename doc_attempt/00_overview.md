### Samurai documentation overview

Samurai is a C++ library for finite-volume computations on Cartesian meshes. Its public API lets you build uniform or adaptive meshes, attach fields and boundary conditions, assemble FV operators, adapt meshes via multiresolution, and save/restart simulations. The demos in `demos/FiniteVolume` illustrate typical usage.

- Meshes
  - Uniform meshes via `samurai::UniformMesh<Config>`
  - Adaptive multiresolution meshes via `samurai::MRMesh<Config>`
- Fields bound to meshes
  - `samurai::ScalarField<Mesh, T>` and `samurai::VectorField<Mesh, T, n_comp, SOA>`
- Boundary conditions for fields
  - Dirichlet and Neumann of selectable order, attachable to any field
- Finite-volume (FV) operators and schemes
  - Convection, diffusion, gradient, divergence, identity, zero operators
  - Operator algebra and block operators for multi-physics
- Multiresolution adaptation (MRA)
  - Automatic refinement/coarsening based on wavelet details with `mra_config`
- I/O and restart
  - HDF5+XDMF output with `samurai::save`; restart via `samurai::dump`/`samurai::load`
- PETSc integration (optional)
  - Linear system assembly and solves for implicit schemes and block systems

What this documentation focuses on
- The user-facing headers and APIs located under `include/samurai/`
- Concrete usage patterns shown in the demos under `demos/FiniteVolume/`

Suggested reading order (progressively deeper)
1. 01_build_and_cmake.md
2. 02_program_structure_and_cli.md
3. 03_mesh_uniform.md and 04_mesh_multiresolution.md
4. 05_fields.md
5. 06_boundary_conditions.md
6. 07_fv_operators_and_schemes.md
7. 08_io_save_load.md
8. 09_multiresolution_adaptation.md
9. 10_petsc_and_linear_solvers.md
10. 11_examples_and_patterns.md

All examples and function names are taken from the public headers and the provided demos (e.g. `FiniteVolume/advection_1d.cpp`, `FiniteVolume/heat.cpp`, `FiniteVolume/stokes_2d.cpp`).

What you will not find here
- Internal implementation details; only user-facing interfaces and patterns used in the demos
- Any invented API: everything shown exists in the headers provided
