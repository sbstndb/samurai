# Finite-volume operators and schemes

Header aggregator: `samurai/schemes/fv.hpp`

What you get (builders, templated on field types):

- Convection (linear, constant velocity): `make_convection_upwind<Field>(velocity)`, `make_convection_weno5<Field>(velocity)`
- Convection (linear, velocity as field): `make_convection_upwind<Field>(vel_field)`, `make_convection_weno5<Field>(vel_field)`
- Convection (nonlinear, e.g. Burgers or v·∇v): `make_convection_upwind<Field>()`, `make_convection_weno5<Field>()`
- Diffusion (2nd order): `make_diffusion_order2<Field>(k)`, `make_diffusion_order2<Field>(K)`, `make_multi_diffusion_order2<Field>(K_comp)`, `make_laplacian_order2<Field>()`
- Gradient (2nd order): `make_gradient_order2<PressureField>()`
- Divergence (2nd order): `make_divergence_order2<VelocityField>()`
- Identity: `make_identity<Field>()`
- Zero operator: `make_zero_operator<Field>()` or `make_zero_operator<OutputField, InputField>()`

Notes on diffusion builders:

- `k` is a scalar isotropic coefficient.
- `K` can be a `samurai::DiffCoeff<dim>` (diagonal anisotropy per direction), or a field whose value type is `DiffCoeff<dim>` for heterogeneous coefficients.
- `make_multi_diffusion_order2<Field>(K_comp)` accepts per-component coefficients when `Field` is a vector field.

How to apply and compose operators (explicit use):

```cpp
// Given fields u (scalar) or velocity (vector) and a mesh already set up
auto diff = samurai::make_diffusion_order2<decltype(u)>(K);
auto id   = samurai::make_identity<decltype(u)>();

// Apply explicitly
auto unp1 = samurai::make_scalar_field<double>("unp1", u.mesh());
unp1 = u - dt * diff(u);

// Convection examples
// 1) Constant velocity a (array or xtensor-fixed) on scalar field u
auto conv_cst = samurai::make_convection_upwind<decltype(u)>(a);
unp1 = u - dt * conv_cst(u);

// 2) Velocity field v driving scalar ink
using InkField = decltype(ink);
auto conv_var = samurai::make_convection_weno5<InkField>(v);
ink_np1 = ink - dt * conv_var(ink);

// 3) Nonlinear convection (Burgers): PDE uses (1/2)∇·(u^2)
auto conv_burgers = 0.5 * samurai::make_convection_upwind<decltype(u)>();
unp1 = u - dt * conv_burgers(u);
```

Time stepping used in demos (explicit):

- Advection on multiresolution meshes: `dt = cfl * mesh.cell_length(max_level)` (see `advection_1d.cpp`, `advection_2d.cpp`).
- Advection with velocity magnitude scaling: `dt = cfl * dx / sum_max_velocities` (see `lid_driven_cavity.cpp`).
- Diffusion: `dt = cfl * (dx*dx) / (pow(2, dim) * diff_coeff)` (see `heat.cpp`).

Implicit use (with PETSc):

```cpp
auto diff = samurai::make_diffusion_order2<decltype(u)>(K);
auto id   = samurai::make_identity<decltype(u)>();
auto back_euler = id + dt * diff;                   // backward Euler
samurai::petsc::solve(back_euler, unp1, u);         // solves [I + dt*Diff](unp1) = u
```

Coupled systems via block operators (example: Stokes, see `demos/FiniteVolume/stokes_2d.cpp`):

```cpp
auto diff    = samurai::make_diffusion_order2<VelocityField>(nu);
auto grad    = samurai::make_gradient_order2<PressureField>();
auto div     = samurai::make_divergence_order2<VelocityField>();
auto zero_op = samurai::make_zero_operator<PressureField>();
auto id      = samurai::make_identity<VelocityField>();

// Backward Euler in time
auto stokes = samurai::make_block_operator<2, 2>(id + dt * diff, dt * grad,
                                                 -div,            zero_op);
```

Boundary conditions and ghosts

- Attach boundary conditions to fields before use (see demos for `Dirichlet`, `Neumann`).
- FV operators from `samurai/schemes/fv.hpp` update ghost values automatically when applied.
- Low-level stencil operators like `samurai::upwind(a, u)` (from `stencil_field.hpp`) require calling `update_ghost_mr(u)` before use (see `advection_*.cpp`).

Per-scheme options (when available)

- Exclude boundary flux contributions during assembly/evaluation: `scheme.include_boundary_fluxes(false)` (used on diffusion in `manual_block_matrix_assembly.cpp`).
- Compute fluxes at a finer level of the multiresolution hierarchy: `scheme.enable_max_level_flux(true)` (used in `burgers_mra.cpp`).
