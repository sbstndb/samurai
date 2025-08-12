## Examples and usage patterns

This page shows minimal, verified usage patterns extracted from the demos in `demos/FiniteVolume/`. Focus is on user-facing APIs: mesh/fields, operators, BCs, time loops, PETSc solves, I/O, and CLI.

### Common program skeleton

```cpp
auto& app = samurai::initialize("...", argc, argv);
// app.add_option(...); app.add_flag(...);
SAMURAI_PARSE(argc, argv);

samurai::Box<double, dim> box({left}, {right});
samurai::MRMesh<samurai::MRConfig<dim>> mesh;
auto u = samurai::make_scalar_field<double>("u", mesh);

if (restart_file.empty()) {
  mesh = {box, min_level, max_level};
  u.resize();
  // initialize u[cell] ...
} else {
  samurai::load(restart_file, mesh, u);
}
```

### Advection 1D (explicit upwind)

File: `demos/FiniteVolume/advection_1d.cpp`

- Mesh: `MRMesh<MRConfig<1>>`, optional periodicity via `{is_periodic}`.
- Field: `make_scalar_field("u", mesh)`.
- Optional BC: `make_bc<Dirichlet<1>>(u, 0.).on(left, right)` when not periodic.
- Time loop: MRA, `update_ghost_mr(u)`, compute `unp1 = u - dt * upwind(a, u)`, swap, save/restart dump.

Snippet

```cpp
double dt = cfl * mesh.cell_length(max_level);
auto MRadaptation = samurai::make_MRAdapt(u);
auto mra_config   = samurai::mra_config().epsilon(2e-4);

auto unp1 = samurai::make_scalar_field<double>("unp1", mesh);
while (t != Tf) {
  MRadaptation(mra_config);
  t += dt; if (t > Tf) { dt += Tf - t; t = Tf; }

  samurai::update_ghost_mr(u);
  unp1.resize(); unp1.fill(0);
  unp1 = u - dt * samurai::upwind(a, u);
  std::swap(u.array(), unp1.array());
}
```

BC on non-periodic domain

```cpp
const xt::xtensor_fixed<int, xt::xshape<1>> left{-1}, right{1};
samurai::make_bc<samurai::Dirichlet<1>>(u, 0.).on(left, right);
```

### Linear convection (WENO5 + TVD-RK3)

File: `demos/FiniteVolume/linear_convection.cpp`

- Operator: `make_convection_weno5(velocity)`.
- Time integration: TVD-RK3.
- Periodic domain example via `std::array<bool,dim> periodic; periodic.fill(true);` passed to the mesh ctor.

Snippet

```cpp
samurai::VelocityVector<dim> velocity; velocity.fill(1); if constexpr (dim==2) velocity(1) = -1;
auto conv = samurai::make_convection_weno5<decltype(u)>(velocity);

// CFL time step
double dx = mesh.cell_length(max_level);
double sum_velocities = xt::sum(xt::abs(velocity))();
dt = cfl * dx / sum_velocities;

// TVD-RK3
auto u1 = samurai::make_scalar_field<double>("u1", mesh);
auto u2 = samurai::make_scalar_field<double>("u2", mesh);
auto unp1 = samurai::make_scalar_field<double>("unp1", mesh);
u1 = u - dt * conv(u);
u2 = 3./4 * u + 1./4 * (u1 - dt * conv(u1));
unp1 = 1./3 * u + 2./3 * (u2 - dt * conv(u2));
samurai::swap(u, unp1);
```

### Heat equation (explicit and implicit)

File: `demos/FiniteVolume/heat.cpp`

- Operator: `make_diffusion_order2` with `DiffCoeff<dim>` or scalar coeff.
- Explicit: `u - dt * diff(u)`; Implicit: `petsc::solve(id + dt*diff, unp1, u)`.
- Example BC: homogeneous Neumann on both `u` and `unp1`.

Snippet (implicit Backward Euler)

```cpp
samurai::DiffCoeff<dim> K; K.fill(diff_coeff);
auto diff = samurai::make_diffusion_order2<decltype(u)>(K);
auto id   = samurai::make_identity<decltype(u)>();

auto unp1 = samurai::make_scalar_field<double>("unp1", mesh);
samurai::make_bc<samurai::Neumann<1>>(u, 0.);
samurai::make_bc<samurai::Neumann<1>>(unp1, 0.);

auto back_euler = id + dt * diff;
samurai::petsc::solve(back_euler, unp1, u);
samurai::swap(u, unp1);
```

L2 error against an exact solution at time `t`

```cpp
double error = samurai::L2_error(u, [&](const auto& coord){ return exact_solution(coord, t, diff_coeff); });
```

### Stokes 2D (block operator + PETSc)

File: `demos/FiniteVolume/stokes_2d.cpp`

- Unknowns: vector velocity and scalar pressure.
- Operator block:
  - stationary: `[[Diff, Grad], [-Div, 0]]`
  - unsteady (BE): `[[I + dt*Diff, dt*Grad], [-Div, 0]]`
- Solve with `petsc::make_solver<monolithic>(op)` and `solve(rhs, zero)`.

Snippet (stationary operator and solve)

```cpp
auto velocity = samurai::make_vector_field<dim, false>("velocity", mesh);
auto pressure = samurai::make_scalar_field<double>("pressure", mesh);

auto diff    = samurai::make_diffusion_order2<decltype(velocity)>();
auto grad    = samurai::make_gradient_order2<decltype(pressure)>();
auto div     = samurai::make_divergence_order2<decltype(velocity)>();
auto zero_op = samurai::make_zero_operator<decltype(pressure)>();

auto stokes = samurai::make_block_operator<2,2>(diff, grad,
                                                 -div, zero_op);
auto solver = samurai::petsc::make_solver<monolithic>(stokes);
solver.set_unknowns(velocity, pressure);
solver.solve(/*rhs_v*/ f, /*rhs_p*/ zero);
```

Dirichlet/Neumann BC examples used in the demo

```cpp
samurai::make_bc<samurai::Dirichlet<1>>(velocity, [](const auto&, const auto&, const auto& x){ /* return vector */ });
samurai::make_bc<samurai::Neumann<1>>(pressure,  [](const auto&, const auto&, const auto& x){ /* return scalar */ });
```

### Manual block matrix assembly (advanced)

File: `demos/FiniteVolume/manual_block_matrix_assembly.cpp`

- Demonstrates custom PETSc block assembly with `petsc::ManualAssembly` blocks.
- After composing a `make_block_operator`, create an assembly, set unknowns, and build the matrix.

Minimal assembly sequence

```cpp
auto block_op = samurai::make_block_operator<3,3>(/* blocks */);
auto assembly = samurai::petsc::make_assembly<true>(block_op); // monolithic
assembly.set_unknowns(u_e, aux_Ce, u_s);
Mat J; assembly.create_matrix(J); assembly.assemble_matrix(J);
```

### I/O: save, restart, and visualization

- Save mesh and fields: `samurai::save(path, filename, mesh, field1, field2, ...)`.
- Write a restart file: `samurai::dump(path, filename, mesh, fields...)`.
- Load a restart: `samurai::load(restart_file, mesh, fields...)`.
- 1D visualization helper: `python path/to/samurai/python/read_mesh.py <prefix> --field u level --start N0 --end N1`.

Example (from heat 1D)

```cpp
std::cout << "python <samurai>/python/read_mesh.py "
          << filename << "_ite_ --field u level --start 1 --end " << nsave << std::endl;
```

### Typical CLI flags used in demos

- `--left, --right` domain box corners
- `--min-level, --max-level` refinement bounds
- `--cfl, --dt, --Tf, --Ti` time parameters
- `--periodic` domain periodicity (when present)
- `--restart-file` HDF5 restart
- `--path, --filename, --nfiles` output control
- Problem-specific flags (e.g., `--explicit` in heat, `--test-case` in Stokes)

All examples call `SAMURAI_PARSE(argc, argv)` after declaring options.
