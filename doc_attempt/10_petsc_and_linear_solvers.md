# PETSc integration and linear solvers

## Prerequisites

- **Enable PETSc at configure time**: add `set(SAMURAI_WITH_PETSC ON)` before `find_package(samurai)` in your CMake.
- **Include headers**: do not include `samurai/petsc.hpp` directly. Include `samurai/schemes/fv.hpp` in your code.

Minimal CMake excerpt:

```cmake
set(SAMURAI_WITH_PETSC ON)
find_package(samurai CONFIG REQUIRED)
add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE samurai::samurai)
```

## Basic linear implicit solve (heat equation)

```cpp
auto diff = samurai::make_diffusion_order2<decltype(u)>(K);
auto id   = samurai::make_identity<decltype(u)>();
auto implicit_op = id + dt * diff;          // [Id + dt * Diff]
samurai::petsc::solve(implicit_op, unp1, u); // solves [Id + dt*Diff](unp1) = u
```

- Boundary conditions must be set on the unknown (e.g., `unp1`) via `make_bc(...)`. They are enforced in the assembled system.
- The PETSc linear solver is configurable via CLI options; by default GMRES with ILU and rtol ~1e-5 is used (see tutorial docs).

RHS can also be given as a field expression:

```cpp
auto rhs = u + dt * react(u);
samurai::petsc::solve(implicit_op, unp1, rhs);
```

## Nonlinear implicit solve

When the operator is nonlinear (e.g., diffusion or reaction depends on the unknown), the same call selects a nonlinear solver automatically. Provide an initial guess in the unknown field.

```cpp
auto implicit_op = id + dt * diff - dt * react; // nonlinear
unp1 = u;                                       // initial guess for Newton
samurai::petsc::solve(implicit_op, unp1, u);
```

## Reusable solver objects (configure once, solve many times)

Create a solver, set the unknown, then solve with varying right-hand sides:

```cpp
auto solver = samurai::petsc::make_solver(implicit_op);
solver.set_unknown(unp1);
solver.solve(u);          // or solver.solve(rhs);
```

- If the operator structure or mesh layout changes (e.g., time step `dt` changes inside the operator expression, or mesh is adapted), rebuild the solver or call `solver.reset()` before the next `solve`.
- You can query `solver.iterations()` after a solve.

## Block systems (Stokes)

Build a 2×2 block operator and solve for velocity/pressure.

```cpp
auto diff = samurai::make_diffusion_order2<VelocityField>();
auto grad = samurai::make_gradient_order2<PressureField>();
auto div  = samurai::make_divergence_order2<VelocityField>();
auto zero_op = samurai::make_zero_operator<PressureField>();

auto stokes = samurai::make_block_operator<2,2>(diff, grad,
                                                -div, zero_op);

// Monolithic block solver (single KSP on assembled matrix)
auto solver = samurai::petsc::make_solver<true>(stokes);
solver.set_unknowns(velocity, pressure);
solver.solve(f, zero_op(pressure)); // velocity RHS 'f' and pressure RHS zero
```

Time-discrete (backward Euler) variant just replaces the blocks, e.g. `id + dt * diff`, `dt * grad`.

Nested (field-split) solver can be created with `make_solver<false>(stokes)`; typical PETSc field-split configuration is done via CLI options. For advanced setups, see `demos/FiniteVolume/stokes_2d.cpp`.

## Configuring solvers via PETSc options

- Pass PETSc options on the command line, e.g. `-ksp_type preonly -pc_type lu`.
- For nonlinear problems, SNES options apply (e.g., `-snes_type`, tolerances).
- For block systems with field splits, use PETSc field-split options; the demos show customary choices (Schur complement preconditioner, per-field KSP/PC).

## Troubleshooting

- If you see a compile-time error about including `samurai/petsc.hpp`, switch to `#include <samurai/schemes/fv.hpp>` and ensure `SAMURAI_WITH_PETSC` is set before `find_package(samurai)`.
- If a message complains about undefined unknowns, call `solver.set_unknown(u)` (or `set_unknowns(...)` for blocks) before `solve(...)`, or use the free function `samurai::petsc::solve(op, u, rhs)` form.
- PETSc divergence messages indicate the linear/nonlinear solver did not converge; adjust PETSc options or preconditioners.

## References and demos

- Linear heat: `demos/FiniteVolume/heat.cpp`, `heat_heterogeneous.cpp`
- Nonlinear: `demos/FiniteVolume/heat_nonlinear.cpp`, `nagumo.cpp`
- Block Stokes (stationary and transient): `demos/FiniteVolume/stokes_2d.cpp`
