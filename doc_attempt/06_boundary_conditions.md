### Boundary conditions

Headers: `samurai/bc/bc.hpp`, `samurai/bc/dirichlet.hpp`, `samurai/bc/neumann.hpp`

### Attaching boundary conditions
- **Dirichlet (order k, default k = 1)**
  - Constant value (scalar field):
    ```cpp
    samurai::make_bc<samurai::Dirichlet<1>>(u, 0.0); // use samurai::Dirichlet<>() for default order 1
    ```
  - Constant per component (vector field of size 2 shown):
    ```cpp
    samurai::make_bc<samurai::Dirichlet<1>>(velocity, 1.0, 0.0);
    ```
  - Value from a function of the face center (and direction, inner cell):
    ```cpp
    samurai::make_bc<samurai::Dirichlet<1>>(u,
        [](const auto& /*dir*/, const auto& /*cell_in*/, const auto& x){ return x[0]*x[0]; });
    ```

- **Neumann (order 1)**
  ```cpp
  samurai::make_bc<samurai::Neumann<1>>(u, 0.0);
  ```

Notes
- The template order parameter defaults to 1; `Dirichlet<>()` is equivalent to `Dirichlet<1>`.
- For vector fields, pass one constant per component; the arity must match the field size.
- Function BCs receive `(direction, cell_in, coords)` and must return a value convertible to the field layout (scalar or vector).

### Limiting where BCs apply
By default, BCs apply on all boundary faces. Use `.on(...)` to restrict them.

- **Cartesian directions** (`xt::xtensor_fixed<int, xt::xshape<dim>>`):
  - 1D: left/right
    ```cpp
    const xt::xtensor_fixed<int, xt::xshape<1>> left{-1};
    const xt::xtensor_fixed<int, xt::xshape<1>> right{1};
    samurai::make_bc<samurai::Dirichlet<1>>(u, 0.0)->on(left, right);
    ```
  - 2D: left/right/bottom/top
    ```cpp
    const xt::xtensor_fixed<int, xt::xshape<2>> left  {-1, 0};
    const xt::xtensor_fixed<int, xt::xshape<2>> right { 1, 0};
    const xt::xtensor_fixed<int, xt::xshape<2>> bottom{ 0,-1};
    const xt::xtensor_fixed<int, xt::xshape<2>> top   { 0, 1};
    samurai::make_bc<samurai::Dirichlet<1>>(u, 0.0)->on(left, right);
    ```

- **Coordinate predicate** (`CoordsRegion`):
  ```cpp
  // Apply only where the boundary face-center satisfies the predicate
  samurai::make_bc<samurai::Dirichlet<1>>(u, 1.0)
      ->on([](const auto& p){ return p[0] > 0.5; });
  ```

Available region builders the `.on(...)` can take:
- `Everywhere` (default): all faces
- one or several direction vectors (`xt::xtensor_fixed<int, xt::xshape<dim>>`)
- a coordinate predicate as shown above
- a set/subset region (see `doc_attempt/13_boundary_regions_and_subsets.md`)

### Order and ghost layers
- `Dirichlet<k>` fills `k` ghost layers using polynomial fits; requires `mesh.config::ghost_width >= k`.
- `Neumann<1>` requires `ghost_width >= 1`.
- Configure ghost width in your mesh config, e.g. `samurai::UniformConfig<dim, /*ghost_width=*/2>`.

### Periodic dimensions
If a mesh dimension is periodic (`mesh.is_periodic(d)`), BCs are not applied along that dimension; ghost values are handled by periodic wrapping.

### Small, concrete examples
- 1D advection-style inflow/outflow
  ```cpp
  auto u = samurai::make_scalar_field<double>("u", mesh);
  const xt::xtensor_fixed<int, xt::xshape<1>> left{-1}, right{1};
  samurai::make_bc<samurai::Dirichlet<1>>(u, 0.0)->on(left, right);
  ```

- 2D lid-driven cavity velocity BCs (vector field size 2)
  ```cpp
  auto velocity = samurai::make_vector_field<double, 2>("velocity", mesh);
  const xt::xtensor_fixed<int, xt::xshape<2>> left{-1,0}, right{1,0}, bottom{0,-1}, top{0,1};
  samurai::make_bc<samurai::Dirichlet<1>>(velocity, 1.0, 0.0)->on(top);
  samurai::make_bc<samurai::Dirichlet<1>>(velocity, 0.0, 0.0)->on(left, bottom, right);
  ```

- Neumann walls (e.g. reaction–diffusion)
  ```cpp
  samurai::make_bc<samurai::Neumann<1>>(u, 0.0);
  ```

### Runtime updates
When you use built-in finite-volume operators and schemes, ghosts and BCs are updated for you. Advanced users writing custom kernels can use:
- `samurai::update_bc_for_scheme(field);`
- `samurai::update_further_ghosts_by_polynomial_extrapolation(field);`
But typical applications only need to attach BCs once and step the solver.

### What is implemented
- `Dirichlet<k>`: orders k = 1..4.
- `Neumann<1>`: only first-order is available.

See `samurai/bc/dirichlet.hpp` and `samurai/bc/neumann.hpp` for details; higher `k` requires wider ghost layers.

