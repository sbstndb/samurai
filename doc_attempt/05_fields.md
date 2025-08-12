# Fields

Header: `samurai/field.hpp`

- **Scalar field**: `samurai::ScalarField<Mesh, T = double>`
- **Vector field**: `samurai::VectorField<Mesh, T = double, n_comp, SOA = false>`

Both store cell-centered data on a Samurai mesh and integrate with the field-expression system for efficient, vectorized operations.

## Create fields

- **Scalar** (default type is `double`):

```cpp
auto u  = samurai::make_scalar_field<double>("u", mesh);
auto u2 = samurai::make_scalar_field("u2", mesh);           // double by default
```

- **Vector**: choose number of components at compile-time:

```cpp
auto v2  = samurai::make_vector_field<2>("v2", mesh);       // 2 components
auto vel = samurai::make_vector_field<double, decltype(mesh)::dim>("velocity", mesh);
```

- **Initialize at creation**

  - With a constant value:

    ```cpp
    auto u0 = samurai::make_scalar_field<double>("u0", mesh, 0.0);
    auto a  = samurai::make_vector_field<double, 2>("a", mesh, 0.0);
    ```

  - From a function of the cell center (point value):

    ```cpp
    auto u_fun = samurai::make_scalar_field<double>(
        "u_fun", mesh,
        [](const auto& x){ return 1.0; }
    );
    ```

  - As a cell average via Gauss–Legendre quadrature:

    ```cpp
    samurai::GaussLegendre<2> gl; // order-2 quadrature
    auto u_avg = samurai::make_scalar_field<double>(
        "u_avg", mesh,
        [](const auto& x){ return x[0]; }, gl
    );
    ```

Note: legacy `make_field(...)` overloads exist but are deprecated; prefer `make_scalar_field` / `make_vector_field`.

## Access and modify values

- **By cell**:

```cpp
samurai::for_each_cell(mesh, [&](const auto& cell){
  u[cell] = 1.0;            // scalar
  v2[cell][0] = 2.0;        // vector component 0
});
```

- **Vectorized blocks (interval views)**: use `for_each_interval` and the block-call operator.

```cpp
using mesh_id_t = typename decltype(mesh)::mesh_id_t;
samurai::for_each_interval(mesh[mesh_id_t::cells], [&](std::size_t level, auto& I, const auto& index){
  auto block = u(level, I, index);   // xtensor-compatible view
  block += 2.0;
});
```

- **Utilities**:

  - `field.fill(value)` to set all entries.
  - `field.name()` to get/set the field name (used in I/O).
  - `samurai::swap(f1, f2)` swaps data and ghost state between compatible fields.

## Resizing when the mesh changes

Call `field.resize()` whenever the underlying mesh has changed (e.g., after multiresolution adaptation) to match the new number of cells:

```cpp
// After mesh adaptation or if the mesh has been reassigned
u.resize();
v2.resize();
```

## Boundary conditions and ghosts

Attach boundary conditions with the helpers from `samurai/bc/bc.hpp` (see `06_boundary_conditions.md`):

```cpp
samurai::make_bc<samurai::Dirichlet<1>>(u, 0.0); // constant Dirichlet

// Restrict to a direction (1D example)
const xt::xtensor_fixed<int, xt::xshape<1>> left{-1};
samurai::make_bc<samurai::Dirichlet<1>>(u, 1.0)->on(left);
```

Before applying finite-volume operators each step, update ghost values (see `07_fv_operators_and_schemes.md`):

```cpp
samurai::update_ghost_mr(u);
```

## Summary of the main API

- **Creation**: `make_scalar_field`, `make_vector_field` with optional constant/function/GL-init.
- **Access**: `field[cell]` (and `[comp]` for vectors), block view `field(level, interval, index...)`.
- **Lifecycle**: `field.resize()`, `field.fill(value)`, `field.name()`.
- **BCs/Ghosts**: attach via `make_bc<...>(field, ...)`; call `update_ghost_mr(field, ...)` when needed.
