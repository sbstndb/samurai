# Errors, norms, and quick testing

Header: `samurai/numeric/error.hpp`

This page summarizes the user-facing utilities to measure errors against an exact solution and to perform quick convergence checks.

## L2 error against an exact solution

- Absolute L2 error (scalar or vector fields):

```cpp
double err = samurai::L2_error(u, [&](const auto& coord){ return exact(coord, t); });
```

- Relative L2 error (normalized by the exact solution L2 norm):

```cpp
double rel_err = samurai::L2_error<true>(u, [&](const auto& coord){ return exact(coord, t); });
```

Notes

- For FV fields on uniform or multiresolution meshes, `L2_error` integrates with one Gauss–Legendre point per cell. It is equivalent to evaluating the error at the cell center and multiplying by the cell measure.
- The exact function must match the field arity:
  - Scalar field: return a `double`.
  - Vector field: return a `samurai::Array<...>` with the correct dimension and SOA layout, as used by your vector field.

Example (vector field, see `demos/FiniteVolume/stokes_2d.cpp`):

```cpp
double v_err = samurai::L2_error(velocity, [](const auto& coord){
  const auto& x = coord[0];
  const auto& y = coord[1];
  double vx = 1.0/(pi*pi) * std::sin(pi*(x+y));
  double vy = -vx;
  return samurai::Array<double, dim, is_soa>{vx, vy};
});
```

## Convergence helpers

Given a mesh size h and an observed error e(h), the following helpers are provided:

- Hidden constant (from an assumed order):

```cpp
double C = samurai::compute_error_bound_hidden_constant<order>(h, error);
```

- Theoretical error bound from a hidden constant:

```cpp
double bound = samurai::theoretical_error_bound<order>(C, h);
```

- Observed convergence order from two resolutions:

```cpp
double p = samurai::convergence_order(h_fine, e_fine, h_coarse, e_coarse);
```

To retrieve a representative mesh size, use the minimum-level cell length (example from demos):

```cpp
double h = mesh.cell_length(mesh.min_level());
```

## Quick testing patterns

- Time-dependent scalar field (example pattern as in `demos/FiniteVolume/heat.cpp`):

```cpp
double err = samurai::L2_error(u, [&](const auto& coord){ return exact_solution(coord, t, diff_coeff); });
std::cout << std::scientific << "L2-error: " << err << std::endl;
```

- Static problem with convergence printout across refinements (example pattern as in `demos/highorder/main.cpp`):

```cpp
double h = mesh.cell_length(mesh.min_level());
double err = samurai::L2_error(u, exact_fn);
if (h_coarse > 0) {
  double order = samurai::convergence_order(h, err, h_coarse, err_coarse);
  std::cout << "L2-error: " << std::scientific << err << " (order = " << std::defaultfloat << order << ")\n";
}
h_coarse = h; err_coarse = err;
```

- Comparing piecewise-constant vs reconstructed field on MR meshes (example pattern as in `demos/FiniteVolume/burgers_mra.cpp`):

```cpp
auto u_recons = samurai::reconstruction(u);
double err_pc  = samurai::L2_error(u,        exact_fn);
double err_rec = samurai::L2_error(u_recons, exact_fn);
```

## Additional notes

- These error utilities work with scalar and vector fields created by Samurai field factories (e.g., `make_scalar_field`, `make_vector_field`).
- For relative errors, the denominator is the L2 norm of the exact solution computed with the same quadrature rule.
- Keep the exact function inexpensive, since it is evaluated at each quadrature point of every active cell.
