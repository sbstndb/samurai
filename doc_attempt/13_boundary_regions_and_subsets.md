# Boundary regions and subsets

Headers: `samurai/boundary.hpp`, `samurai/bc/bc.hpp`, `samurai/subset/*`

## What this page covers

- Boundary cell selections at a given level (domain or subdomain)
- Outer layers used for ghost cell handling
- How to restrict BCs to specific boundary parts with `.on(...)`
- How to build and iterate custom subsets (set operations)

## Direction vectors

Cartesian directions are expressed as fixed-size integer vectors (`xt::xtensor_fixed<int, xt::xshape<dim>>`). Examples in 2D:

```cpp
const xt::xtensor_fixed<int, xt::xshape<2>> left  {-1, 0};
const xt::xtensor_fixed<int, xt::xshape<2>> right { 1, 0};
const xt::xtensor_fixed<int, xt::xshape<2>> bottom{ 0,-1};
const xt::xtensor_fixed<int, xt::xshape<2>> top   { 0, 1};
```

## Predefined boundary helpers

All helpers below return a `LevelCellArray` at the requested `level`.

- Domain boundary on a specific side:

  ```cpp
  auto cells = samurai::domain_boundary(mesh, level, direction);
  ```

- Domain boundary (all sides at once):

  ```cpp
  auto cells = samurai::domain_boundary(mesh, level);
  ```

- Subdomain boundary (useful when you work on a nested subdomain):

  ```cpp
  auto cells = samurai::subdomain_boundary(mesh, level, direction); // single side
  // or
  // auto cells = samurai::subdomain_boundary(mesh, level);        // all sides
  ```

- Outer layers outside the domain (commonly used for BC ghost zones):

  ```cpp
  // All sides, width k (union of k layers in each Cartesian direction)
  auto ghosts_all_dirs = samurai::domain_boundary_outer_layer(mesh, level, /*layer_width=*/k);

  // Only in the given direction, width k
  auto ghosts_dir = samurai::domain_boundary_outer_layer(mesh, level, direction, /*layer_width=*/k);
  ```

Typical iteration over a boundary selection:

```cpp
samurai::for_each_cell(mesh, cells, [&](auto& cell) {
  // user work on boundary cells
});
```

## Restricting boundary conditions to regions

Attach a BC to a field (see `doc_attempt/06_boundary_conditions.md`), then restrict where it applies using `.on(...)`.

- Apply everywhere (default):

  ```cpp
  samurai::make_bc<samurai::Dirichlet<1>>(u, 0.0);
  // same as ->on(Everywhere)
  ```

- Apply only on selected Cartesian sides:

  ```cpp
  samurai::make_bc<samurai::Dirichlet<1>>(u, 0.0)->on(left, right);
  ```

- Apply only where the boundary face-center satisfies a predicate (`CoordsRegion`):

  ```cpp
  samurai::make_bc<samurai::Dirichlet<1>>(u, 1.0)
      ->on([](const auto& p) { return p[0] > 0.5; });
  // p is the face-center coordinates
  ```

- Apply on a set/subset you build (see below):

  ```cpp
  auto L = mesh.max_level();
  auto rim = samurai::difference(
      samurai::self(mesh.domain()).on(L),
      samurai::contract(samurai::self(mesh.domain()).on(L), /*width=*/1)
  );
  samurai::make_bc<samurai::Dirichlet<1>>(u, 0.0)->on(rim);
  ```

Notes

- Passing one or several direction vectors selects those sides (internally handled like `OnDirection`).
- Passing a coordinate predicate builds a `CoordsRegion` at runtime.
- Passing a subset expression is accepted and selects the corresponding boundary faces that intersect the subset.

## Subsets: building custom regions of cells

Subsets are set-expressions evaluated at a given level. Core building blocks (from `samurai/subset/*`):

- Start from a level cell array: `self(lca).on(level)`
- Set algebra:
  - `intersection(a, b, ...)`
  - `union_(a, b, ...)`
  - `difference(a, b)`
  - `translate(set, direction)`
  - `contract(set, width)` in all directions
  - `expand<width>(set)` in all Cartesian directions

Evaluate and iterate:

```cpp
auto L = mesh.max_level();
auto interior = samurai::self(mesh.domain()).on(L);
auto band = samurai::difference(interior, samurai::contract(interior, 1));

// Per-cell iteration on a subset
samurai::for_each_cell(mesh, band, [&](auto& cell) {
  // work
});

// Or interval-level iteration (advanced)
samurai::apply(band, [](const auto& i, const auto& index){
  // i is an interval [start, end), index are the transverse coordinates
});
```

## Practical recipes

- One layer of boundary cells on the left side at level L:

  ```cpp
  const xt::xtensor_fixed<int, xt::xshape<2>> left{-1, 0};
  auto cells = samurai::domain_boundary(mesh, L, left);
  ```

- Two ghost layers outside the top boundary (2D):

  ```cpp
  const xt::xtensor_fixed<int, xt::xshape<2>> top{0, 1};
  auto ghosts = samurai::domain_boundary_outer_layer(mesh, L, top, 2);
  ```

- Restrict a Dirichlet BC to the top side only:

  ```cpp
  samurai::make_bc<samurai::Dirichlet<1>>(u, 0.0)->on(top);
  ```

## Remarks

- Always choose the level explicitly for subset expressions with `.on(level)`.
- Boundary helpers already take a `level` and directly return a `LevelCellArray` at that level.
- For ghost-based schemes, ensure your mesh configuration provides enough ghost width for the BC order you use (see `doc_attempt/06_boundary_conditions.md`).
