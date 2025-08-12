# Multiresolution meshes

Header: `samurai/mr/mesh.hpp`

- Type: `samurai::MRMesh<MRConfig<...>>`
  - `MRConfig<dim, max_stencil_width, graduation_width, prediction_order, max_refinement_level, interval_t>`
  - Mesh ids: `cells`, `cells_and_ghosts`, `proj_cells`, `union_cells`, `all_cells` (aka `reference`)

## What it is

`MRMesh` is a multi-level Cartesian mesh that stores active cells per level and the additional layers needed by multiresolution (projection/prediction) and finite-volume stencils. It exposes simple, uniform methods to query geometry and iterate over cells/intervals across levels.

Typical use appears in demos such as `FiniteVolume/advection_1d.cpp` and `FiniteVolume/burgers_mra.cpp`.

## Configure the mesh type

```cpp
using Config = samurai::MRConfig<2>;               // 2D, defaults for stencil and prediction
// or customize if needed:
// using Config = samurai::MRConfig<2, /*max_stencil_width=*/2, /*graduation_width=*/2,
//                                  /*prediction_order=*/2, /*max_refinement_level=*/20>;
```

## Construct a multiresolution mesh

- From a box and min/max levels:

```cpp
samurai::Box<double,2> box({-1,-1}, {1,1});
samurai::MRMesh<Config> mesh(box, /*min_level=*/3, /*max_level=*/6);
```

- Periodic domain (per dimension):

```cpp
constexpr std::size_t dim = 1;
std::array<bool, dim> periodic{true};
samurai::Box<double,dim> box({-2.}, {2.});
samurai::MRMesh<samurai::MRConfig<dim>> mesh(box, 6, 12, periodic);
```

- From complex domains (holes/added boxes): use the constructor taking a `DomainBuilder<dim>` when building non-rectangular domains. See the IO and geometry sections for details.

## Key properties and queries

- Levels: `min_level()`, `max_level()`
- Sizes: `nb_cells()` and `nb_cells(level)` for counts on a given mesh id (see below)
- Geometry: `cell_length(level)`, `origin_point()`, `scaling_factor()`
- Domain subsets: `domain()`, `subdomain()`
- Periodicity: `periodicity()`, `is_periodic(d)`
- Indexing helpers:
  - `get_interval(level, interval, index_yz)` → canonical interval with linear offset
  - `get_index(level, i, j, ...)` → linear index within level
  - `get_cell(level, i, j, ...)` → a `Cell` with coordinates, center and length

Example (time step on finest level):

```cpp
double dx = mesh.cell_length(mesh.max_level());
double dt = cfl * dx;
```

## Mesh ids and when to use them

Access different cell sets with `mesh[mesh_id_t::X]`:

- `cells`: active cells only (default set used by iteration helpers)
- `cells_and_ghosts`: active cells plus ghost layers sized for `max_stencil_width`
- `union_cells`: level-wise union from fine to coarse, used for reconstruction and projections
- `proj_cells`: cells at level ℓ that are projections of finer cells at ℓ+1
- `all_cells` (alias `reference`): superset containing what is required by multiresolution operations and periodic/MPI exchanges

Notes:

- Iteration helpers that take a mesh use `cells` by default. To iterate on ghosts, pass the desired `CellArray`: `for_each_cell(mesh[mesh_id_t::cells_and_ghosts], fn)`.

## Iterate over levels, intervals and cells

Helpers in `samurai/algorithm.hpp` work on `MRMesh` and its `CellArray` views:

- Over levels:

```cpp
samurai::for_each_level(mesh, [&](std::size_t level){ /* ... */ });
```

- Over intervals per level (fast, per contiguous x-interval at fixed y/z):

```cpp
samurai::for_each_interval(mesh[mesh_id_t::cells],
  [&](std::size_t level, const auto& i, const auto& index_yz) {
    // i.start .. i.end at this level and lateral index index_yz
  });
```

- Over cells:

```cpp
samurai::for_each_cell(mesh, [&](const auto& cell){
  auto c = cell.center();
  (void)c; // use center or length
});
```

## Lookup and indexing

- Linear index at a level:

```cpp
auto idx = mesh.get_index(level, /*i=*/ix, /*j=*/iy);
```

- Retrieve a `Cell` from integer coordinates:

```cpp
auto cell = mesh.get_cell(level, ix, iy);
```

- Find the cell that contains a Cartesian point (across levels):

```cpp
auto found = samurai::find_cell(mesh, xt::xtensor_fixed<double, xt::xshape<2>>{x, y});
if (found.length != 0) { /* found.level, found.index, found.center() */ }
```

## Domain and periodicity

- `domain()` is the global mesh domain; `subdomain()` is identical in serial runs and is used for domain partitioning under MPI.
- Periodicity is per dimension and is set at construction. Query with `is_periodic(d)` or `periodicity()[d]`.

Example (1D, apply boundary conditions only if non-periodic):

```cpp
bool is_periodic = false;
samurai::MRMesh<samurai::MRConfig<1>> mesh({{-2.}, {2.}}, 6, 12, std::array<bool,1>{is_periodic});
auto u = samurai::make_scalar_field<double>("u", mesh);
if (!is_periodic) {
  const xt::xtensor_fixed<int, xt::xshape<1>> left{-1}, right{1};
  samurai::make_bc<samurai::Dirichlet<1>>(u, 0.)->on(left, right);
}
```

## Practical patterns seen in demos

- Build `MRMesh`, make field(s), compute `dt` from `cell_length(max_level)`, run a scheme, and adapt the mesh between steps with `make_MRAdapt(u)` (see multiresolution adaptation section for details).
- Save results with the IO helpers. See the IO section and demos for complete examples.

## Where to see it in action

- `demos/FiniteVolume/advection_1d.cpp`: periodic toggle, time step from `cell_length`, adaptation loop
- `demos/FiniteVolume/burgers_mra.cpp`: two meshes (adaptive and max-level), reconstruction and error reporting
