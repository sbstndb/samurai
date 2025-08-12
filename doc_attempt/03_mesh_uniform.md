### Uniform meshes

Header: `samurai/uniform_mesh.hpp`

### Overview
- **Type**: `samurai::UniformMesh<Config>`
- **What it is**: a single-level, axis-aligned uniform mesh with optional ghost layers.
- **Use it for**: quick structured meshes in 1D/2D/3D to run finite-volume schemes or build/test fields without multiresolution.

### Configuration type
Use `samurai::UniformConfig<dim, ghost_width = samurai::default_config::ghost_width, TInterval = samurai::default_config::interval_t>`.
- **dim**: spatial dimension (1, 2, or 3)
- **ghost_width**: number of ghost layers to build around interior cells
- **interval_t**: interval/indexing type (advanced)

Mesh identifiers: `samurai::UniformMeshId`
- **cells**: interior cells only
- **cells_and_ghosts**: interior plus ghost layers (width = `Config::ghost_width`)
- **reference**: alias of `cells_and_ghosts`, used as the indexing reference

### Construction
Entry points:
- `UniformMesh(const samurai::Box<double, dim>& box, std::size_t level, double approx_box_tol = ca_type::default_approx_box_tol, double scaling_factor = 0)`
- `UniformMesh(const LevelCellList<dim, interval_t>&)`
- `UniformMesh(const LevelCellArray<dim, interval_t>&)`

Notes on domain approximation (for the `Box` constructor):
- `approx_box_tol` controls the tolerance used to approximate the requested physical box onto a uniform index grid. Default is `0.05`.
- `scaling_factor` sets the physical size scale; cell length at level L is `scaling_factor / (1 << L)`. If 0, a suitable value is computed from the box and tolerance.

### Queries and access
- **Mesh sets**: `mesh[mesh_id]` returns a `LevelCellArray` for `mesh_id in {cells, cells_and_ghosts, reference}`.
- **Counts**: `nb_cells(mesh_id)` returns the number of cells in the requested set.
- **Geometry**: `cell_length(level)` gives the uniform cell size at `level`.
- **Origin & scale**: `origin_point()`, `set_origin_point(...)`, `scaling_factor()`, `set_scaling_factor(...)`.
- **Index helpers**:
  - `get_index(i, j, k, ...)` returns the linear index of a cell within a field storage.
  - `get_interval(level, interval, index...)` retrieves the x-interval object for given indices.

### Iteration utilities
`samurai::for_each_cell(mesh, lambda)` and `samurai::for_each_interval(mesh[mesh_id], lambda)` work with uniform meshes.

### Examples
Minimal 1D construction
```cpp
using Conf = samurai::UniformConfig<1>;
samurai::Box<double, 1> box({-1}, {1});
samurai::UniformMesh<Conf> mesh(box, /*level=*/6);

double dx = mesh.cell_length(6);
std::size_t n_interior = mesh.nb_cells(Conf::mesh_id_t::cells);
```

2D mesh with a wider ghost region and basic iteration
```cpp
constexpr std::size_t dim = 2;
using Conf  = samurai::UniformConfig<dim, /*ghost_width=*/2>;
using UMesh = samurai::UniformMesh<Conf>;
samurai::Box<double, dim> box({0.0, 0.0}, {1.0, 1.0});
UMesh mesh(box, /*level=*/4);

// Access interior cells and ghosts
using mesh_id_t = Conf::mesh_id_t;
auto dx = mesh.cell_length(4);
auto n  = mesh.nb_cells(mesh_id_t::cells);

// Iterate over all cells (interior only)
samurai::for_each_cell(mesh,
    [&](const auto& cell) {
        // cell.level == 4, cell.center(), cell.length == dx, cell.index
    });

// Iterate by intervals at interior set
samurai::for_each_interval(mesh[mesh_id_t::cells],
    [&](std::size_t /*level*/, const auto& i, const auto& index_yz) {
        // i.start..i.end-1 at coordinates index_yz
    });

// Compute linear index from integer coordinates
std::size_t lin = mesh.get_index(/*i=*/3, /*j=*/5);
```

Fields on a uniform mesh (scalar field, fill, save)
```cpp
using Conf = samurai::UniformConfig<1>;
samurai::Box<double, 1> box({0.0}, {1.0});
samurai::UniformMesh<Conf> mesh(box, 5);

auto u = samurai::make_scalar_field<double>("u", mesh);
u.fill(0.0);

samurai::for_each_cell(mesh, [&](const auto& cell) {
    const double x = cell.center(0);
    u[cell] = std::sin(2.0 * M_PI * x);
});

// Optional: save mesh and field (HDF5)
samurai::save("uniform_1d", mesh, u);
```

### Practical notes
- **Ghosts**: `cells_and_ghosts` extends the interior by `Config::ghost_width` in each Cartesian direction and is used for stencil/BC updates.
- **Reference set**: `reference` is the same set as `cells_and_ghosts` and defines consistent linear indices across sets.
- **Cell size**: `cell_length(L) = scaling_factor() / (1 << L)`.
- **Box approximation**: For boxes that are not exactly representable on the chosen grid, increase `approx_box_tol` or set a suitable `scaling_factor`.


