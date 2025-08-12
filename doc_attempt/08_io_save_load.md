# I/O: save, dump, restart

Headers: `samurai/io/hdf5.hpp`, `samurai/io/restart.hpp`, `samurai/io/util.hpp`

## Saving XDMF/HDF5 for visualization

Public API (overloads):

- `save(path, filename, mesh, fields...)`
- `save(path, filename, Hdf5Options<Mesh>{...}, mesh, fields...)`
- `save(filename, mesh, fields...)`  // uses current working directory
- `save(filename, { ... }, mesh, fields...)`  // brace-initialize options

What is written

- `.h5` file containing mesh connectivity/points and one dataset per field component
- `.xdmf` descriptor next to it. In MPI, the `.xdmf` is written by rank 0
- If `path` does not exist, it is created

Options

- For general meshes (multiresolution or cell arrays): `Hdf5Options<Mesh>{by_level, by_mesh_id}`
- For uniform meshes: `Hdf5Options<UniformMesh<Config>>{by_mesh_id}`
  - `by_level` is ignored for uniform meshes

Examples

```cpp
// Minimal: save a mesh and one scalar field
samurai::save(path, filename, mesh, u);

// With options (multiresolution meshes): by level and by mesh id
samurai::save(path, filename, samurai::Hdf5Options<decltype(mesh)>{true, true}, mesh, u);

// Brace-init options variant (same effect)
samurai::save(filename, {true, true}, mesh, u);

// Uniform mesh: only by_mesh_id is applicable
samurai::save(path, filename, samurai::Hdf5Options<decltype(mesh)>{true}, mesh, u);
```

Field naming and components

- Fields must have a name (e.g., created via `make_scalar_field<double>("u", mesh)`)
- Vector fields are saved as separate components with suffixed names (`name_0`, `name_1`, ...)

Debug fields (optional)

- Enabling `--save-debug-fields` adds `indices`, `coordinates`, and `levels` fields automatically

## Restart snapshots (compact HDF5)

Public API (overloads):

- `dump(path, filename, mesh, fields...)`  // writes `filename.h5`
- `dump(filename, mesh, fields...)`
- `load(path, filename, mesh, fields&...)`
- `load(filename, mesh, fields&...)`

Behavior

- `dump` stores mesh topology and optional fields in a compact, partitioned HDF5 layout
- `load` reconstructs the mesh and resizes/fills the provided fields
- In MPI, the number of ranks must match between `dump` and `load` (checked via `n_process`)

Examples

```cpp
// Create a checkpoint
samurai::dump(path, filename + "_restart", mesh, u /*, more fields ...*/);

// Restore later
samurai::load(path, filename + "_restart", mesh, u /*, same field names/components */);
```

Notes

- On load, fields are matched by name and component count; they must exist with the same names

## Custom I/O helpers

- `extract_coords_and_connectivity(mesh)`
- `extract_data(field, submesh)`
- `extract_data_as_vector(field, submesh)`
