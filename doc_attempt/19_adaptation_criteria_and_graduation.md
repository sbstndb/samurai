# Graduation (mesh grading) and adaptation

This page explains the graduation (grading) constraints enforced in Samurai during mesh adaptation, what they guarantee, and how to use them. It focuses on user-facing behavior while citing the underlying mechanisms for clarity.

## What is graduation?

Graduation ensures that neighboring cells do not differ by more than one refinement level with respect to a chosen stencil neighborhood. In practice, when some cells are marked for refinement (or coarsening), the tagging is expanded so that resulting meshes are graded:

- No fine cell at level ℓ may directly touch a much coarser cell at level < ℓ−1 in any stencil direction.
- The guarantee is tested using a star stencil of configurable width.

This prevents pathological flux stencils crossing arbitrarily large level jumps and ensures consistent prediction/projection across levels.

## Configuration knobs

- `graduation_width` (compile-time via mesh config): width of the neighborhood used to propagate refine/keep tags to enforce grading. See `MRConfig` or `amr::mesh` configs.
- Numerical stencil width: the “half-stencil width” of your FV operator may require additional boundary-contiguity guarantees (see below). Samurai accounts for this when building the graded mesh if you pass it.

## Typical workflow (user level)

1. Tagging: you fill a per-cell tag array with values from `CellFlag` (e.g., `refine`, `keep`, `coarsen`).
2. Build the new mesh from tags: Samurai expands tags to enforce grading and returns the new `CellArray` or mesh.
3. Transfer fields to the new mesh: persistent cells are copied, refined cells are predicted from parents, coarse cells are projected from children.

Minimal API patterns

```cpp
// From a tag field to a new cell array with grading
auto new_ca = samurai::update_cell_array_from_tag(old_ca, tag);

// Or directly enforce grading on a cell array (no domain/periodicity constraints)
samurai::make_graduation(ca /*, grad_width = 1 */);

// Full control (periodicity/domain and numerical half-stencil width)
samurai::make_graduation(ca,
                         domain,                  // LevelCellArray for the domain (can be empty for full domain)
                         mesh.mpi_neighbourhood(),
                         mesh.periodicity(),
                         mesh_t::config::graduation_width,
                         mesh_t::config::max_stencil_width);
```

## What graduation enforces

- Parent/child consistency: if a fine cell is refined at level ℓ, the overlapping parent at ℓ−1 is marked keep/refine consistently. Siblings are kept together when required to avoid isolated children.
- Cross-level contact: whenever cells at ℓ touch cells at ℓ−2 or coarser in the stencil neighborhood, Samurai marks the missing intermediate layer at ℓ−1 to be refined.
- Boundary contiguity (when needed): for wide numerical stencils, Samurai ensures contiguous boundary cells near the domain boundary across levels to avoid stencils leaving the domain incorrectly.

### Conceptual sketch

```mermaid
graph TD
  A[Fine cells (level L)] -->|touch across stencil| B[Coarse region (≤ L-2)]
  B --> C[Mark intermediate L-1 cells refine]
  C --> D[Repeat until no violations]
```

## Under the hood (verified behavior)

### Tag graduation across levels

```140:197:include/samurai/algorithm/graduation.hpp
for (std::size_t level = max_level; level > 0; --level)
{
    // Tag parents at level-1 as keep/refine wherever their fine children (at level) are keep/refine
    auto ghost_subset = intersection(mesh[mesh_id_t::cells][level], mesh[mesh_id_t::reference][level - 1]).on(level - 1);
    ghost_subset.apply_op(tag_to_keep<0>(tag));

    // Expand refine/keep within a ghost width around fine cells
    auto subset_2 = intersection(mesh[mesh_id_t::cells][level], mesh[mesh_id_t::cells][level]);
    subset_2.apply_op(tag_to_keep<ghost_width>(tag, CellFlag::refine));

    // Keep children together
    auto keep_subset = intersection(mesh[mesh_id_t::cells][level], mesh[mesh_id_t::cells][level]).on(level - 1);
    keep_subset.apply_op(keep_children_together(tag));

    // For each directional stencil offset, graduate parents under shifted children
    for each stencil s:
      subset = intersection(translate(mesh[level], s), mesh[level-1]).on(level)
      subset.apply_op(graduate(tag, s));
}
```

- The star stencil (configurable width) is used to query neighbors for possible cross-level contact that would violate grading.
- The “graduate” operator applies a mask: if a fine cell is refine/keep, its parent gets refine/keep accordingly.

### Detect non-graded locations and refine them

```484:598:include/samurai/algorithm/graduation.hpp
// Build list of coarse intervals to refine so that no fine cell (L) touches a cell at L-2 or lower
list_intervals_to_refine(grad_width, half_stencil_width, ca, domain, mpi_neighbourhood, is_periodic, nb_cells_finest_level, remove_m_all);
// From intervals to refine at level, add children at level+1, remove conflicting coarse, iterate until stable
for levels ...
  new_ca[level] = union(ca[level], add[level]) minus remove[level];
repeat until new_ca == ca
```

- Intervals are encoded in a compact 1D-along-x with per-index “yz” coordinates (higher-dim indices). Samurai uses these to push contiguous runs efficiently.
- Periodic directions generate shifted unions to detect interactions across boundaries.
- For wide numerical stencils near domain boundaries, a specialized pass ensures enough contiguous boundary cells across levels.

### Quick check: is a mesh graded?

```200:229:include/samurai/algorithm/graduation.hpp
bool is_graduated(mesh, stencil=star_stencil(...)):
  for level = min+2 .. max:
    for below = min .. level-2:
      for each stencil offset s:
        if any intersection(translate(mesh[level], s), mesh[below]) is non-empty -> false
  return true
```

## Boundary conditions and wide stencils

For operators with half-stencil width > 1, Samurai ensures additional contiguity at boundaries:

```322:407:include/samurai/algorithm/graduation.hpp
list_interval_to_refine_for_contiguous_boundary_cells(half_stencil_width, ca, domain, is_periodic, out):
  // Jump L -> L-1: need 2*half_stencil_width contiguous at L (minus 1 via projected BC)
  // Jump L -> L+1: ensure at least half_stencil_width contiguous at L+1 when half_stencil_width > 2
```

This avoids constructing ghost values outside the domain when a stencil centered in a cell near the boundary reaches outside.

## Periodicity and MPI

- Periodic directions are handled by constructing shifted unions of fine sets to detect cross-boundary interactions and refine the corresponding coarse layers.
- In MPI, each rank exchanges its cell arrays with neighbor ranks to apply the same detection/refinement logic across subdomain interfaces; intervals to refine are collected per-rank and applied locally.

```251:319:include/samurai/algorithm/graduation.hpp
// In MPI: send cell arrays to neighbors, run the same detection on neighbor vs local ca, then recv and wait_all
```

## Using graduation in your adaptation loop

- Build a tag from your criterion; typical flags: `refine` where detail > ε, `coarsen` where detail < ε (and not keep), `keep` around key regions.
- Expand and enforce grading; build the new mesh:

```cpp
samurai::CellArray<dim> ca = mesh[mesh_id_t::cells];
// Apply grading with domain & periodicity awareness
samurai::make_graduation(ca, mesh.domain(), mesh.mpi_neighbourhood(), mesh.periodicity(),
                         mesh_t::config::graduation_width, mesh_t::config::max_stencil_width);
mesh_t new_mesh{ca, mesh};
```

- Transfer fields to the new mesh; Samurai provides helpers in the adaptation driver.
- Update ghosts if you proceed with low-level stencil ops; otherwise FV operators handle it automatically.

## Tips and constraints

- Choose `graduation_width` at least equal to your stencil neighborhood width for strong guarantees; default values are consistent with FV defaults.
- When using periodicity, ensure your domain bounds align with mesh scaling; periodic shifts are computed from domain indices.
- Large refinement jumps across features will be expanded until all intermediate layers are present; plan CFL/time-step accordingly.

## References

- Implementation: `include/samurai/algorithm/graduation.hpp` (tag graduation, interval lists, refinement loop, boundary contiguity, periodic handling, MPI support).
- Configuration: `include/samurai/mr/mesh.hpp`, `include/samurai/mr/mesh_with_overleaves.hpp`, `include/samurai/samurai_config.hpp`.
- Tests: `tests/test_graduation.cpp` (1D/2D/3D sanity via `make_graduation` and `is_graduated`).

