# CSIR migration notes

Problem encountered when replacing lazy subset `.on(level)` with CSIR pipelines:

- Symptom: wrong cells selected or projection/prediction writing to incorrect indices after converting CSIR results back to LCA, specifically in ghost updates and projection steps.
- Root cause: `from_csir_level(...)` created a `LevelCellArray` with only the target level, dropping mesh geometry (origin and scaling). Samurai's lazy `.on(level)` preserves origin/scaling. Losing them shifts coordinates, breaking operators like `projection(...)` and `variadic_projection(...)` that access parent/child indices across levels.

Fix applied:

- Added geometry-aware overloads in `csir_unified/src/csir.hpp`:
  - `from_csir_level(csir_set, origin, scaling)` for 1D/2D/3D.
- Updated call sites to use:
  - `auto subset_lca = csir::from_csir_level(csir_result, mesh.origin_point(), mesh.scaling_factor());`
  - then wrap with `self(subset_lca)` before `.apply_op(...)`.

Guidelines for future migrations:

- Always level-align before binary ops: use `csir::project_to_level(set, target_level)` so both operands are at the same level.
- After CSIR ops, convert back with geometry: `from_csir_level(result, mesh.origin_point(), mesh.scaling_factor())`.
- Keep domain/subdomain/MPI intersections as-is until a dedicated CSIR path exists for domain boxes and MPI exchanges.
- Prefer small, incremental migrations: replace 1–2 lazy ops at a time, rebuild, and test.

## Key differences: lazy Subset vs. CSIR (eager)

- Lazy `subset` chains (`intersection`, `union_`, `difference`, `.on(level)`) defer work until traversal. CSIR produces concrete level sets immediately.
- With CSIR, you must explicitly:
  - Project inputs to the same level before binary ops.
  - Convert back to `LevelCellArray` with mesh geometry before calling `.apply_op`.

## Core helpers available in `csir_unified/src/csir.hpp`

- Conversions
  - `to_csir_level(const LevelCellArray<Dim,...>&)` → `CSIR_Level[_XD]`
  - `from_csir_level(const CSIR_Level[_XD]& set, Origin, Scale)` → `LevelCellArray<Dim,...>`
  - Box/domain:
    - `to_csir_box(const samurai::Box<T, Dim>& box, std::size_t level)` → CSIR level set for the box at `level`
    - `restrict_to_domain(const CSIR_Level[_XD]& set, const LevelCellArray<Dim,...>& domain, std::size_t level)`
- Level management
  - `project_to_level(const CSIR_Level[_XD]& set, std::size_t level)`
  - `restrict_to_level(...)` alias of `project_to_level`
- Set algebra
  - `intersection(lhs, rhs)`, `union_(lhs, rhs)`, `difference(lhs, rhs)` for 1D/2D/3D
- Neighborhood ops
  - `translate(set, std::array<int, Dim> shift)` (also supports `xt::xtensor_fixed<int, xt::xshape<Dim>>`)
  - `contract(set, width, std::array<bool, Dim> mask)` (1D mask overload provided; for 2D/3D use full-dim versions)
  - `expand(set, width, std::array<bool, Dim> mask)` (1D mask overload provided)
  - `nested_expand(set, width)` for 1D/2D/3D (replacement for `samurai::nestedExpand`)

## Canonical migration pattern (recipe)

Before (lazy):

```cpp
auto subset = intersection(A[level], B[level]).on(level);
subset.apply_op(op(...));
```

After (CSIR):

```cpp
auto A_csir   = csir::to_csir_level(A[level]);
auto B_csir   = csir::to_csir_level(B[level]);
auto A_on     = csir::project_to_level(A_csir, level);
auto B_on     = csir::project_to_level(B_csir, level);
auto inter    = csir::intersection(A_on, B_on);
auto inter_lc = csir::from_csir_level(inter, mesh.origin_point(), mesh.scaling_factor());
self(inter_lc).apply_op(op(...));
```

Notes:
- If `A`/`B` are already `LevelCellArray` at the right level, you may skip `project_to_level` on that operand.
- When the input is a lazy `Subset` (e.g., `translate(self(...).on(lvl), ...)`), materialize to an LCA via `lcl_t` accumulation, then convert to CSIR.

## Translate: direction vectors and types

- Always build `std::array<int, Dim>` (or `xt::xtensor_fixed<int, xt::xshape<Dim>>`) for shifts.
- Extract integer components if you start from `xt` expressions to avoid `xfunction` type issues:

```cpp
std::array<int, Dim> dir{};
for (std::size_t d = 0; d < Dim; ++d) {
    dir[d] = static_cast<int>(direction[d]) * k; // k can be layer width, +/-1, etc.
}
auto shifted = csir::translate(set, dir);
```

## Graduation / interface patterns (level jumps)

- When intersecting fine cells with coarse neighbors, project the coarse set up to the fine level first:

```cpp
auto coarse_csir     = csir::to_csir_level(coarse_lca);
auto coarse_on_fine  = csir::project_to_level(coarse_csir, fine_level);
auto fine_csir       = csir::to_csir_level(fine_lca);
auto inter_csir      = csir::intersection(coarse_on_fine, fine_csir);
auto inter_lca       = csir::from_csir_level(inter_csir, mesh.origin_point(), mesh.scaling_factor());
auto fine_intersect  = self(inter_lca); // keep traversal at fine level
```

## Boundary and ghost recipes

- Outer boundary layer (at `level`, direction `dir`, width `w`):

```cpp
auto dom_on   = csir::project_to_level(csir::to_csir_level(mesh.domain()), level);
auto cells_cs = csir::to_csir_level(mesh[mesh_id_t::cells][level]);
std::array<int, Dim> d{}; for (std::size_t i=0;i<Dim;++i) d[i] = direction[i];
auto outer    = csir::difference(csir::translate(cells_cs, d, static_cast<int>(w)),
                                 csir::translate(dom_on,  d, static_cast<int>(w-1)));
```

- Domain boundary (contracted domain):

```cpp
auto dom_on = csir::project_to_level(csir::to_csir_level(mesh.domain()), level);
std::array<bool, Dim> mask; mask.fill(true);
auto contracted = csir::contract(dom_on, 1, mask);
auto diff       = csir::difference(csir::to_csir_level(mesh[mesh_id_t::cells][level]), contracted);
```

- Projection/prediction ghosts: always intersect with `reference[level_target]` after projecting the ghost source set to `level_target`.

## MPI / periodic neighbors

- Use integer translation vectors; avoid using `xt` expressions directly in CSIR APIs.
- For periodic wrap, translate by lattice vectors, then restrict/intersect with neighbor domains using CSIR:

```cpp
auto sub_csir   = csir::project_to_level(csir::to_csir_level(mesh.subdomain()), level);
auto nbr_csir   = csir::to_csir_level(neighbour.mesh[mesh_id_t::reference][level]);
auto shifted    = csir::translate(sub_csir, shift_array);
auto inter      = csir::intersection(shifted, nbr_csir);
```

## Replacing `nestedExpand`

- Use `csir::nested_expand(set, width)` uniformly in 1D/2D/3D.
- For patterns like `intersection(nestedExpand(U, k), V).on(level)`, do:

```cpp
auto U_csir   = csir::to_csir_level(materialized_U_lca);
auto U_on     = csir::project_to_level(U_csir, level);
auto nexp     = csir::nested_expand(U_on, k);
auto inter    = csir::intersection(nexp, csir::to_csir_level(V[level]));
auto inter_lc = csir::from_csir_level(inter, mesh.origin_point(), mesh.scaling_factor());
```

## Avoid mixing lazy Subset with new LCAs

- Do not chain `translate/difference/union_` lazily on an LCA you just built on the stack, then materialize again later. This has caused invalid indexing and segfaults.
- Prefer going fully CSIR for the whole sub-expression, then convert back once.

## Lazy Subset lifetimes and segfaults (invalid iterator)

- Root cause seen in `domain_boundary_outer_layer`: segfault inside `subset/visitor.hpp` when a lazy `Subset` referenced an LCA temporary that had gone out of scope before traversal. The iterator (`it`) pointed to freed memory.
- Rule: when you build a lazy `Subset` that will be traversed later (e.g., passed to `difference(translate(inner_boundary,...), domain)` or to `for_each_stencil`), ensure the underlying `LevelCellArray` lives at least as long as the traversal.
- Safe patterns:
  - Keep an LCA variable alive in the same scope:
    ```cpp
    auto csir = csir::intersection(...);
    auto lca  = csir::from_csir_level(csir, mesh.origin_point(), mesh.scaling_factor());
    auto subset = self(lca); // lca stays alive until after subset traversal
    ```
  - Or compute the whole thing in CSIR and convert back once.
- Unsafe patterns to avoid:
  - Creating a `self(lca_temp)` or `subset` from an rvalue LCA returned by a helper inside a larger expression; the LCA can be destroyed before traversal starts.

## Boundary layers: correct translate sign and lifetime

- Inner boundary is `cells[level] \ translate(domain[level], -direction)` (minus sign). Using `+direction` flips the face and breaks layer logic.
- When constructing outer layers lazily, keep the inner boundary's LCA alive:
  ```cpp
  auto dom_on   = csir::project_to_level(csir::to_csir_level(mesh.domain()), level);
  auto inner_cs = csir::difference(csir::to_csir_level(mesh[cells][level]), csir::translate(dom_on, -direction_array));
  auto inner_la = csir::from_csir_level(inner_cs, mesh.origin_point(), mesh.scaling_factor());
  auto inner    = self(inner_la);
  auto outer    = difference(translate(inner, layer * direction), domain_subset);
  ```

### Boundary outer layer: lifetime-safe complete recipe

```cpp
// 1) Materialize domain[level] once and keep LCA alive
auto dom_csir   = csir::to_csir_level(mesh.domain());
auto domain_on  = csir::project_to_level(dom_csir, level);
auto domain_lca = csir::from_csir_level(domain_on, mesh.origin_point(), mesh.scaling_factor());
auto domain     = self(domain_lca);

// 2) Build inner boundary from CSIR and keep its LCA alive
auto cells_csir = csir::to_csir_level(mesh[mesh_id_t::cells][level]);
std::array<int, Mesh::dim> dir_neg{}; for (std::size_t k=0;k<Mesh::dim;++k) dir_neg[k] = -direction[k];
auto dom_shift  = csir::translate(domain_on, dir_neg);
auto inner_csir = csir::difference(cells_csir, dom_shift);
auto inner_lca  = csir::from_csir_level(inner_csir, mesh.origin_point(), mesh.scaling_factor());
auto inner      = self(inner_lca);

// 3) Now it is safe to use lazy translates/differences to build outer layers
for (std::size_t layer = 1; layer <= layer_width; ++layer)
{
    auto outer_layer = difference(translate(inner, layer * direction), domain);
    outer_layer([&](const auto& i, const auto& idx){ outer_boundary_lcl[idx].add_interval({i}); });
}
```

Notes:
- Keeping both `domain_lca` and `inner_lca` as named variables avoids dangling references during lazy traversal.
- The sign is `-direction` when translating the domain to create the inner boundary.

## Interface traversal: keep explicit level scoping

- After CSIR intersections used to compute contact sets, use `.on(level)` or `.on(level+1)` on `self(...)` to preserve original traversal semantics:
  ```cpp
  auto inter_lca = csir::from_csir_level(inter_csir, mesh.origin_point(), mesh.scaling_factor());
  auto same_lvl  = self(inter_lca).on(level);
  auto jump_lvl  = self(inter_lca).on(level + 1);
  ```

## When reverting to lazy is safer (for now)

- Complex BC paths (`project_bc`, `predict_bc`, `project_corner_below`) heavily rely on lazy traversal semantics and subset lifetime across nested translates. If a full-CSIR rewrite is not carefully lifetimed, prefer keeping the original lazy chain and only materialize at stable boundaries.
- If you must mix, materialize intermediate results into named LCAs so their lifetime dominates the traversal.

### Current safe choices in BC/update paths
- Keep lazy implementations for `project_bc`, `predict_bc`, and `project_corner_below`, while ensuring any domain/inner sets are materialized to LCAs beforehand when needed.
- Use CSIR for set algebra where both operands are concrete LCAs and immediately convert back with geometry.

## Geometry reminders outside BC

- Always pass geometry in `from_csir_level`:
  - `mesh.hpp` domain construction (added/removed boxes)
  - PETSc FV assembly (`set_0_for_all_ghosts`) when materializing ghost sets
- Missing geometry can shift indices silently and only surface during traversal or assembly.

## Crash signature to watch for

- Backtrace shows `subset/visitor.hpp` with an invalid `it` (e.g., `Cannot access memory at address 0x10`), typically at:
  ```
  visitor.hpp:94  auto i = it->start << m_shift2ref;
  ```
- This almost always means a lazy subset is reading from a destroyed LCA. Fix by extending the LCA lifetime or by using CSIR to compute a final LCA before traversal.

## Known pitfalls and fixes encountered

- Includes
  - Use `#include "csir_unified/src/csir.hpp"` (not `csir.hpp`).
- Missing overloads / name lookup
  - Template ADL: add forward declarations for `to_csir_level`, `project_to_level`, `intersection` in `csir.hpp` (done).
  - Provided 3D overloads for `project_to_level` and `intersection` to satisfy calls.
- Geometry loss
  - Always call `from_csir_level(..., mesh.origin_point(), mesh.scaling_factor())`.
  - Symptom when missing: wrong indices, subtle runtime errors, or segfaults.
- Type issues in translate
  - Convert `xt` expressions to `std::array<int, Dim>` before calling `translate`.
- LCA construction
  - `LevelCellArray` cannot be constructed from `Subset` directly. Materialize via `lcl_t` then build `lca_t`.
- `Tag::dim` / `Field::dim`
  - Replace with `Mesh::dim` in `if constexpr` blocks.
- Syntax / scoping
  - Watch for misplaced `else` or unbalanced braces when refactoring.

## Diagnostics suggestions

- If a migration compiles but crashes:
  - Audit every `from_csir_level` call for geometry arguments.
  - Confirm both operands of a binary op are projected to the same level.
  - Print or assert sizes of LCAs before/after conversion.
  - Reproduce with AddressSanitizer enabled to catch out-of-bounds early.

## Search checklist (ripgrep)

```bash
rg "\\.on\\("                     # lazy level projections
rg "intersection\\(.*\\)\\.on\\("  # lazy intersections
rg "union_\\("                      # lazy unions
rg "difference\\("                 # lazy differences
rg "nestedExpand\\("              # needs csir::nested_expand
rg "translate\\("                 # check shift types
```

## Future improvements (nice-to-have)

- Direct `apply_op` on CSIR sets to avoid round-trip conversions for hot paths.
- Full-dimension mask overloads for `contract/expand` in 2D/3D if not already present.
- Higher-level helpers for periodic boundary handling over boxes/domains.
- Unit tests around conversion geometry to prevent regressions.

