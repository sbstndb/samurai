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
