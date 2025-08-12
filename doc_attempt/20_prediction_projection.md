# Prediction and projection (multilevel transfers)
This page explains how Samurai predicts values from coarse to fine levels and projects values from fine to coarse levels on multiresolution meshes. It focuses on user-facing usage with verified behavior from the numeric kernels.

## Why prediction/projection?

- When adapting the mesh, values on new refined cells must be initialized from their coarse parent consistently (prediction).
- When coarsening or building coarse-level ghosts, consistent averages of fine children are needed (projection).
- FV schemes and MRA logic rely on these transfers to maintain accuracy and stability across levels.

## Quick usage

- Default prediction order is driven by the mesh configuration (`prediction_order` in `MRConfig` or related config). The library provides helpers and applies them in the MRA pipeline and FV operators as needed.

- Predict one field to a finer level (conceptually):

```cpp
// new_field will be filled from old_field using the configured prediction order
auto pred = samurai::prediction<mesh_t::config::prediction_order, true>(new_field, old_field);
pred.apply(mesh); // usually called internally by Samurai during adaptation
```

- Variadic form applies the same operation to multiple fields:

```cpp
auto pred_many = samurai::variadic_prediction<mesh_t::config::prediction_order, true>(u, v, w);
pred_many.apply(mesh);
```

Note: In practice you rarely need to call these directly; Samurai’s adaptation driver calls prediction/projection as part of `update_fields(...)` and ghost updates.

## What prediction does (verified)

- 1D, order 0 (piecewise constant): duplicates parent value to both children.
- 1D, higher order: children get `parent ± Qs_i`, where `Qs_i` is a correction computed from neighboring coarse cells using precomputed coefficients of order equal to `prediction_order`.
- 2D/3D: tensorized corrections (e.g., `qs_i`, `qs_j`, `qs_k`, cross terms `qs_ij`, `qs_ik`, `qs_jk`, `qs_ijk`) combine to set the eight children values from their parent consistently.

Citations (1D/2D/3D):

```398:451:include/samurai/numeric/prediction.hpp
// 1D, order 0 and order>0 (dest_on_level true)
// 1D, order 0 copies parent; order>0 uses Qs_i corrections
```

```541:562:include/samurai/numeric/prediction.hpp
// 2D, order 0 copies parent; order>0 uses Qs_i, Qs_j, Qs_ij
```

```675:704:include/samurai/numeric/prediction.hpp
// 3D, order 0 copies parent; order>0 uses Qs_i, Qs_j, Qs_k and cross terms
```

- Destination on a specific level vs same-level view:
  - `dest_on_level = true`: write directly on level+1 children from level.
  - `dest_on_level = false`: same kernels write into the current level combining even/odd patterns using coarse (level-1) values.

## Projection (fine to coarse)

- Projection is invoked in the ghost update pipeline to build coarse values from fine children and in some BC projection steps.
- In user code you normally rely on higher-level helpers (e.g., `update_ghost_mr`, `update_fields`) that call the proper projection kernels.

Citations (projection in ghost update):

```44:58:include/samurai/algorithm/update.hpp
// set_at_levelm1.apply_op(projection(field, fields...));
```

```121:139:include/samurai/algorithm/update.hpp
// projection for BC layers and corners, with controlled layering
```

## Coefficients and stencils

- Coarse-to-fine corrections use central differences built from neighbor coarse cells with coefficients from `prediction_coeffs<s>()`.
- The functions `Qs_i`, `Qs_j`, `Qs_k` and cross terms `Qs_ij`, `Qs_ik`, `Qs_jk`, `Qs_ijk` assemble those corrections; they are then combined algebraically per child.

Citations (coefficients):

```77:108:include/samurai/numeric/prediction.hpp
// prediction_coeffs<order>() and interp_coeffs()
```

## Interaction with adaptation and ghosts

- During MRA (`mr/adapt.hpp`), after building the new mesh (`make_graduation(...)`), Samurai transfers fields by prediction on refined regions and projection on coarsened regions via `update_fields(...)` and a configurable prediction functor (default from `samurai_config.hpp`).
- During `update_ghost_mr(...)`, projection is used to fill coarse ghosts, then fine-level predictions are applied where needed to complete ghost layers while respecting BC and periodic/MPI exchanges.

High-level flow (adaptation):

```text
compute detail → tag cells → sync tags (periodic/MPI) → graduation → build new mesh →
update_fields(prediction, new_mesh, fields...) → swap(new_mesh)
```

High-level flow (ghost update):

```text
outer BC layers (projection) → periodic wrap → inter-rank exchange → projection to level-1 →
prediction to finer levels for remaining ghosts → periodic + inter-rank completion
```

## Practical guidance

- Choose `prediction_order` according to the accuracy you target and the scheme’s smoothness assumptions; higher orders use wider coarse stencils and are more sensitive near boundaries.
- Near non-periodic boundaries, the detail computation avoids using ghosts two levels down; prediction/projection will rely more on local parents, which is by design for stability.
- Don’t call prediction/projection manually unless you implement a custom adaptation pipeline. Prefer `make_MRAdapt(...)` and the provided ghost-update helpers.

## References

- Implementation: `include/samurai/numeric/prediction.hpp` (prediction kernels and coefficients), `include/samurai/algorithm/update.hpp` (projection/prediction during ghost updates).
- Configuration: `include/samurai/samurai_config.hpp` (`prediction_order`, default prediction functor).
- Usage: integration in `mr/adapt.hpp` (`update_fields(...)`).

