# Testing & TDD Guide (CUDA/Thrust backend)

This document describes the testing strategy we will follow while developing the CUDA/Thrust backend for Samurai. The goals are fast feedback, fine‑grained diagnostics, and deterministic reproducibility.

Principles
- Write tests first (or alongside), keep them very focused, and run continuously.
- Make failures local and explicit: each test should cover a single capability.
- Prefer labeling and regex selection to run only the subset you need.
- Keep commits small and scoped to a single change (test or code). No pushes in this phase.

How to run
- Configure once with tests enabled:
  ```bash
  cmake -S . -B build_test -DBUILD_TESTS=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo
  ```
- Build and run everything:
  ```bash
  cmake --build build_test -j2
  ctest --test-dir build_test -j3 --output-on-failure
  ```
- Run a subset by label or name:
  ```bash
  ctest --test-dir build_test -L cuda-thrust-views --output-on-failure
  ctest --test-dir build_test -R storage_cuda_views --output-on-failure
  ```

Labeling conventions (CTEST)
- `cuda-thrust-views`      – view creation, slicing, shapes, ranges
- `cuda-thrust-ops`        – binary/unary ops, expression build
- `cuda-thrust-assign`     – `noalias(lhs)=rhs` kernels
- `cuda-thrust-inplace`    – `+=`, `-=`, `*=` on views
- `cuda-thrust-reduce`     – `sum`, `sum<axis>`
- `cuda-thrust-field`      – Field integration (`for_each_interval`, iterators)
- `cuda-thrust-algo`       – small algorithmic kernels (prediction/transfer)
- `cuda-thrust-masked`     – `apply_on_masked` semantics (baseline host via UVM, then device)

Test file naming
- `tests/cuda/storage_cuda_views.test.cpp`
- `tests/cuda/storage_cuda_ops.test.cpp`
- `tests/cuda/storage_cuda_assign.test.cpp`
- `tests/cuda/storage_cuda_inplace.test.cpp`
- `tests/cuda/storage_cuda_reduce.test.cpp`
- `tests/cuda/integration_field_scalar_vector.test.cpp`
- `tests/cuda/integration_prediction_transfer.test.cpp`

What each test should verify (non‑exhaustive)
- Views: correct shape; `range(start,end)` slicing; AOS/SoA indexing; `step==1` first, then strides.
- Ops/assign: fusing into a single kernel; numerical equality vs CPU reference on tiny inputs.
- In‑place ops: elementwise updates match a host reference.
- Reductions: `sum` matches reference; later `sum<axis>` for 2D views.
- Masked ops: `apply_on_masked` updates correct subset (baseline host via UVM). Add a device test variant once kernels land.
- Field: scalar/vector fields; `for_each_interval` assigns; iterator chunks equal expected arrays.
- Algorithms: minimal `transfer()` and `prediction()` cases with known outputs.

Diagnostics tips
- Use `--output-on-failure` and optionally `-VV` for verbose CTest logs.
- Seed any random inputs and keep domains tiny (16–128 cells) for quick runs.
- Keep one “golden” CPU reference path per test (xtensor/Eigen backend) and compare against the CUDA backend when feasible.

Commit policy (local only)
- Commit frequently, one logical change per commit.
- Include the failing test in the commit where you implement the fix.
- Do not push during this phase; keep the history local until explicit review.
