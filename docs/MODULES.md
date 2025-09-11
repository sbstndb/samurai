# C++20 Modules: Quick Start (advection_2d)

This guide explains how to build and benchmark the `finite-volume-advection-2d` demo
with and without C++20 modules on this repository.

Currently, a small part of the demo (init/save helpers) is provided as the module
`samurai.advection_utils` to validate the toolchain and CMake workflow. The rest
of Samurai still uses headers. This is an incremental step toward wider adoption.

## Prerequisites

- CMake ≥ 3.28
- Ninja
- Clang 20 toolchain with the scanner:
  - `clang-20`, `clang++-20`, `clang-scan-deps-20` on PATH

Check:

```
clang++-20 --version
which clang-scan-deps-20
cmake --version
ninja --version
```

## Build (modules ON)

The module path is enabled when the toolchain supports it and when
`SAMURAI_USE_MODULES=ON` (default ON). Use Clang + Ninja:

```
cmake -S . -B build-clang20-ninja -G Ninja \
  -DCMAKE_CXX_COMPILER=clang++-20 -DCMAKE_C_COMPILER=clang-20 \
  -DBUILD_DEMOS=ON -DWITH_PETSC=OFF -DSAMURAI_USE_MODULES=ON

cmake --build build-clang20-ninja -j --target finite-volume-advection-2d
```

The build will compile the BMI once (`samurai.advection_utils.pcm`) and import it
in `advection_2d.cpp`.

Run the binary:

```
build-clang20-ninja/demos/FiniteVolume/finite-volume-advection-2d --Tf 0.01
```

## Build (modules OFF)

To compare against a header-only path with the same compiler and generator, reuse
the same build directory and toggle the option:

```
cmake -S . -B build-clang20-ninja -G Ninja \
  -DCMAKE_CXX_COMPILER=clang++-20 -DCMAKE_C_COMPILER=clang-20 \
  -DBUILD_DEMOS=ON -DWITH_PETSC=OFF -DSAMURAI_USE_MODULES=OFF

cmake --build build-clang20-ninja -j --target finite-volume-advection-2d
```

Note: You can also use `Unix Makefiles` or GCC, but then modules will be disabled
by design and you won’t be exercising the module pipeline.

## Benchmark compile times

A helper script toggles the option in-place (reusing the same build tree to avoid
re-downloading dependencies) and times the compile of the demo target:

```
scripts/benchmark_modules.sh
```

It performs:

1. Configure Clang+Ninja with `SAMURAI_USE_MODULES=OFF`, clean, build the target.
2. Configure Clang+Ninja with `SAMURAI_USE_MODULES=ON`, clean, build the target.

Environment variables:

- `CXX`/`CC` (defaults: `clang++-20`/`clang-20`)
- `GEN` (default: `Ninja`)
- `BUILD_DIR` (default: `build-mod-bench`)
- `TARGET` (default: `finite-volume-advection-2d`)

### Sample result (this machine)

For a cold build of the single demo target in this environment:

```
Modules OFF build time: 79s
Modules ON build time: 88s
```

Interpretation:

- This is expected at this stage: only a tiny piece is modularized and there’s
  no cross-translation-unit reuse yet, so the module build adds some setup cost
  without providing amortized wins.
- As more widely used headers are converted into modules (e.g., core interval/box,
  mesh/field layers), and multiple TUs import the same BMI, you can expect faster
  incremental and multi-target builds because the module is compiled once.

## Troubleshooting

- Modules not enabled:
  - Ensure Clang + Ninja are used and `clang-scan-deps-20` is on PATH.
  - Check `SAMURAI_USE_MODULES` is set to `ON` and CMake ≥ 3.28.
  - CMake will print a status line explaining why modules were disabled.

- Full rebuild after toggling ON/OFF:
  - The script cleans the target; you can also `ninja -t clean` in the build dir.

- GCC builds:
  - GCC support is evolving; this setup currently enables the module path for
    Clang/VS + Ninja/VS only and falls back to headers elsewhere.

