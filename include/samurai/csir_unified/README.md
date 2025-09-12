# CSIR Unified: Implémentation, Démos, Tests et Benchmark

Ce dossier contient une implémentation unique et optimisée de CSIR (Compressed Sparse Interval Row) en 2D/3D, avec:

- `csir.hpp`: header-only, opérations d’ensemble (union, intersection, différence), transformations (translate), morpho (contract/expand), projection multi-niveaux 2D/3D (up/down).
- Démos: `main.cpp` (2D), `main_3d.cpp` (3D) qui illustrent projection et opérations.
- Bench: `benchmark_2d.cpp` pour un test simple de perf.
- CMake: `CMakeLists.txt` qui génère 5 exécutables: `demo`, `demo_3d`, `benchmark_2d`, `tests_2d`, `tests_3d`.

## Build

```bash
cmake -S . -B build
cmake --build build -j
```

## Exécution

- Démos:
  - `./build/demo`
  - `./build/demo_3d`

- Bench:
  - `./build/benchmark_2d`

- Tests:
  - `./build/tests_2d`
  - `./build/tests_3d`
