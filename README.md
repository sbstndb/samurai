<h1 align="center">
  <a href="https://github.com/hpc-maths/samurai">
    <picture>
        <source media="(prefers-color-scheme: dark)" height="200" srcset="./docs/source/logo/dark_logo.png">
        <img alt="Text changing depending on mode. Light: 'So light!' Dark: 'So dark!'" height=200 src="./docs/source/logo/light_logo.png">
    </picture>
  </a>
</h1>

<div align="center">
  <br />
  <a href="https://github.com/hpc-maths/samurai/issues/new?assignees=&labels=bug&template=01_BUG_REPORT.md&title=bug%3A+">Report a Bug</a>
  ·
  <a href="https://github.com/hpc-maths/samurai/issues/new?assignees=&labels=enhancement&template=02_FEATURE_REQUEST.md&title=feat%3A+">Request a Feature</a>
  ·
  <a href="https://github.com/hpc-maths/samurai/discussions">Ask a Question</a>
  <br />
  <br />
</div>

<div align="center">
<br />

[![Project license](https://img.shields.io/github/license/hpc-maths/samurai.svg?style=flat-square)](LICENSE)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/9ea988d1c63344ca9a3d361a5459df2f)](https://app.codacy.com/gh/hpc-maths/samurai/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)

[![Pull Requests welcome](https://img.shields.io/badge/PRs-welcome-ff69b4.svg?style=flat-square)](https://github.com/hpc-maths/samurai/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22)
[![code with love by hpc-maths](https://img.shields.io/badge/%3C%2F%3E%20with%20%E2%99%A5%20by-HPC@Maths-ff1414.svg?style=flat-square)](https://github.com/hpc-maths)


</div>

The use of mesh adaptation methods in numerical simulation allows to drastically reduce the memory footprint and the computational costs. There are different kinds of methods: AMR patch-based, AMR cell-based, multiresolution cell-based or point-based, ...

Different open source software is available to the community to manage mesh adaptation: [AMReX](https://amrex-codes.github.io/amrex/) for patch-based AMR, [p4est](https://www.p4est.org/) and [pablo](https://optimad.github.io/PABLO/) for cell-based adaptation.

The strength of samurai is that it allows to implement all the above mentioned mesh adaptation methods from the same data structure. The mesh is represented as intervals and a set algebra allows to efficiently search for subsets among these intervals.
Samurai also offers a flexible and pleasant interface to easily implement numerical methods.

<details>
<summary>Table of Contents</summary>

- [Get started](#get-started)
  - [The advection equation](#the-advection-equation)
  - [The projection operator](#the-projection-operator)
  - [There's more](#theres-more)
- [Features](#features)
- [Installation](#installation)
  - [From conda](#from-conda)
  - [From Conan Center](#from-conan-center)
  - [From source](#from-source)
- [Get help](#get-help)
- [Project assistance](#project-assistance)
- [Contributing](#contributing)
- [License](#license)

</details>

## Get started

In this section, we propose two examples: the first one solves a 2D advection equation with mesh adaptation using multiresolution, the second one shows the use of set algebra on intervals.

### The advection equation

We want to solve the 2D advection equation given by

$$
\partial_t u + a \cdot \nabla u = 0 \\; \text{in} \\; [0, 1]\times [0, 1]
$$

with homogeneous Dirichlet boundary conditions and $a = (1, 1)$. The initial solution is given by

$$
u_0(x, y) = \left\\{
\begin{align*}
1 & \\; \text{in} \\; [0.4, 0.6]\times [0.4, 0.6], \\
0 & \\; \text{elsewhere}.
\end{align*}
\right.
$$

To solve this equation, we use the well known [upwind scheme](https://en.wikipedia.org/wiki/Upwind_scheme).

The following steps describe how to solve this problem with samurai. It is important to note that these steps are generally the same whatever the equations we want to solve.

- Define the configuration of the problem

    ```cpp
    constexpr size_t dim = 2;
    using Config = samurai::MRConfig<dim>;
    std::size_t min_level = 2, max_level = 8;
    ````

- Create the Cartesian mesh

    ```cpp
    const samurai::Box<double, dim> box({0., 0.}, {1., 1.});
    samurai::MRMesh<Config> mesh(box, min_level, max_level);
    ```

- Create the field on this mesh

    ```cpp
    auto u = samurai::make_field<double, 1>("u", mesh);
    samurai::make_bc<samurai::Dirichlet<1>>(u, 0.);
    ```

- Initialization of this field

    ```cpp
    samurai::for_each_cell(mesh, [&](const auto& cell)
    {
        double length = 0.2;
        if (xt::all(xt::abs(cell.center() - 0.5) <= 0.5*length))
        {
            u[cell] = 1;
        }
    });
    ````

- Create the adaptation method

    ```cpp
    auto MRadaptation = samurai::make_MRAdapt(u);
    ```

- Time loop

    ```cpp
    double dx = mesh.cell_length(max_level);
    double dt = 0.5*dx;
    auto unp1 = samurai::make_field<double, 1>("u", mesh);

    // Time loop
    for (std::size_t nite = 0; nite < 50; ++nite)
    {
        // adapt u
        MRadaptation(1e-4, 2);

        // update the ghosts used by the upwind scheme
        samurai::update_ghost_mr(u);

        // upwind scheme
        samurai::for_each_interval(mesh, [&](std::size_t level, const auto& i, const auto& index)
        {
            double dx = mesh.cell_length(level);
            auto j = index[0];

            unp1(level, i, j) = u(level, i, j) - dt / dx * (u(level, i, j) - u(level, i - 1, j)
                                                          + u(level, i, j) - u(level, i, j - 1));
        });

        std::swap(unp1.array(), u.array());
    }
    ```

The whole example can be found [here](./demos/FiniteVolume/advection_2d.cpp).

### The projection operator

When manipulating grids of different resolution levels, it is often necessary to transmit the solution of a level $l$ to a level $l+1$ and vice versa. We are interested here in the projection operator defined by

$$
u(l, i, j) = \frac{1}{4}\sum_{k_i=0}^1\sum_{k_j=0}^1 u(l+1, 2i + k_i, 2j + k_j)
$$

This operator allows to compute the cell-average value of the solution at a grid node at level $l$ from cell-average values of the solution known on children-nodes at grid level $l + 1$ for a 2D problem.

We assume that we already have a samurai mesh with several level defined in the variable `mesh`. To access to a level, we use the operator `mesh[level]`. We also assume that we created a field on this mesh using the `make_field` and initialized it.

The following steps describe how to implement the projection operator with samurai.

- Create a subset of the mesh using set algebra
```cpp
auto set = samurai::intersection(mesh[level], mesh[level+1]).on(level);
```

- Apply an operator on this subset

```cpp
set([&](const auto& i, const auto index)
{
    auto j = index[0];
    u(level, i, j) = 0.25*(u(level+1, 2*i, 2*j)
                         + u(level+1, 2*i+1, 2*j)
                         + u(level+1, 2*i, 2*j+1)
                         + u(level+1, 2*i+1, 2*j+1));
});
```

The multi dimensional projection operator can be found [here](./include/samurai/numeric/projection.hpp).

### There's more

If you want to learn more about samurai skills by looking at examples, we encourage you to browse the [demos](./demos) directory.

The [tutorial](./demos/tutorial/) directory is a good first step followed by the [FiniteVolume](./demos/FiniteVolume/) directory.

## Features

- [x] Facilitate data manipulation by using the formalism on a uniform Cartesian grid
- [x] Facilitate the implementation of complex operators between grid levels
- [x] High memory compression of an adapted mesh
- [x] Complex mesh creation using a set of meshes
- [x] Finite volume methods using flux construction
- [x] Lattice Boltzmann methods examples
- [ ] Finite difference methods
- [ ] Discontinuous Galerkin methods
- [x] Matrix assembling of the discrete operators using PETSc
- [x] AMR cell-based methods
- [ ] AMR patch-based and block-based methods
- [x] MRA cell-based methods
- [ ] MRA point-based methods
- [x] HDF5 output format support
- [ ] MPI implementation

## Installation

### From conda

```bash
mamba install samurai
```

For compiling purposes, you have to install a C++ compiler, `cmake`, and (optionaly) `make`:

```bash
mamba install cxx-compiler cmake [make]
```

If you have to use PETSc to assemble the matrix of your problem, you need to install it:

```bash
mamba install petsc pkg-config
```

For parallel computation,

```bash
mamba install libboost-mpi libboost-devel libboost-headers 'hdf5=*=mpi*'
```

### From Conan Center

If you want to install samurai from Conan, you can use the following command:

```bash
conan install --requires=samurai/0.13.0
```

### From source

Run the cmake configuration

- With mamba or conda

    First, you need to create the environment with all the dependencies
    installed, run

    ```bash
    mamba env create --file conda/environment.yml
    ```

    for sequential computation, or

    ```bash
    mamba env create --file conda/mpi-environment.yml
    ```

    for parallel computation. Then

    ```bash
    mamba activate samurai-env
    ```

    ```bash
    cmake . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_DEMOS=ON
    ```

- With vcpkg

    ```bash
    cmake . -B ./build -DENABLE_VCPKG=ON -DBUILD_DEMOS=ON
    ```

- With conan

    ```bash
    cmake . -B ./build -DCMAKE_BUILD_TYPE=Release -DENABLE_CONAN_OPTION=ON -DBUILD_DEMOS=ON
    ```

Build the demos

```bash
cmake --build ./build --config Release
```

## CMake configuration

Here is a minimal example of `CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.15)
set(CMAKE_CXX_STANDARD 17)

project(my_samurai_project CXX)

set(SAMURAI_WITH_MPI ON)
set(SAMURAI_WITH_PETSC OFF)
find_package(samurai CONFIG REQUIRED)

add_executable(my_samurai_project main.cpp)
target_link_libraries(my_samurai_project PRIVATE samurai::samurai)
```

## Developer notes (CUDA/Thrust + TDD)

We are introducing an experimental CUDA backend for Samurai based on Thrust. The work is organized with test‑driven development so that each capability is covered by targeted tests and validated against the existing suite and demo programs.

TDD roadmap (high level):
- Storage/view unit tests: slicing (`view`), assignment (`noalias(lhs)=rhs`), in‑place ops (`+=`, `*=`), and reductions (`sum`).
- Field integration tests: scalar/vector fields over intervals, `for_each_interval`, and iterator semantics (`*it` equals expected chunks).
- Algorithmic smoke tests: prediction/transfer minimal cases on uniform meshes.
- Demo build checks: compile a subset of demos and run with small grids.
- Masked operations: `apply_on_masked` supported from day one (functional baseline), with a device variant planned next.

Running tests and demos side by side:
- Configure once with demos and tests enabled:
  ```bash
  cmake -S . -B build_test -DBUILD_DEMOS=ON -DBUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
  ```
- Build all tests and a specific demo target:
  ```bash
  # Important: limit parallelism to avoid RAM pressure
  cmake --build build_test -j2
  cmake --build build_test --target finite-volume-advection-2d -j2
  ```
- Run the test suite:
  ```bash
  # Tests can run with at most 3 workers
  ctest --test-dir build_test -j3
  ```

Notes:
- If you see “No rule to make target …”, make sure the CMake configure step completed without interruption and that you are in the build directory created by CMake.
- Demo target names come from `demos/*/CMakeLists.txt` (`add_executable`). For example: `finite-volume-advection-1d`, `finite-volume-advection-2d`, `finite-volume-burgers`, `finite-volume-level-set`, `pablo-bubble-2d`, `tutorial-burgers1d-step-0` …

- For a detailed TDD guide and labeling scheme, see `TESTING.md`.

### Spack workflow (optional)

If you use Spack to setup dependencies and environment:

```bash
. ~/spack/share/spack/setup-env.sh
spack load samurai@0.26.1
spack load cli11/5
cd ~/sbstndbs/samurai_cuda
mkdir -p build_test && cd build_test
cmake -DBUILD_DEMOS=ON -DBUILD_TESTS=ON ..

# Build a specific demo (example):
cmake --build . --target finite-volume-advection-2d -j2

# Run tests
ctest -j3
```

Tip: you can discover available demo targets with

```bash
rg -n "add_executable\(" demos | sed 's/.*add_executable(\([^ ]*\).*/\1/'
```

### Workflow de développement (FR)

- Nous testons en continu pour valider chaque étape.
- Les tests sont écrits de manière très fine, afin d’isoler la sous‑partie en cause en cas d’échec.
- Nous ajoutons des tests spécifiques (backend CUDA) tout en vérifiant la suite existante et certaines démos.
- Commits élémentaires et atomiques; pas de `git push` durant cette phase d’exploration.
- Utiliser les labels CTest pour cibler rapidement:
  ```bash
  ctest --test-dir build_test -L cuda-thrust-views --output-on-failure
  ctest --test-dir build_test -R storage_cuda_views --output-on-failure
  ```

Restrictions initiales du backend CUDA
- Mémoire unifiée CUDA (UVM) activée par défaut pour préserver les accès host (`field[cell]`).
- Layout supporté: row‑major uniquement au départ (erreur CMake si col‑major est demandé avec `-DSAMURAI_FIELD_CONTAINER=cuda`).
- `apply_on_masked`:
  - Implémentation initiale de référence (M1): exécution côté host via UVM (fonctionnelle et validée par tests).
  - Variante device optimisée (M2): via `where(mask, expr_true, expr_false)` et/ou `thrust::for_each_n` avec prédicats device.

## Get help

For a better understanding of all the components of samurai, you can consult the documentation https://hpc-math-samurai.readthedocs.io.

If you have any question or remark, you can write a message on [github discussions](https://github.com/hpc-maths/samurai/discussions) and we will be happy do help you or to discuss with you.


## Project assistance

If you want to say **thank you** or/and support active development of samurai:

- Add a [GitHub Star](https://github.com/hpc-maths/samurai) to the project.
- Tweet about samurai.
- Write interesting articles about the project on [Dev.to](https://dev.to/), [Medium](https://medium.com/) or your personal blog.

Together, we can make samurai **better**!

## Contributing

First off, thanks for taking the time to contribute! Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make will benefit everybody else and are **greatly appreciated**.


Please read [our contribution guidelines](./docs/CONTRIBUTING.md), and thank you for being involved!

## License

This project is licensed under the **BSD license**.

See [LICENSE](LICENSE) for more information.
