# Plan d'Implémentation : Bindings Python Samurai

**Basé sur** : `/home/sbstndbs/sbstndbs/samurai-worktrees/python-bindings/FEASIBILITY_ANALYSIS.md`
**Date** : 2025-01-05

---

## Ordre d'Implémentation (Par Dépendances)

```
┌─────────────────────────────────────────────────────────────────┐
│                    Phase 0: Infrastructure                       │
│  - CMakeLists.txt pour pybind11                                 │
│  - Structure des répertoires                                    │
│  - Configuration build                                          │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Phase 1: Types Simples                       │
│  Box1D, Box2D, Interval, Cell1D, Cell2D                         │
│  └─> Démontre que pybind11 fonctionne avec xtensor              │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Phase 2: Mesh Uniforme                       │
│  UniformMesh1D, UniformMesh2D                                   │
│  └─> Démontre qu'on peut instancier des classes complexes       │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Phase 3: Itération                           │
│  for_each_cell(), wrappers pour lambdas                        │
│  └─> Démontre qu'on peut appeler du Python depuis C++          │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Phase 4: Champs + NumPy                      │
│  ScalarField1D/2D avec zero-copy                                │
│  └─> Démontre la performance (buffer protocol)                  │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Phase 5: Factory Functions                   │
│  make_scalar_field(), make_mesh()                               │
│  └─> Simplifie l'API Python                                     │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Phase 6: I/O HDF5                            │
│  save(), load()                                                 │
│  └─> Intégration complète avec l'écosystème                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Phase 0 : Infrastructure (JOUR 1)

### 0.1 Structure des Répertoires

```bash
mkdir -p src/python_bindings
mkdir -p python/tests
mkdir -p python/examples
mkdir -p build-python
```

### 0.2 CMakeLists.txt Minimal

```cmake
# src/python_bindings/CMakeLists.txt

cmake_minimum_required(VERSION 3.15)
project(samurai_python)

# Find dependencies
find_package(pybind11 2.10 REQUIRED)
find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development)

# Samurai headers (header-only)
set(SAMURAI_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
include_directories(${SAMURAI_ROOT}/include)

# Create module
pybind11_add_module(samurai_core
    samurai_module.cpp
    box_bindings.cpp
)

# Link xtensor (dependency)
target_link_libraries(samurai_core PRIVATE xtensor)
```

### 0.3 Module Principal Minimal

```cpp
// src/python_bindings/samurai_module.cpp

#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(samurai_core, m) {
    m.doc() = "Samurai Python Bindings - Proof of Concept";
    m.attr("__version__") = "0.0.1";
}
```

### 0.4 Build Test

```bash
cd build-python
cmake ../src/python_bindings -DPYBIND11_PYTHON_VERSION=3.10
make
python3 -c "import samurai_core; print(samurai_core.__version__)"
```

**Sortie attendue** : `0.0.1`

---

## Phase 1 : Types Simples (JOURS 2-5)

### 1.1 Box1D Binding

```cpp
// src/python_bindings/box_bindings.cpp

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <samurai/box.hpp>
#include <xtensor-python/pyarray.hpp>  // Pour numpy array conversion

namespace py = pybind11;

using Box1D = samurai::Box<double, 1>;
using Box2D = samurai::Box<double, 2>;

void bind_box_1d(py::module_& m)
{
    py::class_<Box1D>(m, "Box1D")
        .def(py::init<const xt::pyarray<double>&,
                      const xt::pyarray<double>&>(),
             py::arg("min_corner"), py::arg("max_corner"),
             "Create a 1D box from min and max corners")

        .def_property_readonly("min",
            [](const Box1D& b) -> xt::pyarray<double> {
                return xt::pyarray<double>(b.min_corner());
            },
            "Minimum corner of the box")

        .def_property_readonly("max",
            [](const Box1D& b) -> xt::pyarray<double> {
                return xt::pyarray<double>(b.max_corner());
            },
            "Maximum corner of the box")

        .def_property_readonly("length",
            [](const Box1D& b) -> xt::pyarray<double> {
                return xt::pyarray<double>(b.length());
            },
            "Length of the box in each dimension")

        .def("is_valid", &Box1D::is_valid,
             "Check if the box is valid (min < max)")

        .def("__repr__", [](const Box1D& b) {
            std::ostringstream oss;
            oss << "Box1D(min=" << b.min_corner()[0]
                << ", max=" << b.max_corner()[0] << ")";
            return oss.str();
        });
}
```

### 1.2 Test Python

```python
# python/tests/test_box.py

import pytest
import samurai_core
import numpy as np

def test_box_1d_creation():
    """Test basic Box1D creation."""
    box = samurai_core.Box1D([0.], [1.])
    assert box.is_valid()
    assert np.allclose(box.min, [0.])
    assert np.allclose(box.max, [1.])
    assert np.allclose(box.length, [1.])

def test_box_1d_repr():
    """Test Box1D string representation."""
    box = samurai_core.Box1D([0.], [1.])
    assert "Box1D" in repr(box)
    assert "min=0" in repr(box)

def test_box_2d_creation():
    """Test basic Box2D creation."""
    box = samurai_core.Box2D([0., 0.], [1., 1.])
    assert box.is_valid()
    assert np.allclose(box.length, [1., 1.])
```

### 1.3 Interval Binding

```cpp
// src/python_bindings/interval_bindings.cpp

using interval_t = samurai::Interval<int, long long int>;

py::class_<interval_t>(m, "Interval")
    .def(py::init<int, int, long long int>(),
         py::arg("start"), py::arg("end"), py::arg("index")=0,
         "Create an interval [start, end[")

    .def_readwrite("start", &interval_t::start, "Interval start")
    .def_readwrite("end", &interval_t::end, "Interval end + 1")
    .def_readwrite("step", &interval_t::step, "Step inside the interval")
    .def_readwrite("index", &interval_t::index, "Storage index")

    .def("size", &interval_t::size, "Number of elements in the interval")
    .def("is_valid", &interval_t::is_valid, "Check if interval is not empty")

    .def("contains", &interval_t::contains,
         py::arg("x"), "Check if x is in the interval")

    .def("__repr__", [](const interval_t& i) {
        return fmt::format("Interval([{}, {}[, index={}, step={})",
                          i.start, i.end, i.index, i.step);
    });
```

---

## Phase 2 : Mesh Uniforme (JOURS 6-12)

### 2.1 Types Concrets

```cpp
// src/python_bindings/mesh_types.hpp

#pragma once

#include <samurai/uniform_mesh.hpp>

// Fixer les template parameters pour Python
namespace samurai::python {

// Configuration par défaut
using default_interval = samurai::Interval<int, long long int>;

// Configurations pour chaque dimension
using Config1D = samurai::UniformConfig<1, 1, default_interval>;
using Config2D = samurai::UniformConfig<2, 1, default_interval>;
using Config3D = samurai::UniformConfig<3, 1, default_interval>;

// Maillages uniformes
using UniformMesh1D = samurai::UniformMesh<Config1D>;
using UniformMesh2D = samurai::UniformMesh<Config2D>;
using UniformMesh3D = samurai::UniformMesh<Config3D>;

} // namespace samurai::python
```

### 2.2 Mesh Binding

```cpp
// src/python_bindings/mesh_bindings.cpp

#include "mesh_types.hpp"

void bind_uniform_mesh_1d(py::module_& m)
{
    using Mesh = samurai::python::UniformMesh1D;
    using Box = samurai::Box<double, 1>;

    py::class_<Mesh>(m, "UniformMesh1D")
        .def(py::init<const Box&, std::size_t, double, double>(),
             py::arg("box"),
             py::arg("level"),
             py::arg("approx_box_tol")=1e-9,
             py::arg("scaling_factor")=0.,
             "Create a 1D uniform mesh")

        .def("nb_cells", &Mesh::nb_cells,
             "Total number of cells (including ghosts)")

        .def_property_readonly("origin_point",
            [](const Mesh& m) -> xt::pyarray<double> {
                return xt::pyarray<double>(m.origin_point());
            },
            "Origin point of the mesh")

        .def_property_readonly("scaling_factor", &Mesh::scaling_factor,
                              "Scaling factor of the mesh")

        .def("cell_length", &Mesh::cell_length,
             py::arg("level"),
             "Cell length at a given refinement level")

        .def("__repr__", [](const Mesh& m) {
            return fmt::format("UniformMesh1D(cells={})", m.nb_cells());
        });
}
```

### 2.3 Test Mesh

```python
# python/tests/test_mesh.py

def test_uniform_mesh_1d():
    box = samurai_core.Box1D([0.], [1.])
    mesh = samurai_core.UniformMesh1D(box, level=5)

    # 2^5 = 32 cells
    assert mesh.nb_cells() == 32

    # Cell length at level 5: 1/2^5 = 1/32
    assert abs(mesh.cell_length(5) - 1/32) < 1e-10
```

---

## Phase 3 : Itération (JOURS 13-16)

### 3.1 Cell Binding

```cpp
// src/python_bindings/cell_bindings.cpp

using Cell1D = samurai::Cell<1, samurai::python::default_interval>;
using Cell2D = samurai::Cell<2, samurai::python::default_interval>;

void bind_cell_1d(py::module_& m)
{
    py::class_<Cell1D>(m, "Cell1D")
        .def_property_readonly("level", &Cell1D::level,
                               "Refinement level of the cell")

        .def_property_readonly("index", &Cell1D::index,
                               "Index in the data array")

        .def_property_readonly("length", &Cell1D::length,
                               "Physical length of the cell")

        .def_property_readonly("center",
            [](const Cell1D& c) -> xt::pyarray<double> {
                return xt::pyarray<double>(c.center());
            },
            "Center coordinates of the cell")

        .def_property_readonly("corner",
            [](const Cell1D& c) -> xt::pyarray<double> {
                return xt::pyarray<double>(c.corner());
            },
            "Minimum corner of the cell")

        .def("__repr__", [](const Cell1D& c) {
            return fmt::format("Cell1D(level={}, center={})",
                             c.level, c.center()[0]);
        });
}
```

### 3.2 Iteration Wrapper

```cpp
// src/python_bindings/algorithm_bindings.cpp

void bind_for_each_cell(py::module_& m)
{
    // Version 1D
    m.def("for_each_cell", [](samurai::python::UniformMesh1D& mesh,
                               py::function func) {
        samurai::for_each_cell(mesh, [&](auto& cell) {
            // Créer une copie du cell pour Python
            Cell1D cell_copy(cell);
            func(cell_copy);
        });
    }, py::arg("mesh"), py::arg("func"),
       "Iterate over all cells, calling Python function");
}
```

### 3.3 Test Iteration

```python
def test_for_each_cell():
    box = samurai_core.Box1D([0.], [1.])
    mesh = samurai_core.UniformMesh1D(box, level=3)

    cells = []
    samurai_core.for_each_cell(mesh, lambda c: cells.append(c))

    assert len(cells) == 8  # 2^3
    assert all(c.level == 3 for c in cells)
```

---

## Phase 4 : Champs + NumPy (JOURS 17-28)

### 4.1 Field Binding avec Buffer Protocol

```cpp
// src/python_bindings/field_bindings.cpp

#include <samurai/field.hpp>

namespace samurai::python {

    using ScalarField1D = samurai::ScalarField<UniformMesh1D, double>;
    using ScalarField2D = samurai::ScalarField<UniformMesh2D, double>;

}

void bind_scalar_field_1d(py::module_& m)
{
    using Field = samurai::python::ScalarField1D;
    using Mesh = samurai::python::UniformMesh1D;

    py::class_<Field>(m, "ScalarField1D", py::buffer_protocol())
        .def(py::init<std::string, Mesh&>(),
             py::arg("name"), py::arg("mesh"),
             "Create a scalar field on a mesh")

        .def_property_readonly("name", &Field::name,
                              "Name of the field")

        .def_property_readonly("mesh", &Field::mesh,
                              py::return_value_policy::reference,
                              "Mesh the field is defined on")

        .def_property_readonly("size", &Field::size,
                              "Number of elements in the field")

        // BUFFER PROTOCOL - ZERO COPY
        .def_buffer([](Field& f) -> py::buffer_info {
            using T = typename Field::value_type;
            return py::buffer_info(
                f.array().data(),
                sizeof(T),
                py::format_descriptor<T>::format(),
                1,
                {f.array().size()},
                {sizeof(T)}
            );
        })

        // NumPy view explicite
        .def("numpy_view", [](Field& f) -> py::array_t<double> {
            return py::array_t<double>(
                {f.array().size()},
                {sizeof(double)},
                f.array().data(),
                py::cast(f)  // CRITICAL: garde le field en vie
            );
        }, py::return_value_policy::take_ownership,
           "Returns a zero-copy NumPy view of the field data")

        // Accès par cellule
        .def("__getitem__", [](Field& f, const Cell1D& cell) -> double {
            return f[cell];
        }, py::arg("cell"))

        .def("__setitem__", [](Field& f, const Cell1D& cell, double value) {
            f[cell] = value;
        }, py::arg("cell"), py::arg("value"))

        // Remplissage
        .def("fill", &Field::fill, py::arg("value"),
             "Fill field with constant value")

        .def("__repr__", [](const Field& f) {
            return fmt::format("ScalarField1D(name='{}', size={})",
                             f.name(), f.array().size());
        });
}
```

### 4.2 Test Zero-Copy

```python
def test_numpy_zero_copy():
    import numpy as np

    box = samurai_core.Box1D([0.], [1.])
    mesh = samurai_core.UniformMesh1D(box, level=5)
    u = samurai_core.ScalarField1D("u", mesh)

    # Remplir
    u.fill(1.0)

    # View zero-copy
    u_arr = u.numpy_view()

    # Vérifier qu'on partage la mémoire
    assert np.shares_memory(u_arr, u.numpy_view())

    # Modifier via NumPy
    u_arr[0] = 42.0

    # Vérifier que le field est modifié
    cells = list(samurai_core.for_each_cell(mesh, lambda c: c))
    assert u[cells[0]] == 42.0
```

---

## Phase 5 : Factory Functions (JOURS 29-33)

### 5.1 Simplification de l'API

```cpp
// src/python_bindings/factory_bindings.cpp

m.def("make_scalar_field",
    [](std::string name, samurai::python::UniformMesh1D& mesh,
       double value) -> samurai::python::ScalarField1D {
        return samurai::make_scalar_field<double>(name, mesh, value);
    },
    py::arg("name"), py::arg("mesh"), py::arg("value")=0.,
    "Create a scalar field with constant value");

m.def("make_scalar_field",
    [](std::string name, samurai::python::UniformMesh1D& mesh,
       py::function func) -> samurai::python::ScalarField1D {
        // Wrapper Python function -> C++
        auto cpp_func = [func](const xt::xtensor_fixed<double, xt::xshape<1>>& coords) {
            py::array_t<double> result = func(coords);
            return *result.data();
        };
        return samurai::make_scalar_field<double>(name, mesh, cpp_func);
    },
    py::arg("name"), py::arg("mesh"), py::arg("function"),
    "Create a scalar field from a Python function");
```

### 5.2 Usage Simplifié

```python
# Avant
u = samurai_core.ScalarField1D("u", mesh)
u.fill(1.0)

# Après
u = samurai_core.make_scalar_field("u", mesh, value=1.0)

# Ou avec fonction
u = samurai_core.make_scalar_field("u", mesh,
    lambda x: np.sin(2*np.pi*x[0]))
```

---

## Phase 6 : I/O HDF5 (JOURS 34-40)

### 6.1 Save Function

```cpp
// src/python_bindings/io_bindings.cpp

#include <samurai/hdf5.hpp>

void init_io(py::module_& m)
{
    m.def("save", [](const std::string& path,
                     const std::string& filename,
                     samurai::python::ScalarField1D& field) {
        samurai::save(fs::path(path), filename,
                     samurai::Hdf5Options(),
                     field.mesh(), field);
    }, py::arg("path"), py::arg("filename"), py::arg("field"),
       "Save field to HDF5 file");

    m.def("load", [](const std::string& path,
                     const std::string& filename) {
        auto file = samurai::Hdf5(
            fs::path(path) / (filename + ".h5"),
            samurai::Hdf5::FileName
        );
        // Retourner un dict avec mesh et fields
        py::dict result;
        // TODO: implémenter le chargement
        return result;
    }, py::arg("path"), py::arg("filename"),
       "Load fields from HDF5 file");
}
```

---

## Checkpoint Milestones

### Milestone 1 : Semaine 2
- ✅ Box1D/2D fonctionnels
- ✅ Tests passent
- ✅ Build stable

### Milestone 2 : Semaine 4
- ✅ UniformMesh1D/2D fonctionnels
- ✅ for_each_cell fonctionne
- ✅ Cell wrappers complets

### Milestone 3 : Semaine 6
- ✅ ScalarField avec NumPy zero-copy
- ✅ Performance validée (<5% overhead)

### Milestone 4 : Semaine 8
- ✅ Factory functions
- ✅ I/O HDF5
- ✅ Demo complète fonctionnelle

---

## Commandes de Développement

### Build
```bash
cd build-python
cmake ../src/python_bindings -DCMAKE_BUILD_TYPE=Release
make -j4
```

### Tests
```bash
python -m pytest python/tests/ -v
```

### Demo
```bash
python python/examples/demo_1d_heat.py
```
