// Samurai Python Bindings - Algorithm functions
//
// Bindings for iteration primitives like for_each_interval and for_each_cell

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <samurai/algorithm.hpp>
#include <samurai/cell.hpp>
#include <samurai/interval.hpp>
#include <samurai/mesh_config.hpp>
#include <samurai/mr/mesh.hpp>
#include "common_types.hpp"

namespace py = pybind11;

// Use centralized type aliases from common_types.hpp
using namespace samurai::python::bindings;

// Note: algorithm_bindings.cpp uses Interval<int, long long int> for algorithms
// which differs from the algorithm_interval in common_types.hpp (Interval<double, std::size_t>)
// This is intentional for algorithm-specific use cases
using algorithm_interval = samurai::Interval<int, long long int>;

// Helper function to convert xtensor_fixed index to Python tuple
template <std::size_t dim, class IndexArray>
py::tuple convert_index_to_tuple(const IndexArray& index)
{
    if constexpr (dim == 1)
    {
        return py::tuple();
    }
    else if constexpr (dim == 2)
    {
        return py::make_tuple(index[0]);
    }
    else if constexpr (dim == 3)
    {
        return py::make_tuple(index[0], index[1]);
    }
    return py::tuple();
}

// Wrapper functions for each dimension
void for_each_interval_1d(const Mesh1D& mesh, py::function func)
{
    samurai::for_each_interval(mesh,
                               [&func](std::size_t level, const algorithm_interval& interval, const auto& index)
                               {
                                   auto index_tuple = convert_index_to_tuple<1>(index);
                                   func(level, interval, index_tuple);
                               });
}

void for_each_interval_2d(const Mesh2D& mesh, py::function func)
{
    samurai::for_each_interval(mesh,
                               [&func](std::size_t level, const algorithm_interval& interval, const auto& index)
                               {
                                   auto index_tuple = convert_index_to_tuple<2>(index);
                                   func(level, interval, index_tuple);
                               });
}

void for_each_interval_3d(const Mesh3D& mesh, py::function func)
{
    samurai::for_each_interval(mesh,
                               [&func](std::size_t level, const algorithm_interval& interval, const auto& index)
                               {
                                   auto index_tuple = convert_index_to_tuple<3>(index);
                                   func(level, interval, index_tuple);
                               });
}

// ============================================================
// for_each_cell bindings
// ============================================================

// Cell type definition - use same interval type as for_each_interval
// The mesh uses Interval<int, long long int> internally
using cell_interval = samurai::Interval<int, long long int>;

template <std::size_t dim>
using Cell = samurai::Cell<dim, cell_interval>;

// CellWrapper: Lightweight wrapper for exposing Cell to Python
// Stores a copy of Cell data to avoid lifetime issues
template <std::size_t dim>
struct CellWrapper
{
    std::size_t level;
    std::size_t index; // Linear index for field indexing
    double length;
    xt::xtensor_fixed<double, xt::xshape<dim>> center;
    xt::xtensor_fixed<double, xt::xshape<dim>> corner;

    // Constructor from C++ Cell
    explicit CellWrapper(const Cell<dim>& cell)
        : level(cell.level)
        , index(static_cast<std::size_t>(cell.index))
        , length(cell.length)
        , center(cell.center())
        , corner(cell.corner())
    {
    }
};

// Helper to bind CellWrapper class for a specific dimension
template <std::size_t dim>
void bind_cell_wrapper(py::module_& m, const std::string& name)
{
    using Wrapper = CellWrapper<dim>;

    py::class_<Wrapper>(m, name.c_str(), R"pbdoc(Cell wrapper for for_each_cell iteration.)pbdoc")
        .def_property_readonly(
            "level",
            [](const Wrapper& w)
            {
                return w.level;
            },
            "Refinement level of the cell")
        .def_property_readonly(
            "index",
            [](const Wrapper& w)
            {
                return w.index;
            },
            "Linear index in field data array (for field[index] access)")
        .def_property_readonly(
            "length",
            [](const Wrapper& w)
            {
                return w.length;
            },
            "Physical size of the cell")
        .def(
            "center",
            [](const Wrapper& w) -> py::tuple
            {
                if constexpr (dim == 1)
                {
                    return py::make_tuple(w.center[0]);
                }
                else if constexpr (dim == 2)
                {
                    return py::make_tuple(w.center[0], w.center[1]);
                }
                else if constexpr (dim == 3)
                {
                    return py::make_tuple(w.center[0], w.center[1], w.center[2]);
                }
                return py::tuple();
            },
            "Returns cell center as (x, y, z) tuple")
        .def(
            "corner",
            [](const Wrapper& w) -> py::tuple
            {
                if constexpr (dim == 1)
                {
                    return py::make_tuple(w.corner[0]);
                }
                else if constexpr (dim == 2)
                {
                    return py::make_tuple(w.corner[0], w.corner[1]);
                }
                else if constexpr (dim == 3)
                {
                    return py::make_tuple(w.corner[0], w.corner[1], w.corner[2]);
                }
                return py::tuple();
            },
            "Returns cell corner (min point) as (x, y, z) tuple")
        .def("__repr__",
             [name](const Wrapper& w)
             {
                 std::ostringstream oss;
                 oss << name << "(level=" << w.level << ", index=" << w.index << ")";
                 return oss.str();
             });
}

// Wrapper functions for for_each_cell for each dimension
void for_each_cell_1d(const Mesh1D& mesh, py::function func)
{
    samurai::for_each_cell(mesh,
                           [&func](const auto& cell)
                           {
                               CellWrapper<1> wrapper(cell);
                               func(wrapper);
                           });
}

void for_each_cell_2d(const Mesh2D& mesh, py::function func)
{
    samurai::for_each_cell(mesh,
                           [&func](const auto& cell)
                           {
                               CellWrapper<2> wrapper(cell);
                               func(wrapper);
                           });
}

void for_each_cell_3d(const Mesh3D& mesh, py::function func)
{
    samurai::for_each_cell(mesh,
                           [&func](const auto& cell)
                           {
                               CellWrapper<3> wrapper(cell);
                               func(wrapper);
                           });
}

// Module initialization function for algorithm bindings
void init_algorithm_bindings(py::module_& m)
{
    // ============================================================
    // BREAKING CHANGE: No longer bind Cell classes or algorithms to main module
    // Users must use sam.algorithms.Cell1D, sam.algorithms.for_each_cell(), etc.
    // ============================================================

    // ============================================================
    // Create algorithms submodule for organized API access
    // ============================================================
    py::module_ algorithms = m.def_submodule("algorithms",
                                             "Algorithmic primitives for mesh traversal and field operations\n\n"
                                             "Factory Functions:\n"
                                             "  for_each_cell(mesh, function) - Iterate over all cells in mesh\n"
                                             "  for_each_interval(mesh, function) - Iterate over all intervals in mesh\n\n"
                                             "Classes:\n"
                                             "  Cell1D, Cell2D, Cell3D - Cell wrapper objects for iteration\n\n"
                                             "Examples:\n"
                                             "    >>> import samurai_python as sam\n"
                                             "    >>> sam.algorithms.for_each_cell(mesh, lambda cell: print(cell.center()))\n"
                                             "    >>> sam.algorithms.for_each_interval(mesh, lambda interval, index: ...)\n");

    // Bind CellWrapper classes ONLY to algorithms submodule (not to main module)
    bind_cell_wrapper<1>(algorithms, "Cell1D");
    bind_cell_wrapper<2>(algorithms, "Cell2D");
    bind_cell_wrapper<3>(algorithms, "Cell3D");

    // Bind for_each_interval functions ONLY to algorithms submodule (not to main module)
    algorithms.def("for_each_interval",
                   &for_each_interval_1d,
                   py::arg("mesh"),
                   py::arg("function"),
                   "Iterate over all intervals in the 1D mesh.");
    algorithms.def("for_each_interval",
                   &for_each_interval_2d,
                   py::arg("mesh"),
                   py::arg("function"),
                   "Iterate over all intervals in the 2D mesh.");
    algorithms.def("for_each_interval",
                   &for_each_interval_3d,
                   py::arg("mesh"),
                   py::arg("function"),
                   "Iterate over all intervals in the 3D mesh.");

    // Bind for_each_cell functions ONLY to algorithms submodule (not to main module)
    algorithms.def("for_each_cell", &for_each_cell_1d, py::arg("mesh"), py::arg("function"), "Iterate over all cells in the 1D mesh.");
    algorithms.def("for_each_cell", &for_each_cell_2d, py::arg("mesh"), py::arg("function"), "Iterate over all cells in the 2D mesh.");
    algorithms.def("for_each_cell", &for_each_cell_3d, py::arg("mesh"), py::arg("function"), "Iterate over all cells in the 3D mesh.");
}
