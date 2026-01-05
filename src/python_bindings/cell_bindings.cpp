// Cell bindings for Python - Wrapper for Samurai Cell class

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <samurai/cell.hpp>
#include <samurai/interval.hpp>

namespace py = pybind11;

// ============================================================================
// Cell bindings for 2D case (default interval type)
// ============================================================================
template <std::size_t dim, typename interval_t>
void bind_cell(py::module_& m, const std::string& suffix) {
    using Cell = samurai::Cell<dim, interval_t>;
    using coord_t = typename Cell::coords_t;

    // Store dimension as runtime value for lambda captures
    constexpr std::size_t dim_value = dim;

    std::string class_name = "Cell" + std::to_string(dim) + "D" + suffix;

    py::class_<Cell>(m, class_name.c_str())
        .def_property_readonly("level",
             [](const Cell& cell) -> std::size_t {
                 return cell.level;
             },
             "Refinement level of the cell")

        .def_property_readonly("indices",
             [](const Cell& cell) -> py::array_t<std::size_t> {
                 py::array_t<std::size_t> result(dim_value);
                 auto buf = result.mutable_unchecked<1>();
                 for (std::size_t i = 0; i < dim_value; ++i) {
                     buf(i) = cell.indices[i];
                 }
                 return result;
             },
             "Integer coordinates of the cell")

        .def_property_readonly("center",
             [](const Cell& cell) -> py::array_t<double> {
                 auto center = cell.center();
                 py::array_t<double> result(dim_value);
                 auto buf = result.mutable_unchecked<1>();
                 for (std::size_t i = 0; i < dim_value; ++i) {
                     buf(i) = static_cast<double>(center[i]);
                 }
                 return result;
             },
             "Center coordinates of the cell")

        .def_property_readonly("corner",
             [](const Cell& cell) -> py::array_t<double> {
                 auto corner = cell.corner();
                 py::array_t<double> result(dim_value);
                 auto buf = result.mutable_unchecked<1>();
                 for (std::size_t i = 0; i < dim_value; ++i) {
                     buf(i) = static_cast<double>(corner[i]);
                 }
                 return result;
             },
             "Minimum corner coordinates of the cell")

        .def_property_readonly("index",
             [](const Cell& cell) -> std::size_t {
                 return cell.index;
             },
             "Linear index of the cell in the mesh")

        .def_property_readonly("length",
             [](const Cell& cell) -> py::array_t<double> {
                 auto length = cell.length();
                 py::array_t<double> result(dim_value);
                 auto buf = result.mutable_unchecked<1>();
                 for (std::size_t i = 0; i < dim_value; ++i) {
                     buf(i) = static_cast<double>(length[i]);
                 }
                 return result;
             },
             "Length of the cell in each dimension")

        .def("__repr__",
             [dim_value](const Cell& cell) {
                 std::ostringstream oss;
                 oss << "Cell" << dim_value << "D(level=" << cell.level
                     << ", indices=[";
                 for (std::size_t i = 0; i < dim_value; ++i) {
                     if (i > 0) oss << ", ";
                     oss << cell.indices[i];
                 }
                 oss << "])";
                 return oss.str();
             })

        .def("__str__",
             [dim_value](const Cell& cell) {
                 std::ostringstream oss;
                 oss << "Cell" << dim_value << "D(level=" << cell.level
                     << ", indices=[";
                 for (std::size_t i = 0; i < dim_value; ++i) {
                     if (i > 0) oss << ", ";
                     oss << cell.indices[i];
                 }
                 oss << "], center=[";
                 auto center = cell.center();
                 for (std::size_t i = 0; i < dim_value; ++i) {
                     if (i > 0) oss << ", ";
                     oss << center[i];
                 }
                 oss << "])";
                 return oss.str();
             });
}
