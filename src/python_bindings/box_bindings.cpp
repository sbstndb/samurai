// Box bindings for Python - Generic for all T and dimensions
// Demonstrates explicit template instantiation with factory pattern

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <samurai/box.hpp>
#include <sstream>

namespace py = pybind11;

// ============================================================================
// Helper: Convert Python array to xt::xtensor_fixed
// ============================================================================
template <typename value_t, std::size_t dim>
auto array_to_point(py::array_t<double> arr) {
    using point_t = xt::xtensor_fixed<value_t, xt::xshape<dim>>;

    auto buf = arr.request();
    if (buf.size != dim) {
        throw std::runtime_error("Point must have exactly " + std::to_string(dim) + " elements");
    }
    auto* ptr = static_cast<double*>(buf.ptr);

    // Construct xtensor_fixed from array
    point_t point;
    for (std::size_t i = 0; i < dim; ++i) {
        point[i] = static_cast<value_t>(ptr[i]);
    }
    return point;
}

// ============================================================================
// Template binding generator for Box<T, dim>
// ============================================================================
template <typename value_t, std::size_t dim>
void bind_box_typed(py::module_& m, const std::string& type_name) {
    using Box = samurai::Box<value_t, dim>;

    // Create class name: Box2D_double, Box3D_float, etc.
    std::string dim_str = (dim == 1) ? "1D" : (dim == 2) ? "2D" : "3D";
    std::string class_name = "Box" + dim_str + "_" + type_name;

    // Store dimension as runtime value for lambda captures
    constexpr std::size_t dim_value = dim;

    py::class_<Box>(m, class_name.c_str())
        // Constructor from two corners
        .def(py::init([](py::array_t<double> min_arr, py::array_t<double> max_arr) {
            return Box(array_to_point<value_t, dim>(min_arr),
                      array_to_point<value_t, dim>(max_arr));
        }),
             py::arg("min_corner"),
             py::arg("max_corner"),
             ("Create a " + dim_str + " box from min and max corners").c_str())

        // Default constructor
        .def(py::init<>(),
             "Create a default box with corners at origin")

        // Accessors - return NumPy arrays
        .def_property_readonly("min",
             [](const Box& box) -> py::array_t<double> {
                 const auto& min_corner = box.min_corner();
                 py::array_t<double> result(dim_value);
                 auto buf = result.mutable_unchecked<1>();
                 for (std::size_t i = 0; i < dim_value; ++i) {
                     buf(i) = static_cast<double>(min_corner[i]);
                 }
                 return result;
             },
             "Get minimum corner as NumPy array")

        .def_property_readonly("max",
             [](const Box& box) -> py::array_t<double> {
                 const auto& max_corner = box.max_corner();
                 py::array_t<double> result(dim_value);
                 auto buf = result.mutable_unchecked<1>();
                 for (std::size_t i = 0; i < dim_value; ++i) {
                     buf(i) = static_cast<double>(max_corner[i]);
                 }
                 return result;
             },
             "Get maximum corner as NumPy array")

        .def("min_corner", py::overload_cast<>(&Box::min_corner, py::const_),
             py::return_value_policy::reference,
             "Get reference to minimum corner point")

        .def("max_corner", py::overload_cast<>(&Box::max_corner, py::const_),
             py::return_value_policy::reference,
             "Get reference to maximum corner point")

        // Utility methods - wrap xtensor expressions
        .def("length", [](const Box& box) -> py::array_t<double> {
            auto len = box.length();
            py::array_t<double> result(dim_value);
            auto buf = result.mutable_unchecked<1>();
            for (std::size_t i = 0; i < dim_value; ++i) {
                buf(i) = static_cast<double>(len[i]);
            }
            return result;
        },
             "Get box dimensions as NumPy array")

        .def("min_length", [](const Box& box) -> double {
            return static_cast<double>(box.min_length());
        },
             "Get minimum dimension of the box")

        .def("is_valid", &Box::is_valid,
             "Check if box is valid (all max > min)")

        // Geometric operations
        .def("intersects", &Box::intersects,
             py::arg("other"),
             "Check if this box intersects with another box")

        .def("intersection", &Box::intersection,
             py::arg("other"),
             "Return intersection box with another box")

        .def("difference", &Box::difference,
             py::arg("other"),
             "Return list of boxes representing this \\ other")

        // Setters
        .def("set_min", [](Box& box, py::array_t<double> min_arr) {
            box.min_corner() = array_to_point<value_t, dim>(min_arr);
        }, py::arg("min_corner"),
           "Set minimum corner from NumPy array")

        .def("set_max", [](Box& box, py::array_t<double> max_arr) {
            box.max_corner() = array_to_point<value_t, dim>(max_arr);
        }, py::arg("max_corner"),
           "Set maximum corner from NumPy array")

        // Operators
        .def(py::self == py::self,
             "Check if two boxes are equal")

        .def(py::self != py::self,
             "Check if two boxes are not equal")

        .def("__imul__", &Box::operator*=,
             py::arg("factor"),
             "Scale box in-place by factor")

        // Fix: Use value_t for scaling to avoid type mismatch
        .def("__mul__", [](const Box& box, value_t factor) {
            return box * factor;
        }, py::arg("factor"),
           "Scale box by factor (returns new box)")

        .def("__rmul__", [](const Box& box, value_t factor) {
            return factor * box;
        }, py::arg("factor"),
           "Scale box by factor (reverse, for factor * box)")

        // String representation
        .def("__repr__", [class_name, dim_value](const Box& box) {
            std::ostringstream oss;
            const auto& min = box.min_corner();
            const auto& max = box.max_corner();
            oss << class_name << "(min=[";
            for (std::size_t i = 0; i < dim_value; ++i) {
                if (i > 0) oss << ", ";
                oss << min[i];
            }
            oss << "], max=[";
            for (std::size_t i = 0; i < dim_value; ++i) {
                if (i > 0) oss << ", ";
                oss << max[i];
            }
            oss << "])";
            return oss.str();
        })

        .def("__str__", [dim_value](const Box& box) {
            std::ostringstream oss;
            const auto& min = box.min_corner();
            const auto& max = box.max_corner();
            oss << "Box" << dim_value << "D([";
            for (std::size_t i = 0; i < dim_value; ++i) {
                if (i > 0) oss << ", ";
                oss << min[i];
            }
            oss << "] -> [";
            for (std::size_t i = 0; i < dim_value; ++i) {
                if (i > 0) oss << ", ";
                oss << max[i];
            }
            oss << "])";
            return oss.str();
        });
}

// ============================================================================
// Factory function implementations
// ============================================================================
template <typename value_t>
py::object box_factory_impl(int dim, py::array_t<double> min_arr, py::array_t<double> max_arr) {
    switch (dim) {
        case 1: {
            using Box = samurai::Box<value_t, 1>;
            return py::cast(Box(array_to_point<value_t, 1>(min_arr),
                              array_to_point<value_t, 1>(max_arr)));
        }
        case 2: {
            using Box = samurai::Box<value_t, 2>;
            return py::cast(Box(array_to_point<value_t, 2>(min_arr),
                              array_to_point<value_t, 2>(max_arr)));
        }
        case 3: {
            using Box = samurai::Box<value_t, 3>;
            return py::cast(Box(array_to_point<value_t, 3>(min_arr),
                              array_to_point<value_t, 3>(max_arr)));
        }
        default:
            throw std::runtime_error("Unsupported dimension: " + std::to_string(dim) + " (must be 1, 2, or 3)");
    }
}

// Factory with auto-detected dimension and default dtype
py::object box_factory_auto(py::array_t<double> min_arr,
                           py::array_t<double> max_arr,
                           const std::string& dtype) {
    // Auto-detect dimension from array size
    auto buf_min = min_arr.request();
    auto buf_max = max_arr.request();

    if (buf_min.size != buf_max.size) {
        throw std::runtime_error("min_corner and max_corner must have the same size");
    }

    std::size_t dim = buf_min.size;
    if (dim < 1 || dim > 3) {
        throw std::runtime_error("Auto-detected dimension must be 1, 2, or 3, got: " + std::to_string(dim));
    }

    // Dispatch to appropriate type
    if (dtype == "double" || dtype == "float64") {
        return box_factory_impl<double>(static_cast<int>(dim), min_arr, max_arr);
    } else if (dtype == "float" || dtype == "float32") {
        return box_factory_impl<float>(static_cast<int>(dim), min_arr, max_arr);
    } else {
        throw std::runtime_error("Unsupported dtype: " + dtype + " (use 'double' or 'float')");
    }
}

// ============================================================================
// Module definition
// ============================================================================
PYBIND11_MODULE(samurai_core, m) {
    m.doc() = "Samurai V2 Python Bindings - Generic Box Implementation";

    // Version info
    m.attr("__version__") = "0.3.0-dev";

    // ============================================================================
    // EXPLICIT TEMPLATE INSTANTIATIONS
    // ============================================================================
    // Double precision boxes
    bind_box_typed<double, 1>(m, "double");
    bind_box_typed<double, 2>(m, "double");
    bind_box_typed<double, 3>(m, "double");

    // Single precision boxes
    bind_box_typed<float, 1>(m, "float");
    bind_box_typed<float, 2>(m, "float");
    bind_box_typed<float, 3>(m, "float");

    // ============================================================================
    // FACTORY FUNCTIONS
    // ============================================================================

    // 1. Simple factory with auto-detection (DEFAULT)
    m.def("Box", &box_factory_auto,
          R"(
          Create a Box with auto-detected dimension.

          Parameters
          ----------
          min_corner : array-like
              Minimum corner coordinates (dimension auto-detected from size)
          max_corner : array-like
              Maximum corner coordinates (must have same size as min_corner)
          dtype : str, optional
              Data type ('double', 'float64', 'float', or 'float32'). Default is 'double'.

          Returns
          -------
          Box
              A Box object of the appropriate type (Box1D_double, Box2D_double, or Box3D_double)

          Examples
          --------
          >>> import samurai_core
          >>> import numpy as np
          >>> # Auto-detect 2D
          >>> box = samurai_core.Box(np.array([0., 0.]), np.array([1., 1.]))
          >>> # Auto-detect 3D
          >>> box3d = samurai_core.Box(np.array([0., 0., 0.]), np.array([1., 1., 1.]))
          >>> # With explicit dtype
          >>> box_f = samurai_core.Box(np.array([0., 0.]), np.array([1., 1.]), dtype='float')
          )",
          py::arg("min_corner"),
          py::arg("max_corner"),
          py::arg("dtype") = "double");

    // ============================================================================
    // SIMPLE ALIASES - Short names for common cases
    // ============================================================================
    m.attr("Box1D") = m.attr("Box1D_double");
    m.attr("Box2D") = m.attr("Box2D_double");
    m.attr("Box3D") = m.attr("Box3D_double");
}
