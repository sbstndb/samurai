// Samurai Python Bindings - Box class
//
// Bindings for samurai::Box<value_t, dim> class
// Defines a box in multi dimensions by its minimum and maximum corners.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <samurai/box.hpp>

namespace py = pybind11;

// Type aliases for convenience
using Box1D = samurai::Box<double, 1>;
using Box2D = samurai::Box<double, 2>;
using Box3D = samurai::Box<double, 3>;

// Helper function to convert Python list/array to xtensor_fixed
template <std::size_t dim>
auto convert_to_point(const py::object& obj)
{
    using point_t = xt::xtensor_fixed<double, xt::xshape<dim>>;

    // Try to convert from list/tuple
    try
    {
        py::list list = py::cast<py::list>(obj);
        if (list.size() != dim)
        {
            throw std::runtime_error("Expected list of length " + std::to_string(dim));
        }
        point_t point;
        for (std::size_t i = 0; i < dim; ++i)
        {
            point[i] = list[i].cast<double>();
        }
        return point;
    }
    catch (const py::cast_error&)
    {
        // Try numpy array
        try
        {
            py::array_t<double> arr = py::cast<py::array_t<double>>(obj);
            if (arr.size() != dim)
            {
                throw std::runtime_error("Expected array of length " + std::to_string(dim));
            }
            point_t point;
            auto buf  = arr.request();
            auto* ptr = static_cast<double*>(buf.ptr);
            for (std::size_t i = 0; i < dim; ++i)
            {
                point[i] = ptr[i];
            }
            return point;
        }
        catch (const py::cast_error&)
        {
            throw std::runtime_error("Cannot convert to point: expected list or numpy array");
        }
    }
}

// Helper to convert xtensor-like point to numpy array (copy-based for simplicity)
template <typename Point, std::size_t N>
py::array_t<double> point_to_numpy(const Point& point)
{
    py::array_t<double> arr(N);
    auto buf  = arr.request();
    auto* ptr = static_cast<double*>(buf.ptr);
    for (std::size_t i = 0; i < N; ++i)
    {
        ptr[i] = point[i];
    }
    return arr;
}

// Template function to bind Box for any dimension
template <std::size_t dim>
void bind_box(py::module_& m, const std::string& name)
{
    using Box     = samurai::Box<double, dim>;
    using point_t = typename Box::point_t;

    py::class_<Box>(m, name.c_str(), R"pbdoc(
        Box class defining a region in multi-dimensional space.

        A box is defined by its minimum and maximum corners.

        Parameters
        ----------
        min_corner : array_like
            Coordinates of the minimum corner
        max_corner : array_like
            Coordinates of the maximum corner

        Examples
        --------
        >>> import samurai as sam
        >>> box = sam.Box2D([0., 0.], [1., 1.])
        >>> print(box.min_corner)
        [0. 0.]
        >>> print(box.length)
        [1. 1.]
    )pbdoc")

        // Constructor
        .def(py::init(
                 [](const py::object& min_obj, const py::object& max_obj)
                 {
                     auto min_corner = convert_to_point<dim>(min_obj);
                     auto max_corner = convert_to_point<dim>(max_obj);
                     return Box(min_corner, max_corner);
                 }),
             py::arg("min_corner"),
             py::arg("max_corner"),
             "Create a box from min and max corners")

        // Properties
        .def_property_readonly(
            "dim",
            [](const Box&)
            {
                return dim;
            },
            "Dimension of the box")

        .def_property(
            "min_corner",
            [](Box& box) -> py::array_t<double>
            {
                return point_to_numpy<decltype(box.min_corner()), dim>(box.min_corner());
            },
            [](Box& box, const py::object& obj)
            {
                box.min_corner() = convert_to_point<dim>(obj);
            },
            "Minimum corner of the box (read/write)")

        .def_property(
            "max_corner",
            [](Box& box) -> py::array_t<double>
            {
                return point_to_numpy<decltype(box.max_corner()), dim>(box.max_corner());
            },
            [](Box& box, const py::object& obj)
            {
                box.max_corner() = convert_to_point<dim>(obj);
            },
            "Maximum corner of the box (read/write)")

        // Methods
        .def(
            "length",
            [](const Box& box) -> py::array_t<double>
            {
                return point_to_numpy<decltype(box.length()), dim>(box.length());
            },
            "Length of the box in each dimension")

        .def("min_length", &Box::min_length, "Minimum length among all dimensions")

        .def("is_valid", &Box::is_valid, "Check if the box is valid (min_corner < max_corner in all dimensions)")

        .def("intersects", &Box::intersects, py::arg("other"), "Check if this box intersects with another box")

        .def("intersection", &Box::intersection, py::arg("other"), "Return the intersection of this box with another")

        .def("difference", &Box::difference, py::arg("other"), "Return the difference of this box with another (as list of boxes)")

        // Operators
        .def("__eq__", &Box::operator==, "Check if two boxes are equal")

        .def("__ne__", &Box::operator!=, "Check if two boxes are different")

        .def(
            "__imul__",
            [](Box& box, double v) -> Box&
            {
                return box *= v;
            },
            py::arg("v"),
            "Scale the box in-place")

        .def(
            "__mul__",
            [](const Box& box, double v)
            {
                return box * v;
            },
            py::arg("v"),
            "Scale the box (right multiplication)")

        .def(
            "__rmul__",
            [](const Box& box, double v)
            {
                return v * box;
            },
            py::arg("v"),
            "Scale the box (left multiplication)")

        // String representation
        .def("__repr__",
             [name](const Box& box)
             {
                 std::ostringstream oss;
                 oss << name << "(";
                 oss << "[";
                 for (std::size_t i = 0; i < dim; ++i)
                 {
                     if (i > 0)
                     {
                         oss << ", ";
                     }
                     oss << box.min_corner()[i];
                 }
                 oss << "], [";
                 for (std::size_t i = 0; i < dim; ++i)
                 {
                     if (i > 0)
                     {
                         oss << ", ";
                     }
                     oss << box.max_corner()[i];
                 }
                 oss << "])";
                 return oss.str();
             })

        .def("__str__",
             [name](const Box& box)
             {
                 std::ostringstream oss;
                 oss << name << "(";
                 oss << "min=";
                 for (std::size_t i = 0; i < dim; ++i)
                 {
                     if (i > 0)
                     {
                         oss << ", ";
                     }
                     oss << box.min_corner()[i];
                 }
                 oss << ", max=";
                 for (std::size_t i = 0; i < dim; ++i)
                 {
                     if (i > 0)
                     {
                         oss << ", ";
                     }
                     oss << box.max_corner()[i];
                 }
                 oss << ")";
                 return oss.str();
             });
}

// Helper function to detect dimension from Python object
std::size_t detect_dimension_from_input(const py::object& obj)
{
    // Try list/tuple
    try
    {
        py::list list = py::cast<py::list>(obj);
        return list.size();
    }
    catch (const py::cast_error&)
    {
        // Try numpy array
        try
        {
            py::array_t<double> arr = py::cast<py::array_t<double>>(obj);
            return arr.size();
        }
        catch (const py::cast_error&)
        {
            throw std::runtime_error("Cannot determine dimension: expected list or numpy array");
        }
    }
}

// Module initialization function for Box bindings
void init_box_bindings(py::module_& m)
{
    // ============================================================
    // BREAKING CHANGE (v0.30.0): Explicit Box classes removed from public API
    // Users must use the factory: sam.geometry.box(min_corner, max_corner)
    // The factory auto-detects dimension from array length.
    //
    // NOTE: We still register the Box types with pybind11 (as _Box1D, _Box2D, _Box3D)
    // because the factory function needs to return these types. The underscore prefix
    // indicates they are internal implementation details, not public API.
    // ============================================================

    // ============================================================
    // Create geometry submodule for organized API access
    // ============================================================
    py::module_ geometry = m.def_submodule("geometry",
        "Geometric primitives for Samurai AMR simulations\n\n"
        "Factory Functions:\n"
        "  box(min_corner, max_corner) - Create Box with inferred dimension\n\n"
        "Examples:\n"
        "    >>> import samurai_python as sam\n"
        "    >>> # Factory function (auto-detects dimension)\n"
        "    >>> box = sam.geometry.box([0., 0.], [1., 1.])\n"
        "    >>> box_1d = sam.geometry.box([0.0], [1.0])\n"
        "    >>> box_3d = sam.geometry.box([0., 0., 0.], [1., 1., 1.])\n");

    // Register Box types (internal, with _ prefix) for factory function return types
    bind_box<1>(geometry, "_Box1D");
    bind_box<2>(geometry, "_Box2D");
    bind_box<3>(geometry, "_Box3D");

    // ============================================================
    // Factory function: sam.geometry.box(min_corner, max_corner)
    // Infers dimension from array length
    // ============================================================
    geometry.def("box",
        [](const py::object& min_obj, const py::object& max_obj) -> py::object
        {
            // Detect dimension from input
            std::size_t dim = detect_dimension_from_input(min_obj);

            // Validate both inputs have same dimension
            std::size_t dim_max = detect_dimension_from_input(max_obj);
            if (dim != dim_max)
            {
                throw std::runtime_error("min_corner and max_corner must have the same dimension");
            }

            // Create appropriate Box based on dimension
            if (dim == 1)
            {
                auto min_corner = convert_to_point<1>(min_obj);
                auto max_corner = convert_to_point<1>(max_obj);
                Box1D box(min_corner, max_corner);
                return py::cast(box);
            }
            else if (dim == 2)
            {
                auto min_corner = convert_to_point<2>(min_obj);
                auto max_corner = convert_to_point<2>(max_obj);
                Box2D box(min_corner, max_corner);
                return py::cast(box);
            }
            else if (dim == 3)
            {
                auto min_corner = convert_to_point<3>(min_obj);
                auto max_corner = convert_to_point<3>(max_obj);
                Box3D box(min_corner, max_corner);
                return py::cast(box);
            }
            else
            {
                throw std::runtime_error("Unsupported dimension: " + std::to_string(dim) + " (must be 1, 2, or 3)");
            }
        },
        py::arg("min_corner"),
        py::arg("max_corner"),
        R"pbdoc(
        Create a Box by inferring dimension from array length.

        Parameters
        ----------
        min_corner : array_like
            Minimum corner coordinates (e.g., [0.0] for 1D, [0.0, 0.0] for 2D)
        max_corner : array_like
            Maximum corner coordinates

        Returns
        -------
        Box
            Box object (dimension inferred from input array length)

        Examples
        --------
        >>> import samurai_python as sam
        >>> box_1d = sam.geometry.box([0.0], [1.0])
        >>> box_2d = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        >>> box_3d = sam.geometry.box([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    )pbdoc");

    // Also include Interval if it's bound in the main module
    // (interval_bindings.cpp may be initialized before or after this file)
    try
    {
        py::object interval = m.attr("Interval");
        geometry.attr("Interval") = interval;
    }
    catch (const py::error_already_set&)
    {
        // Interval not yet bound, will be added later by interval_bindings.cpp
    }

    // Note: DomainBuilder classes are added by domain_builder_bindings.cpp
    // They will be available as geometry.DomainBuilder1D, etc.
}
