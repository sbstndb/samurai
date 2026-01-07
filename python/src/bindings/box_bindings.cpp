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

// Module initialization function for Box bindings
void init_box_bindings(py::module_& m)
{
    // Bind Box classes for dimensions 1, 2, 3
    bind_box<1>(m, "Box1D");
    bind_box<2>(m, "Box2D");
    bind_box<3>(m, "Box3D");

    // ============================================================
    // Create geometry submodule for organized API access
    // ============================================================
    py::module_ geometry = m.def_submodule("geometry",
        "Geometric primitives for Samurai AMR simulations\n\n"
        "This submodule provides organized access to geometric classes.\n"
        "Both sam.geometry.Box2D and sam.Box2D reference the same class.\n\n"
        "Examples:\n"
        "    >>> import samurai_python as sam\n"
        "    >>> # New organized API (recommended)\n"
        "    >>> box = sam.geometry.Box2D([0., 0.], [1., 1.])\n"
        "    >>> # Old API (still works)\n"
        "    >>> box = sam.Box2D([0., 0.], [1., 1.])\n");

    // Reference existing Box classes in the submodule
    geometry.attr("Box1D") = m.attr("Box1D");
    geometry.attr("Box2D") = m.attr("Box2D");
    geometry.attr("Box3D") = m.attr("Box3D");

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
