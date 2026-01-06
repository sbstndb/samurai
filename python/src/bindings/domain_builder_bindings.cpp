// Samurai Python Bindings - DomainBuilder class
//
// Bindings for samurai::DomainBuilder<dim> class
// Used for creating complex computational domains with holes/obstacles

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <samurai/domain_builder.hpp>

namespace py = pybind11;

// Type aliases for convenience
using Box1D = samurai::Box<double, 1>;
using Box2D = samurai::Box<double, 2>;
using Box3D = samurai::Box<double, 3>;

using DomainBuilder1D = samurai::DomainBuilder<1>;
using DomainBuilder2D = samurai::DomainBuilder<2>;
using DomainBuilder3D = samurai::DomainBuilder<3>;

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

// Template function to bind DomainBuilder for any dimension
template <std::size_t dim>
void bind_domain_builder(py::module_& m, const std::string& name)
{
    using DomainBuilder = samurai::DomainBuilder<dim>;
    using Box           = samurai::Box<double, dim>;

    py::class_<DomainBuilder>(m, name.c_str(), R"pbdoc(
        DomainBuilder for creating computational domains with obstacles.

        DomainBuilder allows you to construct complex domains by adding regions
        and removing regions (creating holes/obstacles). The resulting domain
        can be used to create an adaptive mesh.

        Parameters
        ----------
        min_corner : array_like, optional
            Coordinates of the minimum corner of the initial domain
        max_corner : array_like, optional
            Coordinates of the maximum corner of the initial domain

        Examples
        --------
        Create a domain with a rectangular obstacle:

        >>> import samurai as sam
        >>> domain = sam.DomainBuilder2D([-1., -1.], [1., 1.])
        >>> domain.remove([0.0, 0.0], [0.4, 0.4])  # Create obstacle
        >>> config = sam.MeshConfig2D()
        >>> config.min_level = 1
        >>> config.max_level = 3
        >>> mesh = sam.MRMesh2D(domain, config)

        Notes
        -----
        - Periodic boundary conditions are NOT supported with DomainBuilder
        - MPI parallelization is NOT supported with DomainBuilder
    )pbdoc")

        // Default constructor
        .def(py::init(
                 []()
                 {
                     return DomainBuilder();
                 }),
             "Create an empty domain builder")

        // Constructor from corner points
        .def(py::init(
                 [](const py::object& min_obj, const py::object& max_obj)
                 {
                     auto min_corner = convert_to_point<dim>(min_obj);
                     auto max_corner = convert_to_point<dim>(max_obj);
                     return DomainBuilder(min_corner, max_corner);
                 }),
             py::arg("min_corner"),
             py::arg("max_corner"),
             "Create a domain builder from min and max corners")

        // Constructor from Box
        .def(py::init(
                 [](const Box& box)
                 {
                     return DomainBuilder(box);
                 }),
             py::arg("box"),
             "Create a domain builder from a Box")

        // add() method - add a region to the domain
        .def("add",
             [](DomainBuilder& db, const py::object& min_obj, const py::object& max_obj)
             {
                 auto min_corner = convert_to_point<dim>(min_obj);
                 auto max_corner = convert_to_point<dim>(max_obj);
                 db.add(min_corner, max_corner);
             },
             py::arg("min_corner"),
             py::arg("max_corner"),
             "Add a rectangular region to the domain")

        .def("add",
             [](DomainBuilder& db, const Box& box)
             {
                 db.add(box);
             },
             py::arg("box"),
             "Add a box region to the domain")

        // remove() method - remove a region (create hole/obstacle)
        .def("remove",
             [](DomainBuilder& db, const py::object& min_obj, const py::object& max_obj)
             {
                 auto min_corner = convert_to_point<dim>(min_obj);
                 auto max_corner = convert_to_point<dim>(max_obj);
                 db.remove(min_corner, max_corner);
             },
             py::arg("min_corner"),
             py::arg("max_corner"),
             "Remove a rectangular region (create a hole/obstacle)")

        .def("remove",
             [](DomainBuilder& db, const Box& box)
             {
                 db.remove(box);
             },
             py::arg("box"),
             "Remove a box region (create a hole/obstacle)")

        // Accessors
        .def_property_readonly(
            "added_boxes",
            [](const DomainBuilder& db) -> std::vector<Box>
            {
                return db.added_boxes();
            },
            "List of boxes added to the domain")

        .def_property_readonly(
            "removed_boxes",
            [](const DomainBuilder& db) -> std::vector<Box>
            {
                return db.removed_boxes();
            },
            "List of boxes removed from the domain (holes/obstacles)")

        // Geometric computations
        .def("origin_point",
             [](const DomainBuilder& db) -> py::array_t<double>
             {
                 auto origin = db.origin_point();
                 py::array_t<double> arr(dim);
                 auto buf  = arr.request();
                 auto* ptr = static_cast<double*>(buf.ptr);
                 for (std::size_t i = 0; i < dim; ++i)
                 {
                     ptr[i] = origin[i];
                 }
                 return arr;
             },
             "Get the origin point (minimum corner) of the domain")

        .def("largest_subdivision",
             &DomainBuilder::largest_subdivision,
             "Compute the largest subdivision length for mesh discretization")

        // String representation
        .def("__repr__",
             [name](const DomainBuilder& db)
             {
                 std::ostringstream oss;
                 oss << name << "(";
                 oss << "added=" << db.added_boxes().size();
                 oss << ", removed=" << db.removed_boxes().size();
                 oss << ")";
                 return oss.str();
             })

        .def("__str__",
             [name](const DomainBuilder& db)
             {
                 std::ostringstream oss;
                 oss << name << " with ";
                 oss << db.added_boxes().size() << " added region(s)";
                 if (!db.removed_boxes().empty())
                 {
                     oss << " and " << db.removed_boxes().size() << " obstacle(s)";
                 }
                 return oss.str();
             });
}

// Module initialization function for DomainBuilder bindings
void init_domain_builder_bindings(py::module_& m)
{
    // Bind DomainBuilder classes for dimensions 1, 2, 3
    bind_domain_builder<1>(m, "DomainBuilder1D");
    bind_domain_builder<2>(m, "DomainBuilder2D");
    bind_domain_builder<3>(m, "DomainBuilder3D");

    // Also expose them in a submodule for better organization
    py::module_ geometry = m.def_submodule("geometry", "Geometric primitives");
    geometry.attr("DomainBuilder1D") = m.attr("DomainBuilder1D");
    geometry.attr("DomainBuilder2D") = m.attr("DomainBuilder2D");
    geometry.attr("DomainBuilder3D") = m.attr("DomainBuilder3D");
}
