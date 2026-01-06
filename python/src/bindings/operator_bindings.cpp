// Samurai Python Bindings - Operator functions
//
// Bindings for finite volume operators like upwind

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <samurai/algorithm.hpp>
#include <samurai/field.hpp>
#include <samurai/mesh_config.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/stencil_field.hpp>

namespace py = pybind11;

// Type aliases matching mesh_bindings.cpp
using default_interval = samurai::Interval<int, long long int>;

using Config1D         = samurai::mesh_config<1>;
using CompleteConfig1D = samurai::complete_mesh_config<Config1D, samurai::MRMeshId>;
using Mesh1D           = samurai::MRMesh<CompleteConfig1D>;

using Config2D         = samurai::mesh_config<2>;
using CompleteConfig2D = samurai::complete_mesh_config<Config2D, samurai::MRMeshId>;
using Mesh2D           = samurai::MRMesh<CompleteConfig2D>;

using Config3D         = samurai::mesh_config<3>;
using CompleteConfig3D = samurai::complete_mesh_config<Config3D, samurai::MRMeshId>;
using Mesh3D           = samurai::MRMesh<CompleteConfig3D>;

// Field type aliases
template <std::size_t dim>
using ScalarField = samurai::ScalarField<samurai::MRMesh<samurai::complete_mesh_config<samurai::mesh_config<dim>, samurai::MRMeshId>>, double>;

// 1D upwind operator - immediate evaluation version
py::object upwind_1d(double velocity, ScalarField<1>& field)
{
    auto& mesh = field.mesh();

    // Create output field with same mesh
    auto result = samurai::make_scalar_field<double>(field.name() + "_upwind", mesh);

    // Get the upwind expression
    auto upwind_expr = samurai::upwind(velocity, field);

    // Evaluate the expression immediately using for_each_interval
    samurai::for_each_interval(mesh,
                               [&result, &upwind_expr](std::size_t level, const default_interval& interval, const auto& index)
                               {
                                   result(level, interval, index) = upwind_expr(level, interval, index);
                               });

    return py::cast(result);
}

// 2D upwind operator - immediate evaluation version
py::object upwind_2d(const std::array<double, 2>& velocity, ScalarField<2>& field)
{
    auto& mesh = field.mesh();

    // Create output field with same mesh
    auto result = samurai::make_scalar_field<double>(field.name() + "_upwind", mesh);

    // Get the upwind expression
    auto upwind_expr = samurai::upwind(velocity, field);

    // Evaluate the expression immediately using for_each_interval
    samurai::for_each_interval(mesh,
                               [&result, &upwind_expr](std::size_t level, const default_interval& interval, const auto& index)
                               {
                                   result(level, interval, index) = upwind_expr(level, interval, index);
                               });

    return py::cast(result);
}

// 3D upwind operator - immediate evaluation version
py::object upwind_3d(const std::array<double, 3>& velocity, ScalarField<3>& field)
{
    auto& mesh = field.mesh();

    // Create output field with same mesh
    auto result = samurai::make_scalar_field<double>(field.name() + "_upwind", mesh);

    // Get the upwind expression
    auto upwind_expr = samurai::upwind(velocity, field);

    // Evaluate the expression immediately using for_each_interval
    samurai::for_each_interval(mesh,
                               [&result, &upwind_expr](std::size_t level, const default_interval& interval, const auto& index)
                               {
                                   result(level, interval, index) = upwind_expr(level, interval, index);
                               });

    return py::cast(result);
}

// Convenience wrapper accepting Python list/tuple for velocity (2D)
py::object upwind_2d_py(py::sequence velocity_seq, ScalarField<2>& field)
{
    if (len(velocity_seq) != 2)
    {
        throw std::runtime_error("Velocity must have exactly 2 elements for 2D");
    }

    std::array<double, 2> velocity;
    velocity[0] = velocity_seq[0].cast<double>();
    velocity[1] = velocity_seq[1].cast<double>();

    return upwind_2d(velocity, field);
}

// Convenience wrapper accepting Python list/tuple for velocity (3D)
py::object upwind_3d_py(py::sequence velocity_seq, ScalarField<3>& field)
{
    if (len(velocity_seq) != 3)
    {
        throw std::runtime_error("Velocity must have exactly 3 elements for 3D");
    }

    std::array<double, 3> velocity;
    velocity[0] = velocity_seq[0].cast<double>();
    velocity[1] = velocity_seq[1].cast<double>();
    velocity[2] = velocity_seq[2].cast<double>();

    return upwind_3d(velocity, field);
}

// -------------------------------------------------------------------------
// In-place upwind operators (no allocation, for efficient time stepping)
// -------------------------------------------------------------------------

// 1D upwind operator - in-place version (no allocation)
void apply_upwind_1d(ScalarField<1>& output, double velocity, const ScalarField<1>& input)
{
    // Get the upwind expression (lazy)
    auto upwind_expr = samurai::upwind(velocity, input);

    // Evaluate directly into output field (single pass, no allocation)
    samurai::for_each_interval(output.mesh(),
                               [&output, &upwind_expr](std::size_t level, const default_interval& interval, const auto& index)
                               {
                                   output(level, interval, index) = upwind_expr(level, interval, index);
                               });
}

// 2D upwind operator - in-place version (no allocation)
void apply_upwind_2d(ScalarField<2>& output, const std::array<double, 2>& velocity, const ScalarField<2>& input)
{
    // Get the upwind expression (lazy)
    auto upwind_expr = samurai::upwind(velocity, input);

    // Evaluate directly into output field (single pass, no allocation)
    samurai::for_each_interval(output.mesh(),
                               [&output, &upwind_expr](std::size_t level, const default_interval& interval, const auto& index)
                               {
                                   output(level, interval, index) = upwind_expr(level, interval, index);
                               });
}

// 3D upwind operator - in-place version (no allocation)
void apply_upwind_3d(ScalarField<3>& output, const std::array<double, 3>& velocity, const ScalarField<3>& input)
{
    // Get the upwind expression (lazy)
    auto upwind_expr = samurai::upwind(velocity, input);

    // Evaluate directly into output field (single pass, no allocation)
    samurai::for_each_interval(output.mesh(),
                               [&output, &upwind_expr](std::size_t level, const default_interval& interval, const auto& index)
                               {
                                   output(level, interval, index) = upwind_expr(level, interval, index);
                               });
}

// Convenience wrapper accepting Python list/tuple for velocity (2D) - in-place version
void apply_upwind_2d_py(ScalarField<2>& output, py::sequence velocity_seq, const ScalarField<2>& input)
{
    if (len(velocity_seq) != 2)
    {
        throw std::runtime_error("Velocity must have exactly 2 elements for 2D");
    }

    std::array<double, 2> velocity;
    velocity[0] = velocity_seq[0].cast<double>();
    velocity[1] = velocity_seq[1].cast<double>();

    apply_upwind_2d(output, velocity, input);
}

// Convenience wrapper accepting Python list/tuple for velocity (3D) - in-place version
void apply_upwind_3d_py(ScalarField<3>& output, py::sequence velocity_seq, const ScalarField<3>& input)
{
    if (len(velocity_seq) != 3)
    {
        throw std::runtime_error("Velocity must have exactly 3 elements for 3D");
    }

    std::array<double, 3> velocity;
    velocity[0] = velocity_seq[0].cast<double>();
    velocity[1] = velocity_seq[1].cast<double>();
    velocity[2] = velocity_seq[2].cast<double>();

    apply_upwind_3d(output, velocity, input);
}

// Module initialization function for operator bindings
void init_operator_bindings(py::module_& m)
{
    // ============================================================
    // In-place upwind operators (efficient, no allocation)
    // ============================================================

    // Bind 1D in-place upwind operator
    m.def("apply_upwind_1d",
          &apply_upwind_1d,
          py::arg("output"),
          py::arg("velocity"),
          py::arg("input"),
          R"pbdoc(
        Apply upwind operator in-place (efficient, no allocation).

        Computes upwind flux and stores directly in output field.

        Parameters
        ----------
        output : ScalarField1D
            Output field (must be pre-allocated)
        velocity : float
            Advection velocity
        input : ScalarField1D
            Input scalar field

        Examples
        --------
        >>> import samurai as sam
        >>> flux = sam.ScalarField1D("flux", mesh, 0.0)
        >>> sam.apply_upwind_1d(flux, 1.0, u)
        >>> # Use in time step
        >>> sam.euler_update_1d(unp1, u, dt, flux)
        )pbdoc");

    // Bind 2D in-place upwind operator - std::array version
    m.def("apply_upwind_2d",
          &apply_upwind_2d,
          py::arg("output"),
          py::arg("velocity"),
          py::arg("input"),
          R"pbdoc(
        Apply upwind operator in-place (2D, efficient, no allocation).

        Parameters
        ----------
        output : ScalarField2D
            Output field (must be pre-allocated)
        velocity : std::array<double, 2>
            2D velocity vector [vx, vy]
        input : ScalarField2D
            Input scalar field

        Examples
        --------
        >>> flux = sam.ScalarField2D("flux", mesh, 0.0)
        >>> sam.apply_upwind_2d(flux, [1.0, 1.0], u)
        >>> sam.euler_update_2d(unp1, u, dt, flux)
        )pbdoc");

    // Bind 2D in-place upwind operator - Python sequence version
    m.def("apply_upwind_2d",
          &apply_upwind_2d_py,
          py::arg("output"),
          py::arg("velocity"),
          py::arg("input"),
          R"pbdoc(
        Apply upwind operator in-place (2D, Python sequence version).

        Parameters
        ----------
        output : ScalarField2D
            Output field (must be pre-allocated)
        velocity : sequence of float
            2D velocity vector [vx, vy] (list or tuple)
        input : ScalarField2D
            Input scalar field
        )pbdoc");

    // Bind 3D in-place upwind operator - std::array version
    m.def("apply_upwind_3d",
          &apply_upwind_3d,
          py::arg("output"),
          py::arg("velocity"),
          py::arg("input"),
          R"pbdoc(
        Apply upwind operator in-place (3D, efficient, no allocation).

        Parameters
        ----------
        output : ScalarField3D
            Output field (must be pre-allocated)
        velocity : std::array<double, 3>
            3D velocity vector [vx, vy, vz]
        input : ScalarField3D
            Input scalar field
        )pbdoc");

    // Bind 3D in-place upwind operator - Python sequence version
    m.def("apply_upwind_3d",
          &apply_upwind_3d_py,
          py::arg("output"),
          py::arg("velocity"),
          py::arg("input"),
          R"pbdoc(
        Apply upwind operator in-place (3D, Python sequence version).

        Parameters
        ----------
        output : ScalarField3D
            Output field (must be pre-allocated)
        velocity : sequence of float
            3D velocity vector [vx, vy, vz] (list or tuple)
        input : ScalarField3D
            Input scalar field
        )pbdoc");

    // ============================================================
    // Original upwind operators (return new fields)
    // ============================================================

    // Bind 1D upwind operator
    m.def("upwind",
          &upwind_1d,
          py::arg("velocity"),
          py::arg("field"),
          R"pbdoc(
        Upwind operator for 1D advection.

        Computes the upwind flux for a scalar field in 1D.

        Parameters
        ----------
        velocity : float
            Advection velocity (scalar for 1D)
        field : ScalarField1D
            Input scalar field

        Returns
        -------
        ScalarField1D
            New field containing upwind flux values

        Examples
        --------
        >>> import samurai as sam
        >>> mesh = sam.MRMesh1D(box, config)
        >>> u = sam.ScalarField1D("u", mesh)
        >>> flux = sam.upwind(1.0, u)
        >>> # Use in time step: unp1 = u - dt * flux
        )pbdoc");

    // Bind 2D upwind operator - std::array version
    m.def("upwind",
          &upwind_2d,
          py::arg("velocity"),
          py::arg("field"),
          R"pbdoc(
        Upwind operator for 2D advection (std::array version).

        Parameters
        ----------
        velocity : std::array<double, 2>
            2D velocity vector [vx, vy]
        field : ScalarField2D
            Input scalar field

        Returns
        -------
        ScalarField2D
            New field containing upwind flux values
        )pbdoc");

    // Bind 2D upwind operator - Python sequence version (more convenient)
    m.def("upwind",
          &upwind_2d_py,
          py::arg("velocity"),
          py::arg("field"),
          R"pbdoc(
        Upwind operator for 2D advection.

        Computes the upwind flux for a scalar field in 2D.

        Parameters
        ----------
        velocity : sequence of float
            2D velocity vector [vx, vy] (list or tuple)
        field : ScalarField2D
            Input scalar field

        Returns
        -------
        ScalarField2D
            New field containing upwind flux values

        Examples
        --------
        >>> import samurai as sam
        >>> mesh = sam.MRMesh2D(box, config)
        >>> u = sam.ScalarField2D("u", mesh)
        >>> velocity = [1.0, 1.0]  # [vx, vy]
        >>> flux = sam.upwind(velocity, u)
        >>> # Use in time step: unp1 = u - dt * flux
        )pbdoc");

    // Bind 3D upwind operator - std::array version
    m.def("upwind",
          &upwind_3d,
          py::arg("velocity"),
          py::arg("field"),
          R"pbdoc(
        Upwind operator for 3D advection (std::array version).

        Parameters
        ----------
        velocity : std::array<double, 3>
            3D velocity vector [vx, vy, vz]
        field : ScalarField3D
            Input scalar field

        Returns
        -------
        ScalarField3D
            New field containing upwind flux values
        )pbdoc");

    // Bind 3D upwind operator - Python sequence version (more convenient)
    m.def("upwind",
          &upwind_3d_py,
          py::arg("velocity"),
          py::arg("field"),
          R"pbdoc(
        Upwind operator for 3D advection.

        Computes the upwind flux for a scalar field in 3D.

        Parameters
        ----------
        velocity : sequence of float
            3D velocity vector [vx, vy, vz] (list or tuple)
        field : ScalarField3D
            Input scalar field

        Returns
        -------
        ScalarField3D
            New field containing upwind flux values

        Examples
        --------
        >>> import samurai as sam
        >>> mesh = sam.MRMesh3D(box, config)
        >>> u = sam.ScalarField3D("u", mesh)
        >>> velocity = [1.0, 1.0, 0.0]  # [vx, vy, vz]
        >>> flux = sam.upwind(velocity, u)
        >>> # Use in time step: unp1 = u - dt * flux
        )pbdoc");
}
