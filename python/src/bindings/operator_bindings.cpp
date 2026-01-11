// Samurai Python Bindings - Operator functions
//
// Bindings for finite volume operators like upwind, convection_weno5

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <samurai/algorithm.hpp>
#include <samurai/algorithm/update.hpp>
#include <samurai/field.hpp>
#include <samurai/mesh_config.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/schemes/fv/flux_based/explicit_flux_based_scheme__nonlin.hpp>
#include <samurai/schemes/fv/operators/convection_lin.hpp>
#include <samurai/schemes/fv/operators/convection_nonlin.hpp>
#include <samurai/stencil_field.hpp>
#include "common_types.hpp"

namespace py = pybind11;

// Use centralized type aliases from common_types.hpp
using namespace samurai::python::bindings;

// Note: operator_bindings.cpp uses Interval<int, long long int> for algorithms
// which differs from the algorithm_interval in common_types.hpp (Interval<double, std::size_t>)
// This is intentional for algorithm-specific use cases
using algorithm_interval = samurai::Interval<int, long long int>;

// 1D upwind operator - immediate evaluation version
py::object upwind_1d(ScalarField<1>& field, double velocity)
{
    auto& mesh = field.mesh();

    // Create output field with same mesh
    auto result = samurai::make_scalar_field<double>(field.name() + "_upwind", mesh);

    // Get the upwind expression
    auto upwind_expr = samurai::upwind(velocity, field);

    // Evaluate the expression immediately using for_each_interval
    samurai::for_each_interval(mesh,
                               [&result, &upwind_expr](std::size_t level, const algorithm_interval& interval, const auto& index)
                               {
                                   result(level, interval, index) = upwind_expr(level, interval, index);
                               });

    return py::cast(result);
}

// 2D upwind operator - immediate evaluation version
py::object upwind_2d(ScalarField<2>& field, const std::array<double, 2>& velocity)
{
    auto& mesh = field.mesh();

    // Create output field with same mesh
    auto result = samurai::make_scalar_field<double>(field.name() + "_upwind", mesh);

    // Get the upwind expression
    auto upwind_expr = samurai::upwind(velocity, field);

    // Evaluate the expression immediately using for_each_interval
    samurai::for_each_interval(mesh,
                               [&result, &upwind_expr](std::size_t level, const algorithm_interval& interval, const auto& index)
                               {
                                   result(level, interval, index) = upwind_expr(level, interval, index);
                               });

    return py::cast(result);
}

// 3D upwind operator - immediate evaluation version
py::object upwind_3d(ScalarField<3>& field, const std::array<double, 3>& velocity)
{
    auto& mesh = field.mesh();

    // Create output field with same mesh
    auto result = samurai::make_scalar_field<double>(field.name() + "_upwind", mesh);

    // Get the upwind expression
    auto upwind_expr = samurai::upwind(velocity, field);

    // Evaluate the expression immediately using for_each_interval
    samurai::for_each_interval(mesh,
                               [&result, &upwind_expr](std::size_t level, const algorithm_interval& interval, const auto& index)
                               {
                                   result(level, interval, index) = upwind_expr(level, interval, index);
                               });

    return py::cast(result);
}

// Convenience wrapper accepting Python list/tuple for velocity (2D)
py::object upwind_2d_py(ScalarField<2>& field, py::sequence velocity_seq)
{
    if (len(velocity_seq) != 2)
    {
        throw std::runtime_error("Velocity must have exactly 2 elements for 2D");
    }

    std::array<double, 2> velocity;
    velocity[0] = velocity_seq[0].cast<double>();
    velocity[1] = velocity_seq[1].cast<double>();

    return upwind_2d(field, velocity);
}

// Convenience wrapper accepting Python list/tuple for velocity (3D)
py::object upwind_3d_py(ScalarField<3>& field, py::sequence velocity_seq)
{
    if (len(velocity_seq) != 3)
    {
        throw std::runtime_error("Velocity must have exactly 3 elements for 3D");
    }

    std::array<double, 3> velocity;
    velocity[0] = velocity_seq[0].cast<double>();
    velocity[1] = velocity_seq[1].cast<double>();
    velocity[2] = velocity_seq[2].cast<double>();

    return upwind_3d(field, velocity);
}

// -------------------------------------------------------------------------
// In-place upwind operators (no allocation, for efficient time stepping)
// -------------------------------------------------------------------------

// 1D upwind operator - in-place version (no allocation)
void apply_upwind_1d(const ScalarField<1>& input, ScalarField<1>& output, double velocity)
{
    // Get the upwind expression (lazy)
    auto upwind_expr = samurai::upwind(velocity, input);

    // Evaluate directly into output field (single pass, no allocation)
    samurai::for_each_interval(output.mesh(),
                               [&output, &upwind_expr](std::size_t level, const algorithm_interval& interval, const auto& index)
                               {
                                   output(level, interval, index) = upwind_expr(level, interval, index);
                               });
}

// 2D upwind operator - in-place version (no allocation)
void apply_upwind_2d(const ScalarField<2>& input, ScalarField<2>& output, const std::array<double, 2>& velocity)
{
    // Get the upwind expression (lazy)
    auto upwind_expr = samurai::upwind(velocity, input);

    // Evaluate directly into output field (single pass, no allocation)
    samurai::for_each_interval(output.mesh(),
                               [&output, &upwind_expr](std::size_t level, const algorithm_interval& interval, const auto& index)
                               {
                                   output(level, interval, index) = upwind_expr(level, interval, index);
                               });
}

// 3D upwind operator - in-place version (no allocation)
void apply_upwind_3d(const ScalarField<3>& input, ScalarField<3>& output, const std::array<double, 3>& velocity)
{
    // Get the upwind expression (lazy)
    auto upwind_expr = samurai::upwind(velocity, input);

    // Evaluate directly into output field (single pass, no allocation)
    samurai::for_each_interval(output.mesh(),
                               [&output, &upwind_expr](std::size_t level, const algorithm_interval& interval, const auto& index)
                               {
                                   output(level, interval, index) = upwind_expr(level, interval, index);
                               });
}

// Convenience wrapper accepting Python list/tuple for velocity (2D) - in-place version
void apply_upwind_2d_py(const ScalarField<2>& input, ScalarField<2>& output, py::sequence velocity_seq)
{
    if (len(velocity_seq) != 2)
    {
        throw std::runtime_error("Velocity must have exactly 2 elements for 2D");
    }

    std::array<double, 2> velocity;
    velocity[0] = velocity_seq[0].cast<double>();
    velocity[1] = velocity_seq[1].cast<double>();

    apply_upwind_2d(input, output, velocity);
}

// Convenience wrapper accepting Python list/tuple for velocity (3D) - in-place version
void apply_upwind_3d_py(const ScalarField<3>& input, ScalarField<3>& output, py::sequence velocity_seq)
{
    if (len(velocity_seq) != 3)
    {
        throw std::runtime_error("Velocity must have exactly 3 elements for 3D");
    }

    std::array<double, 3> velocity;
    velocity[0] = velocity_seq[0].cast<double>();
    velocity[1] = velocity_seq[1].cast<double>();
    velocity[2] = velocity_seq[2].cast<double>();

    apply_upwind_3d(input, output, velocity);
}

// -------------------------------------------------------------------------
// WENO5 Convection operators (5th order Weighted Essentially Non-Oscillatory)
// -------------------------------------------------------------------------

// ============================================================
// Non-linear WENO5 (for Burgers equation): f(u) = u^2 or u(d)*u
// ============================================================

// 1D non-linear WENO5 (scalar Burgers)
py::object convection_weno5_nonlin_1d(ScalarField<1>& field)
{
    auto& mesh = field.mesh();

    // Create output field with same mesh
    auto result = samurai::make_scalar_field<double>(field.name() + "_conv", mesh);

    // Create WENO5 convection operator (nonlinear)
    auto conv = samurai::make_convection_weno5<std::decay_t<decltype(field)>>();

    // Get the expression and evaluate it
    auto conv_expr = conv(field);
    result         = conv_expr;

    return py::cast(result);
}

// 2D non-linear WENO5 (scalar Burgers)
py::object convection_weno5_nonlin_2d(ScalarField<2>& field)
{
    auto& mesh = field.mesh();

    // Create output field with same mesh
    auto result = samurai::make_scalar_field<double>(field.name() + "_conv", mesh);

    // Create WENO5 convection operator (nonlinear)
    auto conv = samurai::make_convection_weno5<std::decay_t<decltype(field)>>();

    // Get the expression and evaluate it
    auto conv_expr = conv(field);
    result         = conv_expr;

    return py::cast(result);
}

// 3D non-linear WENO5 (scalar Burgers)
py::object convection_weno5_nonlin_3d(ScalarField<3>& field)
{
    auto& mesh = field.mesh();

    // Create output field with same mesh
    auto result = samurai::make_scalar_field<double>(field.name() + "_conv", mesh);

    // Create WENO5 convection operator (nonlinear)
    auto conv = samurai::make_convection_weno5<std::decay_t<decltype(field)>>();

    // Get the expression and evaluate it
    auto conv_expr = conv(field);
    result         = conv_expr;

    return py::cast(result);
}

// ============================================================
// Linear WENO5 with constant velocity: f(u) = velocity * u
// ============================================================

// 1D linear WENO5 with constant velocity
py::object convection_weno5_linear_1d(ScalarField<1>& field, double velocity)
{
    using VelocityVector = samurai::VelocityVector<1>;
    auto& mesh           = field.mesh();

    // Create output field with same mesh
    auto result = samurai::make_scalar_field<double>(field.name() + "_conv", mesh);

    // Create velocity vector
    VelocityVector vel;
    vel(0) = velocity;

    // Create WENO5 convection operator (linear with constant velocity)
    auto conv = samurai::make_convection_weno5<std::decay_t<decltype(field)>>(vel);

    // Get the expression and evaluate it
    auto conv_expr = conv(field);
    result         = conv_expr;

    return py::cast(result);
}

// 2D linear WENO5 with constant velocity
py::object convection_weno5_linear_2d(ScalarField<2>& field, const std::array<double, 2>& velocity)
{
    using VelocityVector = samurai::VelocityVector<2>;
    auto& mesh           = field.mesh();

    // Create output field with same mesh
    auto result = samurai::make_scalar_field<double>(field.name() + "_conv", mesh);

    // Create velocity vector
    VelocityVector vel;
    vel(0) = velocity[0];
    vel(1) = velocity[1];

    // Create WENO5 convection operator (linear with constant velocity)
    auto conv = samurai::make_convection_weno5<std::decay_t<decltype(field)>>(vel);

    // Get the expression and evaluate it
    auto conv_expr = conv(field);
    result         = conv_expr;

    return py::cast(result);
}

// 2D linear WENO5 - Python sequence version
py::object convection_weno5_linear_2d_py(ScalarField<2>& field, py::sequence velocity_seq)
{
    if (len(velocity_seq) != 2)
    {
        throw std::runtime_error("Velocity must have exactly 2 elements for 2D");
    }

    std::array<double, 2> velocity;
    velocity[0] = velocity_seq[0].cast<double>();
    velocity[1] = velocity_seq[1].cast<double>();

    return convection_weno5_linear_2d(field, velocity);
}

// 3D linear WENO5 with constant velocity
py::object convection_weno5_linear_3d(ScalarField<3>& field, const std::array<double, 3>& velocity)
{
    using VelocityVector = samurai::VelocityVector<3>;
    auto& mesh           = field.mesh();

    // Create output field with same mesh
    auto result = samurai::make_scalar_field<double>(field.name() + "_conv", mesh);

    // Create velocity vector
    VelocityVector vel;
    vel(0) = velocity[0];
    vel(1) = velocity[1];
    vel(2) = velocity[2];

    // Create WENO5 convection operator (linear with constant velocity)
    auto conv = samurai::make_convection_weno5<std::decay_t<decltype(field)>>(vel);

    // Get the expression and evaluate it
    auto conv_expr = conv(field);
    result         = conv_expr;

    return py::cast(result);
}

// 3D linear WENO5 - Python sequence version
py::object convection_weno5_linear_3d_py(ScalarField<3>& field, py::sequence velocity_seq)
{
    if (len(velocity_seq) != 3)
    {
        throw std::runtime_error("Velocity must have exactly 3 elements for 3D");
    }

    std::array<double, 3> velocity;
    velocity[0] = velocity_seq[0].cast<double>();
    velocity[1] = velocity_seq[1].cast<double>();
    velocity[2] = velocity_seq[2].cast<double>();

    return convection_weno5_linear_3d(field, velocity);
}

// ============================================================
// Non-linear WENO5 for VectorField (Burgers equation): f(u) = u(d)*u
// ============================================================

// 2D non-linear WENO5 for VectorField2D_2 (Burgers 2D)
py::object convection_weno5_nonlin_vector_2d(VectorField2D_2& field)
{
    auto& mesh = field.mesh();

    // Create output field with same mesh
    auto result = samurai::make_vector_field<double, 2, false>(field.name() + "_conv", mesh);

    // Create WENO5 convection operator (nonlinear for vector field)
    auto conv = samurai::make_convection_weno5<std::decay_t<decltype(field)>>();

    // Get the expression and evaluate it
    auto conv_expr = conv(field);
    result         = conv_expr;

    return py::cast(result);
}

// 3D non-linear WENO5 for VectorField3D_3 (Burgers 3D)
py::object convection_weno5_nonlin_vector_3d(VectorField3D_3& field)
{
    auto& mesh = field.mesh();

    // Create output field with same mesh
    auto result = samurai::make_vector_field<double, 3, false>(field.name() + "_conv", mesh);

    // Create WENO5 convection operator (nonlinear for vector field)
    auto conv = samurai::make_convection_weno5<std::decay_t<decltype(field)>>();

    // Get the expression and evaluate it
    auto conv_expr = conv(field);
    result         = conv_expr;

    return py::cast(result);
}

// ============================================================
// Linear WENO5 with VectorField velocity: f(u) = velocity(x) * u
// For convection with spatially varying velocity fields
// ============================================================

// 2D ScalarField with VectorField2D_2 velocity
py::object convection_weno5_vectorfield_2d(ScalarField<2>& field, VectorField2D_2& velocity)
{
    auto& mesh = field.mesh();

    // Create output field with same mesh
    auto result = samurai::make_scalar_field<double>(field.name() + "_conv", mesh);

    // Create WENO5 convection operator with VectorField velocity
    // Uses template deduction to select the correct overload
    auto conv = samurai::make_convection_weno5<std::decay_t<decltype(field)>>(velocity);

    // Get the expression and evaluate it
    auto conv_expr = conv(field);
    result         = conv_expr;

    return py::cast(result);
}

// 3D ScalarField with VectorField3D_3 velocity
py::object convection_weno5_vectorfield_3d(ScalarField<3>& field, VectorField3D_3& velocity)
{
    auto& mesh = field.mesh();

    // Create output field with same mesh
    auto result = samurai::make_scalar_field<double>(field.name() + "_conv", mesh);

    // Create WENO5 convection operator with VectorField velocity
    auto conv = samurai::make_convection_weno5<std::decay_t<decltype(field)>>(velocity);

    // Get the expression and evaluate it
    auto conv_expr = conv(field);
    result         = conv_expr;

    return py::cast(result);
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
          py::arg("input"),
          py::arg("output"),
          py::arg("velocity"),
          R"pbdoc(
        Apply upwind operator in-place (efficient, no allocation).

        Computes upwind flux and stores directly in output field.

        Parameters
        ----------
        input : ScalarField1D
            Input scalar field
        output : ScalarField1D
            Output field (must be pre-allocated)
        velocity : float
            Advection velocity

        Examples
        --------
        >>> import samurai as sam
        >>> flux = sam.ScalarField1D("flux", mesh, 0.0)
        >>> sam.apply_upwind_1d(u, flux, 1.0)
        >>> # Use in time step
        >>> sam.euler_update_1d(unp1, u, dt, flux)
        )pbdoc");

    // Bind 2D in-place upwind operator - std::array version
    m.def("apply_upwind_2d",
          &apply_upwind_2d,
          py::arg("input"),
          py::arg("output"),
          py::arg("velocity"),
          R"pbdoc(
        Apply upwind operator in-place (2D, efficient, no allocation).

        Parameters
        ----------
        input : ScalarField2D
            Input scalar field
        output : ScalarField2D
            Output field (must be pre-allocated)
        velocity : std::array<double, 2>
            2D velocity vector [vx, vy]

        Examples
        --------
        >>> flux = sam.ScalarField2D("flux", mesh, 0.0)
        >>> sam.apply_upwind_2d(u, flux, [1.0, 1.0])
        >>> sam.euler_update_2d(unp1, u, dt, flux)
        )pbdoc");

    // Bind 2D in-place upwind operator - Python sequence version
    m.def("apply_upwind_2d",
          &apply_upwind_2d_py,
          py::arg("input"),
          py::arg("output"),
          py::arg("velocity"),
          R"pbdoc(
        Apply upwind operator in-place (2D, Python sequence version).

        Parameters
        ----------
        input : ScalarField2D
            Input scalar field
        output : ScalarField2D
            Output field (must be pre-allocated)
        velocity : sequence of float
            2D velocity vector [vx, vy] (list or tuple)
        )pbdoc");

    // Bind 3D in-place upwind operator - std::array version
    m.def("apply_upwind_3d",
          &apply_upwind_3d,
          py::arg("input"),
          py::arg("output"),
          py::arg("velocity"),
          R"pbdoc(
        Apply upwind operator in-place (3D, efficient, no allocation).

        Parameters
        ----------
        input : ScalarField3D
            Input scalar field
        output : ScalarField3D
            Output field (must be pre-allocated)
        velocity : std::array<double, 3>
            3D velocity vector [vx, vy, vz]
        )pbdoc");

    // Bind 3D in-place upwind operator - Python sequence version
    m.def("apply_upwind_3d",
          &apply_upwind_3d_py,
          py::arg("input"),
          py::arg("output"),
          py::arg("velocity"),
          R"pbdoc(
        Apply upwind operator in-place (3D, Python sequence version).

        Parameters
        ----------
        input : ScalarField3D
            Input scalar field
        output : ScalarField3D
            Output field (must be pre-allocated)
        velocity : sequence of float
            3D velocity vector [vx, vy, vz] (list or tuple)
        )pbdoc");

    // ============================================================
    // Original upwind operators (return new fields)
    // ============================================================

    // Bind 1D upwind operator
    m.def("upwind",
          &upwind_1d,
          py::arg("field"),
          py::arg("velocity"),
          R"pbdoc(
        Upwind operator for 1D advection.

        Computes the upwind flux for a scalar field in 1D.

        Parameters
        ----------
        field : ScalarField1D
            Input scalar field
        velocity : float
            Advection velocity (scalar for 1D)

        Returns
        -------
        ScalarField1D
            New field containing upwind flux values

        Examples
        --------
        >>> import samurai as sam
        >>> mesh = sam.MRMesh1D(box, config)
        >>> u = sam.ScalarField1D("u", mesh)
        >>> flux = sam.upwind(u, 1.0)
        >>> # Use in time step: unp1 = u - dt * flux
        )pbdoc");

    // Bind 2D upwind operator - std::array version
    m.def("upwind",
          &upwind_2d,
          py::arg("field"),
          py::arg("velocity"),
          R"pbdoc(
        Upwind operator for 2D advection (std::array version).

        Parameters
        ----------
        field : ScalarField2D
            Input scalar field
        velocity : std::array<double, 2>
            2D velocity vector [vx, vy]

        Returns
        -------
        ScalarField2D
            New field containing upwind flux values
        )pbdoc");

    // Bind 2D upwind operator - Python sequence version (more convenient)
    m.def("upwind",
          &upwind_2d_py,
          py::arg("field"),
          py::arg("velocity"),
          R"pbdoc(
        Upwind operator for 2D advection.

        Computes the upwind flux for a scalar field in 2D.

        Parameters
        ----------
        field : ScalarField2D
            Input scalar field
        velocity : sequence of float
            2D velocity vector [vx, vy] (list or tuple)

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
        >>> flux = sam.upwind(u, velocity)
        >>> # Use in time step: unp1 = u - dt * flux
        )pbdoc");

    // Bind 3D upwind operator - std::array version
    m.def("upwind",
          &upwind_3d,
          py::arg("field"),
          py::arg("velocity"),
          R"pbdoc(
        Upwind operator for 3D advection (std::array version).

        Parameters
        ----------
        field : ScalarField3D
            Input scalar field
        velocity : std::array<double, 3>
            3D velocity vector [vx, vy, vz]

        Returns
        -------
        ScalarField3D
            New field containing upwind flux values
        )pbdoc");

    // Bind 3D upwind operator - Python sequence version (more convenient)
    m.def("upwind",
          &upwind_3d_py,
          py::arg("field"),
          py::arg("velocity"),
          R"pbdoc(
        Upwind operator for 3D advection.

        Computes the upwind flux for a scalar field in 3D.

        Parameters
        ----------
        field : ScalarField3D
            Input scalar field
        velocity : sequence of float
            3D velocity vector [vx, vy, vz] (list or tuple)

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
        >>> flux = sam.upwind(u, velocity)
        >>> # Use in time step: unp1 = u - dt * flux
        )pbdoc");

    // ============================================================
    // WENO5 Convection operators
    // ============================================================

    // ------------------------------------------------------------
    // Non-linear WENO5 (Burgers equation): f(u) = u^2 or u(d)*u
    // ------------------------------------------------------------

    // 1D non-linear WENO5
    m.def("make_convection_weno5",
          &convection_weno5_nonlin_1d,
          py::arg("field"),
          R"pbdoc(
        WENO5 convection operator for 1D Burgers equation (nonlinear).

        5th order Weighted Essentially Non-Oscillatory scheme for nonlinear convection.
        Flux: f(u) = u^2 (scalar) or f(u) = u(d)*u (vector)

        Parameters
        ----------
        field : ScalarField1D
            Input scalar field

        Returns
        -------
        ScalarField1D
            New field containing convection flux values

        Examples
        --------
        >>> import samurai as sam
        >>> mesh = sam.MRMesh1D(box, config)
        >>> u = sam.ScalarField1D("u", mesh)
        >>> flux = sam.make_convection_weno5(u)
        >>> # Use in time step: unp1 = u - dt * flux
        >>> # For Burgers: flux = u^2/2, so use: unp1 = u - dt * flux
        )pbdoc");

    // 2D non-linear WENO5
    m.def("make_convection_weno5",
          &convection_weno5_nonlin_2d,
          py::arg("field"),
          R"pbdoc(
        WENO5 convection operator for 2D Burgers equation (nonlinear).

        5th order Weighted Essentially Non-Oscillatory scheme for nonlinear convection.
        Flux: f(u) = u^2 (scalar) or f(u) = u(d)*u (vector)

        Parameters
        ----------
        field : ScalarField2D
            Input scalar field

        Returns
        -------
        ScalarField2D
            New field containing convection flux values

        Examples
        --------
        >>> import samurai as sam
        >>> mesh = sam.MRMesh2D(box, config)
        >>> u = sam.ScalarField2D("u", mesh)
        >>> flux = sam.make_convection_weno5(u)
        >>> # Use in time step: unp1 = u - dt * flux
        )pbdoc");

    // 3D non-linear WENO5
    m.def("make_convection_weno5",
          &convection_weno5_nonlin_3d,
          py::arg("field"),
          R"pbdoc(
        WENO5 convection operator for 3D Burgers equation (nonlinear).

        5th order Weighted Essentially Non-Oscillatory scheme for nonlinear convection.
        Flux: f(u) = u^2 (scalar) or f(u) = u(d)*u (vector)

        Parameters
        ----------
        field : ScalarField3D
            Input scalar field

        Returns
        -------
        ScalarField3D
            New field containing convection flux values

        Examples
        --------
        >>> import samurai as sam
        >>> mesh = sam.MRMesh3D(box, config)
        >>> u = sam.ScalarField3D("u", mesh)
        >>> flux = sam.make_convection_weno5(u)
        >>> # Use in time step: unp1 = u - dt * flux
        )pbdoc");

    // ------------------------------------------------------------
    // Linear WENO5 with constant velocity: f(u) = velocity * u
    // ------------------------------------------------------------

    // 1D linear WENO5
    m.def("make_convection_weno5",
          &convection_weno5_linear_1d,
          py::arg("field"),
          py::arg("velocity"),
          R"pbdoc(
        WENO5 convection operator for 1D linear advection with constant velocity.

        5th order Weighted Essentially Non-Oscillatory scheme for linear convection.
        Flux: f(u) = velocity * u

        Parameters
        ----------
        field : ScalarField1D
            Input scalar field
        velocity : float
            Advection velocity (scalar for 1D)

        Returns
        -------
        ScalarField1D
            New field containing convection flux values

        Examples
        --------
        >>> import samurai as sam
        >>> mesh = sam.MRMesh1D(box, config)
        >>> u = sam.ScalarField1D("u", mesh)
        >>> flux = sam.make_convection_weno5(u, 1.0)
        >>> # Use in time step: unp1 = u - dt * flux
        )pbdoc");

    // 2D linear WENO5 - std::array version
    m.def("make_convection_weno5",
          &convection_weno5_linear_2d,
          py::arg("field"),
          py::arg("velocity"),
          R"pbdoc(
        WENO5 convection operator for 2D linear advection with constant velocity (std::array version).

        Parameters
        ----------
        field : ScalarField2D
            Input scalar field
        velocity : std::array<double, 2>
            2D velocity vector [vx, vy]

        Returns
        -------
        ScalarField2D
            New field containing convection flux values
        )pbdoc");

    // 2D linear WENO5 - Python sequence version
    m.def("make_convection_weno5",
          &convection_weno5_linear_2d_py,
          py::arg("field"),
          py::arg("velocity"),
          R"pbdoc(
        WENO5 convection operator for 2D linear advection with constant velocity.

        5th order Weighted Essentially Non-Oscillatory scheme for linear convection.
        Flux: f(u) = velocity · u

        Parameters
        ----------
        field : ScalarField2D
            Input scalar field
        velocity : sequence of float
            2D velocity vector [vx, vy] (list or tuple)

        Returns
        -------
        ScalarField2D
            New field containing convection flux values

        Examples
        --------
        >>> import samurai as sam
        >>> mesh = sam.MRMesh2D(box, config)
        >>> u = sam.ScalarField2D("u", mesh)
        >>> velocity = [1.0, 1.0]  # [vx, vy]
        >>> flux = sam.make_convection_weno5(u, velocity)
        >>> # Use in time step: unp1 = u - dt * flux
        )pbdoc");

    // 3D linear WENO5 - std::array version
    m.def("make_convection_weno5",
          &convection_weno5_linear_3d,
          py::arg("field"),
          py::arg("velocity"),
          R"pbdoc(
        WENO5 convection operator for 3D linear advection with constant velocity (std::array version).

        Parameters
        ----------
        field : ScalarField3D
            Input scalar field
        velocity : std::array<double, 3>
            3D velocity vector [vx, vy, vz]

        Returns
        -------
        ScalarField3D
            New field containing convection flux values
        )pbdoc");

    // 3D linear WENO5 - Python sequence version
    m.def("make_convection_weno5",
          &convection_weno5_linear_3d_py,
          py::arg("field"),
          py::arg("velocity"),
          R"pbdoc(
        WENO5 convection operator for 3D linear advection with constant velocity.

        5th order Weighted Essentially Non-Oscillatory scheme for linear convection.
        Flux: f(u) = velocity · u

        Parameters
        ----------
        field : ScalarField3D
            Input scalar field
        velocity : sequence of float
            3D velocity vector [vx, vy, vz] (list or tuple)

        Returns
        -------
        ScalarField3D
            New field containing convection flux values

        Examples
        --------
        >>> import samurai as sam
        >>> mesh = sam.MRMesh3D(box, config)
        >>> u = sam.ScalarField3D("u", mesh)
        >>> velocity = [1.0, 1.0, 0.0]  # [vx, vy, vz]
        >>> flux = sam.make_convection_weno5(u, velocity)
        >>> # Use in time step: unp1 = u - dt * flux
        )pbdoc");

    // ============================================================
    // WENO5 Convection operators for VectorField (Burgers equation)
    // ============================================================

    // 2D non-linear WENO5 for VectorField2D_2 (Burgers 2D)
    m.def("make_convection_weno5",
          &convection_weno5_nonlin_vector_2d,
          py::arg("field"),
          R"pbdoc(
        WENO5 convection operator for 2D Burgers equation (nonlinear, vector field).

        5th order Weighted Essentially Non-Oscillatory scheme for nonlinear convection.
        Solves the vector Burgers equation: ∂u/∂t + u·∇u = 0
        where u = [u, v] is the velocity vector field.

        Flux: F(u) = u ⊗ u = [[u^2, uv], [uv, v^2]]

        Parameters
        ----------
        field : VectorField2D_2
            Input vector field [u, v] with 2 components

        Returns
        -------
        VectorField2D_2
            New field containing convection flux values

        Examples
        --------
        >>> import samurai as sam
        >>> mesh = sam.MRMesh2D(box, config)
        >>> u = sam.VectorField2D_2("u", mesh, 0.0)
        >>> # Initialize u with velocity field
        >>> flux = sam.make_convection_weno5(u)
        >>> # RK3 time stepping
        >>> u1 = u - dt * flux
        >>> u2 = 3./4 * u + 1./4 * (u1 - dt * sam.make_convection_weno5(u1))
        >>> unp1 = 1./3 * u + 2./3 * (u2 - dt * sam.make_convection_weno5(u2))
        )pbdoc");

    // 3D non-linear WENO5 for VectorField3D_3 (Burgers 3D)
    m.def("make_convection_weno5",
          &convection_weno5_nonlin_vector_3d,
          py::arg("field"),
          R"pbdoc(
        WENO5 convection operator for 3D Burgers equation (nonlinear, vector field).

        5th order Weighted Essentially Non-Oscillatory scheme for nonlinear convection.
        Solves the vector Burgers equation: ∂u/∂t + u·∇u = 0
        where u = [u, v, w] is the velocity vector field.

        Flux: F(u) = u ⊗ u (tensor product)

        Parameters
        ----------
        field : VectorField3D_3
            Input vector field [u, v, w] with 3 components

        Returns
        -------
        VectorField3D_3
            New field containing convection flux values

        Examples
        --------
        >>> import samurai as sam
        >>> mesh = sam.MRMesh3D(box, config)
        >>> u = sam.VectorField3D_3("u", mesh, 0.0)
        >>> # Initialize u with velocity field
        >>> flux = sam.make_convection_weno5(u)
        >>> # RK3 time stepping
        >>> u1 = u - dt * flux
        >>> u2 = 3./4 * u + 1./4 * (u1 - dt * sam.make_convection_weno5(u1))
        >>> unp1 = 1./3 * u + 2./3 * (u2 - dt * sam.make_convection_weno5(u2))
        )pbdoc");

    // ============================================================
    // Linear WENO5 with VectorField velocity: f(u) = velocity(x) * u
    // ============================================================

    // 2D ScalarField with VectorField2D_2 velocity
    m.def("make_convection_weno5",
          &convection_weno5_vectorfield_2d,
          py::arg("field"),
          py::arg("velocity"),
          R"pbdoc(
        WENO5 convection operator for 2D linear advection with VectorField velocity.

        5th order Weighted Essentially Non-Oscillatory scheme for linear convection
        with spatially varying velocity field.
        Flux: f(u) = velocity(x) · u

        Parameters
        ----------
        field : ScalarField2D
            Input scalar field
        velocity : VectorField2D_2
            Velocity field [u, v] (can vary in space)

        Returns
        -------
        ScalarField2D
            New field containing convection flux values

        Examples
        --------
        >>> import samurai as sam
        >>> mesh = sam.MRMesh2D(domain, config)
        >>> velocity = sam.VectorField2D_2("vel", mesh, 0.0)
        >>> # Initialize velocity field (constant or space-dependent)
        >>> velocity = sam.make_vector_field(mesh, "velocity",
        ...     lambda center: [1.0, -1.0], 2)
        >>> u = sam.ScalarField2D("u", mesh, 0.0)
        >>> flux = sam.make_convection_weno5(u, velocity)
        >>> # Use in time step: unp1 = u - dt * flux

        Notes
        -----
        This overload is useful for:
        - Obstacle problems where velocity varies near boundaries
        - Complex flow fields with spatial variation
        - Consistent velocity treatment across mesh adaptation
        )pbdoc");

    // 3D ScalarField with VectorField3D_3 velocity
    m.def("make_convection_weno5",
          &convection_weno5_vectorfield_3d,
          py::arg("field"),
          py::arg("velocity"),
          R"pbdoc(
        WENO5 convection operator for 3D linear advection with VectorField velocity.

        Similar to 2D version but for 3D meshes with VectorField3D_3 velocity.

        Flux: f(u) = velocity(x) · u

        Parameters
        ----------
        field : ScalarField3D
            Input scalar field
        velocity : VectorField3D_3
            Velocity field [u, v, w] (can vary in space)

        Returns
        -------
        ScalarField3D
            New field containing convection flux values
        )pbdoc");

    // ============================================================
    // Create operators submodule for better organization
    // ============================================================

    // Create the operators submodule
    py::module_ operators = m.def_submodule("operators", "Finite volume operators for AMR");

    // Reference all operator functions in the submodule
    // This maintains backward compatibility (operators still in main module)
    // while also providing them in the organized submodule

    // In-place upwind operators
    operators.attr("apply_upwind_1d") = m.attr("apply_upwind_1d");
    operators.attr("apply_upwind_2d") = m.attr("apply_upwind_2d");
    operators.attr("apply_upwind_3d") = m.attr("apply_upwind_3d");

    // Upwind operators (return new fields)
    operators.attr("upwind") = m.attr("upwind");

    // WENO5 convection operators
    operators.attr("make_convection_weno5") = m.attr("make_convection_weno5");

    // Alias for shorter name (without 'make_' prefix, more Pythonic)
    operators.attr("convection_weno5") = m.attr("make_convection_weno5");
}
