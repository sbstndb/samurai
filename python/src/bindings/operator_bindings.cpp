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

template <std::size_t dim, std::size_t n_comp, bool SOA = false>
using VectorField = samurai::VectorField<samurai::MRMesh<samurai::complete_mesh_config<samurai::mesh_config<dim>, samurai::MRMeshId>>, double, n_comp, SOA>;

// Specific VectorField types for Burgers equation (n_comp == dim)
using VectorField1D_2 = VectorField<1, 2, false>;
using VectorField2D_2 = VectorField<2, 2, false>;
using VectorField3D_3 = VectorField<3, 3, false>;

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
    result = conv_expr;

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
    result = conv_expr;

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
    result = conv_expr;

    return py::cast(result);
}

// ============================================================
// Linear WENO5 with constant velocity: f(u) = velocity * u
// ============================================================

// 1D linear WENO5 with constant velocity
py::object convection_weno5_linear_1d(double velocity, ScalarField<1>& field)
{
    using VelocityVector = samurai::VelocityVector<1>;
    auto& mesh = field.mesh();

    // Create output field with same mesh
    auto result = samurai::make_scalar_field<double>(field.name() + "_conv", mesh);

    // Create velocity vector
    VelocityVector vel;
    vel(0) = velocity;

    // Create WENO5 convection operator (linear with constant velocity)
    auto conv = samurai::make_convection_weno5<std::decay_t<decltype(field)>>(vel);

    // Get the expression and evaluate it
    auto conv_expr = conv(field);
    result = conv_expr;

    return py::cast(result);
}

// 2D linear WENO5 with constant velocity
py::object convection_weno5_linear_2d(const std::array<double, 2>& velocity, ScalarField<2>& field)
{
    using VelocityVector = samurai::VelocityVector<2>;
    auto& mesh = field.mesh();

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
    result = conv_expr;

    return py::cast(result);
}

// 2D linear WENO5 - Python sequence version
py::object convection_weno5_linear_2d_py(py::sequence velocity_seq, ScalarField<2>& field)
{
    if (len(velocity_seq) != 2)
    {
        throw std::runtime_error("Velocity must have exactly 2 elements for 2D");
    }

    std::array<double, 2> velocity;
    velocity[0] = velocity_seq[0].cast<double>();
    velocity[1] = velocity_seq[1].cast<double>();

    return convection_weno5_linear_2d(velocity, field);
}

// 3D linear WENO5 with constant velocity
py::object convection_weno5_linear_3d(const std::array<double, 3>& velocity, ScalarField<3>& field)
{
    using VelocityVector = samurai::VelocityVector<3>;
    auto& mesh = field.mesh();

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
    result = conv_expr;

    return py::cast(result);
}

// 3D linear WENO5 - Python sequence version
py::object convection_weno5_linear_3d_py(py::sequence velocity_seq, ScalarField<3>& field)
{
    if (len(velocity_seq) != 3)
    {
        throw std::runtime_error("Velocity must have exactly 3 elements for 3D");
    }

    std::array<double, 3> velocity;
    velocity[0] = velocity_seq[0].cast<double>();
    velocity[1] = velocity_seq[1].cast<double>();
    velocity[2] = velocity_seq[2].cast<double>();

    return convection_weno5_linear_3d(velocity, field);
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
    result = conv_expr;

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
    result = conv_expr;

    return py::cast(result);
}

// ============================================================
// Linear WENO5 with VectorField velocity: f(u) = velocity(x) * u
// For convection with spatially varying velocity fields
// ============================================================

// 2D ScalarField with VectorField2D_2 velocity
py::object convection_weno5_vectorfield_2d(VectorField2D_2& velocity, ScalarField<2>& field)
{
    auto& mesh = field.mesh();

    // Create output field with same mesh
    auto result = samurai::make_scalar_field<double>(field.name() + "_conv", mesh);

    // Create WENO5 convection operator with VectorField velocity
    // Uses template deduction to select the correct overload
    auto conv = samurai::make_convection_weno5<std::decay_t<decltype(field)>>(velocity);

    // Get the expression and evaluate it
    auto conv_expr = conv(field);
    result = conv_expr;

    return py::cast(result);
}

// 3D ScalarField with VectorField3D_3 velocity
py::object convection_weno5_vectorfield_3d(VectorField3D_3& velocity, ScalarField<3>& field)
{
    auto& mesh = field.mesh();

    // Create output field with same mesh
    auto result = samurai::make_scalar_field<double>(field.name() + "_conv", mesh);

    // Create WENO5 convection operator with VectorField velocity
    auto conv = samurai::make_convection_weno5<std::decay_t<decltype(field)>>(velocity);

    // Get the expression and evaluate it
    auto conv_expr = conv(field);
    result = conv_expr;

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
          py::arg("velocity"),
          py::arg("field"),
          R"pbdoc(
        WENO5 convection operator for 1D linear advection with constant velocity.

        5th order Weighted Essentially Non-Oscillatory scheme for linear convection.
        Flux: f(u) = velocity * u

        Parameters
        ----------
        velocity : float
            Advection velocity (scalar for 1D)
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
        >>> flux = sam.make_convection_weno5(1.0, u)
        >>> # Use in time step: unp1 = u - dt * flux
        )pbdoc");

    // 2D linear WENO5 - std::array version
    m.def("make_convection_weno5",
          &convection_weno5_linear_2d,
          py::arg("velocity"),
          py::arg("field"),
          R"pbdoc(
        WENO5 convection operator for 2D linear advection with constant velocity (std::array version).

        Parameters
        ----------
        velocity : std::array<double, 2>
            2D velocity vector [vx, vy]
        field : ScalarField2D
            Input scalar field

        Returns
        -------
        ScalarField2D
            New field containing convection flux values
        )pbdoc");

    // 2D linear WENO5 - Python sequence version
    m.def("make_convection_weno5",
          &convection_weno5_linear_2d_py,
          py::arg("velocity"),
          py::arg("field"),
          R"pbdoc(
        WENO5 convection operator for 2D linear advection with constant velocity.

        5th order Weighted Essentially Non-Oscillatory scheme for linear convection.
        Flux: f(u) = velocity · u

        Parameters
        ----------
        velocity : sequence of float
            2D velocity vector [vx, vy] (list or tuple)
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
        >>> velocity = [1.0, 1.0]  # [vx, vy]
        >>> flux = sam.make_convection_weno5(velocity, u)
        >>> # Use in time step: unp1 = u - dt * flux
        )pbdoc");

    // 3D linear WENO5 - std::array version
    m.def("make_convection_weno5",
          &convection_weno5_linear_3d,
          py::arg("velocity"),
          py::arg("field"),
          R"pbdoc(
        WENO5 convection operator for 3D linear advection with constant velocity (std::array version).

        Parameters
        ----------
        velocity : std::array<double, 3>
            3D velocity vector [vx, vy, vz]
        field : ScalarField3D
            Input scalar field

        Returns
        -------
        ScalarField3D
            New field containing convection flux values
        )pbdoc");

    // 3D linear WENO5 - Python sequence version
    m.def("make_convection_weno5",
          &convection_weno5_linear_3d_py,
          py::arg("velocity"),
          py::arg("field"),
          R"pbdoc(
        WENO5 convection operator for 3D linear advection with constant velocity.

        5th order Weighted Essentially Non-Oscillatory scheme for linear convection.
        Flux: f(u) = velocity · u

        Parameters
        ----------
        velocity : sequence of float
            3D velocity vector [vx, vy, vz] (list or tuple)
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
        >>> velocity = [1.0, 1.0, 0.0]  # [vx, vy, vz]
        >>> flux = sam.make_convection_weno5(velocity, u)
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
          py::arg("velocity"),
          py::arg("field"),
          R"pbdoc(
        WENO5 convection operator for 2D linear advection with VectorField velocity.

        5th order Weighted Essentially Non-Oscillatory scheme for linear convection
        with spatially varying velocity field.
        Flux: f(u) = velocity(x) · u

        Parameters
        ----------
        velocity : VectorField2D_2
            Velocity field [u, v] (can vary in space)
        field : ScalarField2D
            Input scalar field

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
        >>> flux = sam.make_convection_weno5(velocity, u)
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
          py::arg("velocity"),
          py::arg("field"),
          R"pbdoc(
        WENO5 convection operator for 3D linear advection with VectorField velocity.

        Similar to 2D version but for 3D meshes with VectorField3D_3 velocity.

        Flux: f(u) = velocity(x) · u

        Parameters
        ----------
        velocity : VectorField3D_3
            Velocity field [u, v, w] (can vary in space)
        field : ScalarField3D
            Input scalar field

        Returns
        -------
        ScalarField3D
            New field containing convection flux values
        )pbdoc");
}
