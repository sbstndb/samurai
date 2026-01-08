// Samurai Python Bindings - Boundary Conditions
//
// Bindings for make_bc and boundary condition types

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <samurai/bc/bc.hpp>
#include <samurai/bc/dirichlet.hpp>
#include <samurai/bc/neumann.hpp>
#include <samurai/field.hpp>
#include <samurai/mesh_config.hpp>
#include <samurai/mr/mesh.hpp>

namespace py = pybind11;

// Type aliases matching field_bindings.cpp
template <std::size_t dim>
using MRMesh = samurai::MRMesh<samurai::complete_mesh_config<samurai::mesh_config<dim>, samurai::MRMeshId>>;

template <std::size_t dim>
using ScalarField = samurai::ScalarField<MRMesh<dim>, double>;

template <std::size_t dim, std::size_t n_comp, bool SOA = false>
using VectorField = samurai::VectorField<MRMesh<dim>, double, n_comp, SOA>;

// Specific VectorField types for Burgers equation (n_comp == dim)
using VectorField1D_2 = VectorField<1, 2, false>;
using VectorField2D_2 = VectorField<2, 2, false>;
using VectorField3D_3 = VectorField<3, 3, false>;

// ============================================================
// Dirichlet boundary condition bindings
// ============================================================

// Helper to attach Dirichlet BC for a specific dimension and order
// Returns void because the BC is attached to the field internally
template <std::size_t dim, std::size_t order>
void make_dirichlet_bc_scalar(ScalarField<dim>& field, double value)
{
    using DirichletOrder = samurai::Dirichlet<order>;
    samurai::make_bc<DirichletOrder>(field, value);
    // BC is now attached to the field - no need to return anything
}

// Wrapper function to dispatch based on order parameter
template <std::size_t dim>
void make_dirichlet_bc_dispatch(ScalarField<dim>& field, double value, std::size_t order)
{
    switch (order)
    {
        case 1:
            return make_dirichlet_bc_scalar<dim, 1>(field, value);
        case 2:
            return make_dirichlet_bc_scalar<dim, 2>(field, value);
        case 3:
            return make_dirichlet_bc_scalar<dim, 3>(field, value);
        case 4:
            return make_dirichlet_bc_scalar<dim, 4>(field, value);
        default:
            throw std::runtime_error("Dirichlet BC order must be between 1 and 4, got " + std::to_string(order));
    }
}

// 1D wrapper
void make_dirichlet_bc_1d(ScalarField<1>& field, double value, std::size_t order)
{
    make_dirichlet_bc_dispatch<1>(field, value, order);
}

// 2D wrapper
void make_dirichlet_bc_2d(ScalarField<2>& field, double value, std::size_t order)
{
    make_dirichlet_bc_dispatch<2>(field, value, order);
}

// 3D wrapper
void make_dirichlet_bc_3d(ScalarField<3>& field, double value, std::size_t order)
{
    make_dirichlet_bc_dispatch<3>(field, value, order);
}

// ============================================================
// Neumann boundary condition bindings
// Note: Neumann BC only supports order 1 (first-order accurate)
// ============================================================

// Helper to attach Neumann BC for a specific dimension
// value is the derivative (∂u/∂n) at the boundary
template <std::size_t dim>
void make_neumann_bc_scalar(ScalarField<dim>& field, double value)
{
    samurai::make_bc<samurai::Neumann<1>>(field, value);
}

// Helper to attach Neumann BC with default (zero derivative)
template <std::size_t dim>
void make_neumann_bc_default(ScalarField<dim>& field)
{
    samurai::make_bc<samurai::Neumann<1>>(field);
}

// 1D wrappers
void make_neumann_bc_1d(ScalarField<1>& field, double value)
{
    make_neumann_bc_scalar<1>(field, value);
}

void make_neumann_bc_1d_default(ScalarField<1>& field)
{
    make_neumann_bc_default<1>(field);
}

// 2D wrappers
void make_neumann_bc_2d(ScalarField<2>& field, double value)
{
    make_neumann_bc_scalar<2>(field, value);
}

void make_neumann_bc_2d_default(ScalarField<2>& field)
{
    make_neumann_bc_default<2>(field);
}

// 3D wrappers
void make_neumann_bc_3d(ScalarField<3>& field, double value)
{
    make_neumann_bc_scalar<3>(field, value);
}

void make_neumann_bc_3d_default(ScalarField<3>& field)
{
    make_neumann_bc_default<3>(field);
}

// ============================================================
// Dirichlet boundary condition bindings for VectorField
// ============================================================

// Helper to attach Dirichlet BC for VectorField2D_2 with list of values
template <std::size_t order>
void make_dirichlet_bc_vector_2d_2(VectorField2D_2& field, const std::vector<double>& values)
{
    if (values.size() != 2)
    {
        throw std::runtime_error("Expected 2 values for VectorField2D_2, got " + std::to_string(values.size()));
    }
    using DirichletOrder = samurai::Dirichlet<order>;
    samurai::make_bc<DirichletOrder>(field, values[0], values[1]);
}

// Wrapper function to dispatch based on order parameter
void make_dirichlet_bc_vector_2d_2_dispatch(VectorField2D_2& field, const std::vector<double>& values, std::size_t order)
{
    switch (order)
    {
        case 1:
            return make_dirichlet_bc_vector_2d_2<1>(field, values);
        case 2:
            return make_dirichlet_bc_vector_2d_2<2>(field, values);
        case 3:
            return make_dirichlet_bc_vector_2d_2<3>(field, values);
        case 4:
            return make_dirichlet_bc_vector_2d_2<4>(field, values);
        default:
            throw std::runtime_error("Dirichlet BC order must be between 1 and 4, got " + std::to_string(order));
    }
}

// Helper to attach Dirichlet BC for VectorField3D_3 with list of values
template <std::size_t order>
void make_dirichlet_bc_vector_3d_3(VectorField3D_3& field, const std::vector<double>& values)
{
    if (values.size() != 3)
    {
        throw std::runtime_error("Expected 3 values for VectorField3D_3, got " + std::to_string(values.size()));
    }
    using DirichletOrder = samurai::Dirichlet<order>;
    samurai::make_bc<DirichletOrder>(field, values[0], values[1], values[2]);
}

// Wrapper function to dispatch based on order parameter
void make_dirichlet_bc_vector_3d_3_dispatch(VectorField3D_3& field, const std::vector<double>& values, std::size_t order)
{
    switch (order)
    {
        case 1:
            return make_dirichlet_bc_vector_3d_3<1>(field, values);
        case 2:
            return make_dirichlet_bc_vector_3d_3<2>(field, values);
        case 3:
            return make_dirichlet_bc_vector_3d_3<3>(field, values);
        case 4:
            return make_dirichlet_bc_vector_3d_3<4>(field, values);
        default:
            throw std::runtime_error("Dirichlet BC order must be between 1 and 4, got " + std::to_string(order));
    }
}

// Module initialization function for BC bindings
void init_bc_bindings(py::module_& m)
{
    // ============================================================
    // Bind make_dirichlet_bc function for each dimension
    // ============================================================

    // 1D version
    m.def("make_dirichlet_bc",
          &make_dirichlet_bc_1d,
          py::arg("field"),
          py::arg("value"),
          py::arg("order") = 1,
          "Create and attach Dirichlet boundary condition to a 1D scalar field.\n\n"
          "Args:\n"
          "    field: ScalarField1D to apply BC to\n"
          "    value: Constant boundary value\n"
          "    order: Approximation order (1-4, default=1)\n\n"
          "Note:\n"
          "    The BC is attached to the field automatically. No return value.");

    // 2D version
    m.def("make_dirichlet_bc",
          &make_dirichlet_bc_2d,
          py::arg("field"),
          py::arg("value"),
          py::arg("order") = 1,
          "Create and attach Dirichlet boundary condition to a 2D scalar field.\n\n"
          "Args:\n"
          "    field: ScalarField2D to apply BC to\n"
          "    value: Constant boundary value\n"
          "    order: Approximation order (1-4, default=1)\n\n"
          "Note:\n"
          "    The BC is attached to the field automatically. No return value.");

    // 3D version
    m.def("make_dirichlet_bc",
          &make_dirichlet_bc_3d,
          py::arg("field"),
          py::arg("value"),
          py::arg("order") = 1,
          "Create and attach Dirichlet boundary condition to a 3D scalar field.\n\n"
          "Args:\n"
          "    field: ScalarField3D to apply BC to\n"
          "    value: Constant boundary value\n"
          "    order: Approximation order (1-4, default=1)\n\n"
          "Note:\n"
          "    The BC is attached to the field automatically. No return value.");

    // ============================================================
    // Bind make_neumann_bc function for each dimension
    // ============================================================

    // 1D version with value
    m.def("make_neumann_bc",
          &make_neumann_bc_1d,
          py::arg("field"),
          py::arg("value"),
          "Create and attach Neumann boundary condition to a 1D scalar field.\n\n"
          "Args:\n"
          "    field: ScalarField1D to apply BC to\n"
          "    value: Constant derivative value (∂u/∂n) at boundary\n\n"
          "Note:\n"
          "    Neumann BC specifies the derivative normal to the boundary.\n"
          "    The BC is attached to the field automatically. No return value.");

    // 1D version with default (zero derivative)
    m.def("make_neumann_bc",
          &make_neumann_bc_1d_default,
          py::arg("field"),
          "Create and attach Neumann boundary condition (zero derivative) to a 1D scalar field.\n\n"
          "Args:\n"
          "    field: ScalarField1D to apply BC to\n\n"
          "Note:\n"
          "    Zero derivative means no-flux boundary condition.\n"
          "    The BC is attached to the field automatically. No return value.");

    // 2D version with value
    m.def("make_neumann_bc",
          &make_neumann_bc_2d,
          py::arg("field"),
          py::arg("value"),
          "Create and attach Neumann boundary condition to a 2D scalar field.\n\n"
          "Args:\n"
          "    field: ScalarField2D to apply BC to\n"
          "    value: Constant derivative value (∂u/∂n) at boundary\n\n"
          "Note:\n"
          "    Neumann BC specifies the derivative normal to the boundary.\n"
          "    The BC is attached to the field automatically. No return value.");

    // 2D version with default (zero derivative)
    m.def("make_neumann_bc",
          &make_neumann_bc_2d_default,
          py::arg("field"),
          "Create and attach Neumann boundary condition (zero derivative) to a 2D scalar field.\n\n"
          "Args:\n"
          "    field: ScalarField2D to apply BC to\n\n"
          "Note:\n"
          "    Zero derivative means no-flux boundary condition.\n"
          "    The BC is attached to the field automatically. No return value.");

    // 3D version with value
    m.def("make_neumann_bc",
          &make_neumann_bc_3d,
          py::arg("field"),
          py::arg("value"),
          "Create and attach Neumann boundary condition to a 3D scalar field.\n\n"
          "Args:\n"
          "    field: ScalarField3D to apply BC to\n"
          "    value: Constant derivative value (∂u/∂n) at boundary\n\n"
          "Note:\n"
          "    Neumann BC specifies the derivative normal to the boundary.\n"
          "    The BC is attached to the field automatically. No return value.");

    // 3D version with default (zero derivative)
    m.def("make_neumann_bc",
          &make_neumann_bc_3d_default,
          py::arg("field"),
          "Create and attach Neumann boundary condition (zero derivative) to a 3D scalar field.\n\n"
          "Args:\n"
          "    field: ScalarField3D to apply BC to\n\n"
          "Note:\n"
          "    Zero derivative means no-flux boundary condition.\n"
          "    The BC is attached to the field automatically. No return value.");

    // ============================================================
    // Bind make_dirichlet_bc function for VectorField
    // ============================================================

    // VectorField2D_2 version (2D Burgers)
    m.def("make_dirichlet_bc",
          &make_dirichlet_bc_vector_2d_2_dispatch,
          py::arg("field"),
          py::arg("values"),
          py::arg("order") = 1,
          "Create and attach Dirichlet boundary condition to a 2D vector field.\n\n"
          "Args:\n"
          "    field: VectorField2D_2 to apply BC to\n"
          "    values: List of 2 constant boundary values [u, v]\n"
          "    order: Approximation order (1-4, default=1)\n\n"
          "Note:\n"
          "    The BC is attached to the field automatically. No return value.\n\n"
          "Example:\n"
          "    >>> make_dirichlet_bc(u, [0.0, 0.0], order=2)");

    // VectorField3D_3 version (3D Burgers)
    m.def("make_dirichlet_bc",
          &make_dirichlet_bc_vector_3d_3_dispatch,
          py::arg("field"),
          py::arg("values"),
          py::arg("order") = 1,
          "Create and attach Dirichlet boundary condition to a 3D vector field.\n\n"
          "Args:\n"
          "    field: VectorField3D_3 to apply BC to\n"
          "    values: List of 3 constant boundary values [u, v, w]\n"
          "    order: Approximation order (1-4, default=1)\n\n"
          "Note:\n"
          "    The BC is attached to the field automatically. No return value.\n\n"
          "Example:\n"
          "    >>> make_dirichlet_bc(u, [0.0, 0.0, 0.0], order=2)");

    // ============================================================
    // Create boundary submodule for organized API access
    // ============================================================
    py::module_ boundary = m.def_submodule("boundary",
                                           "Boundary condition functions for Samurai AMR simulations\n\n"
                                           "This submodule provides organized access to boundary condition functions.\n"
                                           "Both sam.boundary.dirichlet() and sam.make_dirichlet_bc() reference the same function.\n\n"
                                           "Examples:\n"
                                           "    >>> import samurai_python as sam\n"
                                           "    >>> # New organized API (recommended)\n"
                                           "    >>> sam.boundary.dirichlet(u, 0.0)\n"
                                           "    >>> # Old API (still works)\n"
                                           "    >>> sam.make_dirichlet_bc(u, 0.0)\n");

    // Reference existing BC functions in the submodule
    // Both paths reference the SAME function object
    boundary.attr("dirichlet") = m.attr("make_dirichlet_bc");
    boundary.attr("neumann")   = m.attr("make_neumann_bc");
}
