// Samurai Python Bindings - Boundary Conditions
//
// Bindings for make_bc and boundary condition types

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <samurai/bc/bc.hpp>
#include <samurai/bc/dirichlet.hpp>
#include <samurai/bc/neumann.hpp>
#include <samurai/bc/polynomial_extrapolation.hpp>
#include <samurai/field.hpp>
#include <samurai/mesh_config.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/stencil.hpp>

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

// ============================================================
// Polynomial extrapolation boundary condition bindings
// ============================================================

// Helper to attach polynomial extrapolation BC for a specific dimension and stencil size
// The order parameter is actually stencil_size / 2 (e.g., order=1 -> stencil_size=2, order=2 -> stencil_size=4)
// Note: PolynomialExtrapolation doesn't use the bcvalue parameter (it extrapolates from existing cells)
template <std::size_t dim>
void make_polynomial_extrapolation_bc_dispatch(ScalarField<dim>& field, std::size_t order)
{
    // Convert Python "order" to stencil_size: order 1 -> 2, order 2 -> 4, order 3 -> 6
    std::size_t stencil_size = order * 2;
    if (stencil_size < 2 || stencil_size > 6 || stencil_size % 2 != 0)
    {
        throw std::runtime_error("Polynomial extrapolation order must be 1, 2, or 3 (stencil sizes 2, 4, or 6), got "
                                 + std::to_string(order));
    }

    // Get the mesh domain directly
    // For MRMesh, mesh.domain() returns the LevelCellArray domain
    const auto& domain = samurai::detail::get_mesh(field.mesh());

    // Create and attach the appropriate polynomial extrapolation BC
    // PolynomialExtrapolation takes Field and stencil_size as template parameters
    switch (stencil_size)
    {
        case 2:
        {
            using PE = samurai::PolynomialExtrapolation<ScalarField<dim>, 2>;
            samurai::ConstantBc<ScalarField<dim>> dummy_bcv;
            field.attach_bc(PE(domain, dummy_bcv, true));
            break;
        }
        case 4:
        {
            using PE = samurai::PolynomialExtrapolation<ScalarField<dim>, 4>;
            samurai::ConstantBc<ScalarField<dim>> dummy_bcv;
            field.attach_bc(PE(domain, dummy_bcv, true));
            break;
        }
        case 6:
        {
            using PE = samurai::PolynomialExtrapolation<ScalarField<dim>, 6>;
            samurai::ConstantBc<ScalarField<dim>> dummy_bcv;
            field.attach_bc(PE(domain, dummy_bcv, true));
            break;
        }
        default:
            throw std::runtime_error("Unsupported stencil size for polynomial extrapolation: " + std::to_string(stencil_size));
    }
}

// 1D polynomial extrapolation wrapper
void make_polynomial_extrapolation_bc_1d(ScalarField<1>& field, std::size_t order)
{
    make_polynomial_extrapolation_bc_dispatch<1>(field, order);
}

// 2D polynomial extrapolation wrapper
void make_polynomial_extrapolation_bc_2d(ScalarField<2>& field, std::size_t order)
{
    make_polynomial_extrapolation_bc_dispatch<2>(field, order);
}

// 3D polynomial extrapolation wrapper
void make_polynomial_extrapolation_bc_3d(ScalarField<3>& field, std::size_t order)
{
    make_polynomial_extrapolation_bc_dispatch<3>(field, order);
}

// ============================================================
// Direction-specific boundary condition wrappers
// ============================================================

// Helper class to store BC configuration before applying to specific directions
template <std::size_t dim>
class DirectionalBCWrapper
{
  public:
    using Field = ScalarField<dim>;
    using direction_t = samurai::DirectionVector<dim>;

    DirectionalBCWrapper(Field& field, double value, bool is_dirichlet)
        : m_field(field)
        , m_value(value)
        , m_is_dirichlet(is_dirichlet)
    {
    }

    void on(const std::vector<direction_t>& directions)
    {
        for (const auto& dir : directions)
        {
            if (m_is_dirichlet)
            {
                // Apply Dirichlet BC with default order 1 (can be extended)
                // make_bc returns a pointer, so we need to use ->on()
                samurai::make_bc<samurai::Dirichlet<1>>(m_field, m_value)->on(dir);
            }
            else
            {
                // Apply Neumann BC
                samurai::make_bc<samurai::Neumann<1>>(m_field, m_value)->on(dir);
            }
        }
    }

  private:
    Field& m_field;
    double m_value;
    bool m_is_dirichlet;
};

// Factory functions for directional BC wrappers
std::shared_ptr<DirectionalBCWrapper<1>> make_dirichlet_directional_1d(ScalarField<1>& field, double value)
{
    return std::make_shared<DirectionalBCWrapper<1>>(field, value, true);
}

std::shared_ptr<DirectionalBCWrapper<2>> make_dirichlet_directional_2d(ScalarField<2>& field, double value)
{
    return std::make_shared<DirectionalBCWrapper<2>>(field, value, true);
}

std::shared_ptr<DirectionalBCWrapper<3>> make_dirichlet_directional_3d(ScalarField<3>& field, double value)
{
    return std::make_shared<DirectionalBCWrapper<3>>(field, value, true);
}

std::shared_ptr<DirectionalBCWrapper<1>> make_neumann_directional_1d(ScalarField<1>& field, double value)
{
    return std::make_shared<DirectionalBCWrapper<1>>(field, value, false);
}

std::shared_ptr<DirectionalBCWrapper<2>> make_neumann_directional_2d(ScalarField<2>& field, double value)
{
    return std::make_shared<DirectionalBCWrapper<2>>(field, value, false);
}

std::shared_ptr<DirectionalBCWrapper<3>> make_neumann_directional_3d(ScalarField<3>& field, double value)
{
    return std::make_shared<DirectionalBCWrapper<3>>(field, value, false);
}

// ============================================================
// Module initialization function for BC bindings
// ============================================================
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
    // Bind make_polynomial_extrapolation_bc function for each dimension
    // ============================================================

    // 1D version
    m.def("make_polynomial_extrapolation_bc",
          &make_polynomial_extrapolation_bc_1d,
          py::arg("field"),
          py::arg("order") = 2,
          "Create and attach polynomial extrapolation boundary condition to a 1D scalar field.\n\n"
          "Args:\n"
          "    field: ScalarField1D to apply BC to\n"
          "    order: Extrapolation order (1-3, default=2)\n"
          "           order=1: 2-point stencil (constant)\n"
          "           order=2: 4-point stencil (linear)\n"
          "           order=3: 6-point stencil (quadratic)\n\n"
          "Note:\n"
          "    Polynomial extrapolation is used for outer ghost cells.\n"
          "    The BC is attached to the field automatically. No return value.");

    // 2D version
    m.def("make_polynomial_extrapolation_bc",
          &make_polynomial_extrapolation_bc_2d,
          py::arg("field"),
          py::arg("order") = 2,
          "Create and attach polynomial extrapolation boundary condition to a 2D scalar field.\n\n"
          "Args:\n"
          "    field: ScalarField2D to apply BC to\n"
          "    order: Extrapolation order (1-3, default=2)\n"
          "           order=1: 2-point stencil (constant)\n"
          "           order=2: 4-point stencil (linear)\n"
          "           order=3: 6-point stencil (quadratic)\n\n"
          "Note:\n"
          "    Polynomial extrapolation is used for outer ghost cells.\n"
          "    The BC is attached to the field automatically. No return value.");

    // 3D version
    m.def("make_polynomial_extrapolation_bc",
          &make_polynomial_extrapolation_bc_3d,
          py::arg("field"),
          py::arg("order") = 2,
          "Create and attach polynomial extrapolation boundary condition to a 3D scalar field.\n\n"
          "Args:\n"
          "    field: ScalarField3D to apply BC to\n"
          "    order: Extrapolation order (1-3, default=2)\n"
          "           order=1: 2-point stencil (constant)\n"
          "           order=2: 4-point stencil (linear)\n"
          "           order=3: 6-point stencil (quadratic)\n\n"
          "Note:\n"
          "    Polynomial extrapolation is used for outer ghost cells.\n"
          "    The BC is attached to the field automatically. No return value.");

    // ============================================================
    // Bind DirectionalBCWrapper class for direction-specific BCs
    // ============================================================

    // 1D DirectionalBCWrapper bindings
    py::class_<DirectionalBCWrapper<1>, std::shared_ptr<DirectionalBCWrapper<1>>>(m, "DirectionalBCWrapper1D")
        .def("on",
             [](DirectionalBCWrapper<1>& self, const std::vector<samurai::DirectionVector<1>>& directions)
             {
                 self.on(directions);
             },
             py::arg("directions"),
             "Apply boundary condition to specific directions.\n\n"
             "Args:\n"
             "    directions: List of direction vectors\n\n"
             "Example:\n"
             "    >>> bc.on(sam.direction.LEFT, sam.direction.RIGHT)");

    // 2D DirectionalBCWrapper bindings
    py::class_<DirectionalBCWrapper<2>, std::shared_ptr<DirectionalBCWrapper<2>>>(m, "DirectionalBCWrapper2D")
        .def("on",
             [](DirectionalBCWrapper<2>& self, const std::vector<samurai::DirectionVector<2>>& directions)
             {
                 self.on(directions);
             },
             py::arg("directions"),
             "Apply boundary condition to specific directions.\n\n"
             "Args:\n"
             "    directions: List of direction vectors\n\n"
             "Example:\n"
             "    >>> bc.on(sam.direction.LEFT, sam.direction.RIGHT)");

    // 3D DirectionalBCWrapper bindings
    py::class_<DirectionalBCWrapper<3>, std::shared_ptr<DirectionalBCWrapper<3>>>(m, "DirectionalBCWrapper3D")
        .def("on",
             [](DirectionalBCWrapper<3>& self, const std::vector<samurai::DirectionVector<3>>& directions)
             {
                 self.on(directions);
             },
             py::arg("directions"),
             "Apply boundary condition to specific directions.\n\n"
             "Args:\n"
             "    directions: List of direction vectors\n\n"
             "Example:\n"
             "    >>> bc.on(sam.direction.LEFT, sam.direction.RIGHT)");

    // ============================================================
    // Bind directional BC factory functions
    // ============================================================

    // 1D directional Dirichlet
    m.def("make_dirichlet_bc_directional",
          &make_dirichlet_directional_1d,
          py::arg("field"),
          py::arg("value"),
          "Create a directional Dirichlet boundary condition wrapper for 1D field.\n\n"
          "Returns a wrapper that can be applied to specific directions using .on()\n\n"
          "Example:\n"
          "    >>> bc = sam.make_dirichlet_bc_directional(u, 0.0)\n"
          "    >>> bc.on(sam.direction.LEFT, sam.direction.RIGHT)");

    // 2D directional Dirichlet
    m.def("make_dirichlet_bc_directional",
          &make_dirichlet_directional_2d,
          py::arg("field"),
          py::arg("value"),
          "Create a directional Dirichlet boundary condition wrapper for 2D field.\n\n"
          "Returns a wrapper that can be applied to specific directions using .on()\n\n"
          "Example:\n"
          "    >>> bc = sam.make_dirichlet_bc_directional(u, 0.0)\n"
          "    >>> bc.on(sam.direction.LEFT, sam.direction.RIGHT, sam.direction.TOP)");

    // 3D directional Dirichlet
    m.def("make_dirichlet_bc_directional",
          &make_dirichlet_directional_3d,
          py::arg("field"),
          py::arg("value"),
          "Create a directional Dirichlet boundary condition wrapper for 3D field.\n\n"
          "Returns a wrapper that can be applied to specific directions using .on()\n\n"
          "Example:\n"
          "    >>> bc = sam.make_dirichlet_bc_directional(u, 0.0)\n"
          "    >>> bc.on(sam.direction.LEFT, sam.direction.RIGHT, sam.direction.TOP)");

    // 1D directional Neumann
    m.def("make_neumann_bc_directional",
          &make_neumann_directional_1d,
          py::arg("field"),
          py::arg("value"),
          "Create a directional Neumann boundary condition wrapper for 1D field.\n\n"
          "Returns a wrapper that can be applied to specific directions using .on()\n\n"
          "Example:\n"
          "    >>> bc = sam.make_neumann_bc_directional(u, 0.0)\n"
          "    >>> bc.on(sam.direction.LEFT)");

    // 2D directional Neumann
    m.def("make_neumann_bc_directional",
          &make_neumann_directional_2d,
          py::arg("field"),
          py::arg("value"),
          "Create a directional Neumann boundary condition wrapper for 2D field.\n\n"
          "Returns a wrapper that can be applied to specific directions using .on()\n\n"
          "Example:\n"
          "    >>> bc = sam.make_neumann_bc_directional(u, 0.0)\n"
          "    >>> bc.on(sam.direction.LEFT, sam.direction.RIGHT)");

    // 3D directional Neumann
    m.def("make_neumann_bc_directional",
          &make_neumann_directional_3d,
          py::arg("field"),
          py::arg("value"),
          "Create a directional Neumann boundary condition wrapper for 3D field.\n\n"
          "Returns a wrapper that can be applied to specific directions using .on()\n\n"
          "Example:\n"
          "    >>> bc = sam.make_neumann_bc_directional(u, 0.0)\n"
          "    >>> bc.on(sam.direction.LEFT, sam.direction.RIGHT)");

    // ============================================================
    // Create direction submodule for direction vectors
    // NOTE: Direction vectors are temporarily disabled because they require
    // xtensor-python type registration which is not available in the build.
    // This can be enabled later by adding proper type bindings.
    // ============================================================
    // py::module_ direction = m.def_submodule("direction",
    //                                         "Direction vectors for boundary conditions\n\n"
    //                                         "Provides predefined direction vectors for applying BCs to specific boundaries.\n\n"
    //                                         "Examples:\n"
    //                                         "    >>> import samurai_python as sam\n"
    //                                         "    >>> bc = sam.make_dirichlet_bc_directional(u, 0.0)\n"
    //                                         "    >>> bc.on(sam.direction.LEFT, sam.direction.RIGHT)\n"
    //                                         "    >>> bc = sam.make_neumann_bc_directional(u, 0.0)\n"
    //                                         "    >>> bc.on(sam.direction.TOP)");
    //
    // // 1D direction vectors
    // direction.attr("LEFT_1D")  = samurai::DirectionVector<1>({ -1 });
    // direction.attr("RIGHT_1D") = samurai::DirectionVector<1>({ 1 });
    //
    // // 2D direction vectors
    // direction.attr("LEFT_2D")   = samurai::DirectionVector<2>({ -1, 0 });
    // direction.attr("RIGHT_2D")  = samurai::DirectionVector<2>({ 1, 0 });
    // direction.attr("BOTTOM")    = samurai::DirectionVector<2>({ 0, -1 });
    // direction.attr("TOP")       = samurai::DirectionVector<2>({ 0, 1 });
    //
    // // 3D direction vectors
    // direction.attr("LEFT_3D")   = samurai::DirectionVector<3>({ -1, 0, 0 });
    // direction.attr("RIGHT_3D")  = samurai::DirectionVector<3>({ 1, 0, 0 });
    // direction.attr("FRONT")     = samurai::DirectionVector<3>({ 0, -1, 0 });
    // direction.attr("BACK")      = samurai::DirectionVector<3>({ 0, 1, 0 });
    // direction.attr("BOTTOM_3D") = samurai::DirectionVector<3>({ 0, 0, -1 });
    // direction.attr("TOP_3D")    = samurai::DirectionVector<3>({ 0, 0, 1 });
    //
    // // Also expose dimension-agnostic aliases (most common use case for 2D)
    // direction.attr("LEFT")   = samurai::DirectionVector<2>({ -1, 0 });
    // direction.attr("RIGHT")  = samurai::DirectionVector<2>({ 1, 0 });
    // direction.attr("BOTTOM") = samurai::DirectionVector<2>({ 0, -1 });
    // direction.attr("TOP")    = samurai::DirectionVector<2>({ 0, 1 });

    // ============================================================
    // Create boundary submodule for organized API access
    // ============================================================
    py::module_ boundary = m.def_submodule("boundary",
                                           "Boundary condition functions for Samurai AMR simulations\n\n"
                                           "This submodule provides organized access to boundary condition functions.\n"
                                           "Both sam.boundary.dirichlet() and sam.make_dirichlet_bc() reference the same function.\n\n"
                                           "Examples:\n"
                                           "    >>> import samurai_python as sam\n"
                                           "    >>> # Constant BC everywhere\n"
                                           "    >>> sam.boundary.dirichlet(u, 0.0)\n"
                                           "    >>> # Polynomial extrapolation\n"
                                           "    >>> sam.boundary.polynomial_extrapolation(u, order=2)");

    // Reference existing BC functions in the submodule
    // Both paths reference the SAME function object
    boundary.attr("dirichlet") = m.attr("make_dirichlet_bc");
    boundary.attr("neumann")   = m.attr("make_neumann_bc");
    boundary.attr("polynomial_extrapolation") = m.attr("make_polynomial_extrapolation_bc");
}
