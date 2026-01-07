// Samurai Python Bindings - Multiresolution Adaptation
//
// Bindings for make_MRAdapt and update_ghost_mr functions

#include <memory>
#include <pybind11/pybind11.h>
#include <samurai/algorithm/update.hpp>
#include <samurai/field.hpp>
#include <samurai/mesh_config.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>

namespace py = pybind11;

// ============================================================
// Type aliases matching field_bindings.cpp pattern
// ============================================================

using default_interval = samurai::Interval<double, std::size_t>;

template <std::size_t dim>
using MRMesh = samurai::MRMesh<samurai::complete_mesh_config<samurai::mesh_config<dim>, samurai::MRMeshId>>;

template <std::size_t dim>
using ScalarField = samurai::ScalarField<MRMesh<dim>, double>;

template <std::size_t dim, std::size_t n_comp, bool SOA = false>
using VectorField = samurai::VectorField<MRMesh<dim>, double, n_comp, SOA>;

// Specific VectorField types for Burgers equation (n_comp == dim)
using VectorField2D_2 = VectorField<2, 2, false>;
using VectorField3D_3 = VectorField<3, 3, false>;

// ============================================================
// Python-callable wrapper for Adapt objects
// ============================================================

// Base class for type erasure
class PyAdaptBase
{
  public:

    virtual ~PyAdaptBase()                                = default;
    virtual void call(samurai::mra_config& config)        = 0;
    virtual void call_with_velocity_2d(samurai::mra_config& config, VectorField2D_2& velocity) = 0;
    virtual void call_with_velocity_3d(samurai::mra_config& config, VectorField3D_3& velocity) = 0;
};

// 1D Adapt wrapper (no velocity support needed)
template <class AdaptType>
class PyAdaptImpl1D : public PyAdaptBase
{
  public:

    explicit PyAdaptImpl1D(AdaptType&& adapt)
        : m_adapt(std::move(adapt))
    {
    }

    void call(samurai::mra_config& config) override
    {
        m_adapt(config);
    }

    void call_with_velocity_2d(samurai::mra_config& config, VectorField2D_2& velocity) override
    {
        throw std::runtime_error("Cannot call 1D Adapt with 2D velocity field");
    }

    void call_with_velocity_3d(samurai::mra_config& config, VectorField3D_3& velocity) override
    {
        throw std::runtime_error("Cannot call 1D Adapt with 3D velocity field");
    }

  private:

    AdaptType m_adapt;
};

// 2D Adapt wrapper (with 2D velocity support)
template <class AdaptType>
class PyAdaptImpl2D : public PyAdaptBase
{
  public:

    explicit PyAdaptImpl2D(AdaptType&& adapt)
        : m_adapt(std::move(adapt))
    {
    }

    void call(samurai::mra_config& config) override
    {
        m_adapt(config);
    }

    void call_with_velocity_2d(samurai::mra_config& config, VectorField2D_2& velocity) override
    {
        m_adapt(config, velocity);
    }

    void call_with_velocity_3d(samurai::mra_config& config, VectorField3D_3& velocity) override
    {
        throw std::runtime_error("Cannot call 2D Adapt with 3D velocity field");
    }

  private:

    AdaptType m_adapt;
};

// 3D Adapt wrapper (with 3D velocity support)
template <class AdaptType>
class PyAdaptImpl3D : public PyAdaptBase
{
  public:

    explicit PyAdaptImpl3D(AdaptType&& adapt)
        : m_adapt(std::move(adapt))
    {
    }

    void call(samurai::mra_config& config) override
    {
        m_adapt(config);
    }

    void call_with_velocity_2d(samurai::mra_config& config, VectorField2D_2& velocity) override
    {
        throw std::runtime_error("Cannot call 3D Adapt with 2D velocity field");
    }

    void call_with_velocity_3d(samurai::mra_config& config, VectorField3D_3& velocity) override
    {
        m_adapt(config, velocity);
    }

  private:

    AdaptType m_adapt;
};

// Wrapper class that can be created with or without velocity support
class PyAdaptVariant
{
  public:

    // 1D constructor
    template <class AdaptType>
    static PyAdaptVariant make_1d(AdaptType&& adapt)
    {
        PyAdaptVariant result;
        result.m_impl = std::make_unique<PyAdaptImpl1D<AdaptType>>(std::move(adapt));
        return result;
    }

    // 2D constructor
    template <class AdaptType>
    static PyAdaptVariant make_2d(AdaptType&& adapt)
    {
        PyAdaptVariant result;
        result.m_impl = std::make_unique<PyAdaptImpl2D<AdaptType>>(std::move(adapt));
        return result;
    }

    // 3D constructor
    template <class AdaptType>
    static PyAdaptVariant make_3d(AdaptType&& adapt)
    {
        PyAdaptVariant result;
        result.m_impl = std::make_unique<PyAdaptImpl3D<AdaptType>>(std::move(adapt));
        return result;
    }

    void call(samurai::mra_config& config)
    {
        m_impl->call(config);
    }

    void call_with_velocity_2d(samurai::mra_config& config, VectorField2D_2& velocity)
    {
        m_impl->call_with_velocity_2d(config, velocity);
    }

    void call_with_velocity_3d(samurai::mra_config& config, VectorField3D_3& velocity)
    {
        m_impl->call_with_velocity_3d(config, velocity);
    }

  private:

    std::unique_ptr<PyAdaptBase> m_impl;

    // Private default constructor
    PyAdaptVariant() = default;
};

// ============================================================
// Dimension-specific factory functions
// Following pattern from operator_bindings.cpp (upwind_1d, upwind_2d, upwind_3d)
// ============================================================

// 1D update_ghost_mr wrapper
void update_ghost_mr_1d(ScalarField<1>& field)
{
    samurai::update_ghost_mr(field);
}

// 2D update_ghost_mr wrapper
void update_ghost_mr_2d(ScalarField<2>& field)
{
    samurai::update_ghost_mr(field);
}

// 3D update_ghost_mr wrapper
void update_ghost_mr_3d(ScalarField<3>& field)
{
    samurai::update_ghost_mr(field);
}

// 1D make_MRAdapt wrapper
PyAdaptVariant make_mr_adapt_1d(ScalarField<1>& field)
{
    auto adapt_obj = samurai::make_MRAdapt(field);
    return PyAdaptVariant::make_1d(std::move(adapt_obj));
}

// 2D make_MRAdapt wrapper
PyAdaptVariant make_mr_adapt_2d(ScalarField<2>& field)
{
    auto adapt_obj = samurai::make_MRAdapt(field);
    return PyAdaptVariant::make_2d(std::move(adapt_obj));
}

// 3D make_MRAdapt wrapper
PyAdaptVariant make_mr_adapt_3d(ScalarField<3>& field)
{
    auto adapt_obj = samurai::make_MRAdapt(field);
    return PyAdaptVariant::make_3d(std::move(adapt_obj));
}

// ============================================================
// VectorField wrappers for make_MRAdapt
// ============================================================

// 2D VectorField (VectorField2D_2) make_MRAdapt wrapper
PyAdaptVariant make_mr_adapt_vector_2d_2(VectorField2D_2& field)
{
    auto adapt_obj = samurai::make_MRAdapt(field);
    return PyAdaptVariant::make_2d(std::move(adapt_obj));
}

// 3D VectorField (VectorField3D_3) make_MRAdapt wrapper
PyAdaptVariant make_mr_adapt_vector_3d_3(VectorField3D_3& field)
{
    auto adapt_obj = samurai::make_MRAdapt(field);
    return PyAdaptVariant::make_3d(std::move(adapt_obj));
}

// ============================================================
// VectorField wrappers for update_ghost_mr
// ============================================================

// 2D VectorField (VectorField2D_2) update_ghost_mr wrapper
void update_ghost_mr_vector_2d_2(VectorField2D_2& field)
{
    samurai::update_ghost_mr(field);
}

// 3D VectorField (VectorField3D_3) update_ghost_mr wrapper
void update_ghost_mr_vector_3d_3(VectorField3D_3& field)
{
    samurai::update_ghost_mr(field);
}

// ============================================================
// Module initialization
// ============================================================

void init_adapt_bindings(py::module_& m)
{
    // Bind Adapt wrapper class
    py::class_<PyAdaptVariant>(m, "MRAdapt", R"pbdoc(
        Multiresolution mesh adaptation callable.

        Created by make_MRAdapt(), this object performs adaptive mesh refinement
        based on the Harten multiresolution analysis algorithm.

        Examples
        --------
        >>> import samurai_python as sam
        >>> config = sam.MRAConfig()
        >>> config.epsilon = 2e-4
        >>> config.regularity = 2.0
        >>> MRadaptation = sam.make_MRAdapt(field)
        >>> MRadaptation(config)  # Perform adaptation

        For domains with obstacles, pass velocity field during call:
        >>> MRadaptation = sam.make_MRAdapt(field)  # Create with scalar field only
        >>> MRadaptation(config, velocity)  # Pass velocity during call for obstacle BC

        Notes
        -----
        Create the adaptation object once and reuse it throughout your simulation.
        The same configuration can also be reused across multiple adaptation calls.
        When using DomainBuilder with obstacles, pass velocity field to handle BC correctly.
    )pbdoc")
        .def(
            "__call__",
            [](PyAdaptVariant& self, samurai::mra_config& config)
            {
                self.call(config);
            },
            py::arg("config"),
            "Perform mesh adaptation with the given configuration.")
        .def(
            "__call__",
            [](PyAdaptVariant& self, samurai::mra_config& config, VectorField2D_2& velocity)
            {
                self.call_with_velocity_2d(config, velocity);
            },
            py::arg("config"),
            py::arg("velocity"),
            "Perform mesh adaptation with 2D velocity field (required for obstacle BC).")
        .def(
            "__call__",
            [](PyAdaptVariant& self, samurai::mra_config& config, VectorField3D_3& velocity)
            {
                self.call_with_velocity_3d(config, velocity);
            },
            py::arg("config"),
            py::arg("velocity"),
            "Perform mesh adaptation with 3D velocity field (required for obstacle BC).");

    // Bind update_ghost_mr for all dimensions
    // Following pattern from operator_bindings.cpp where multiple functions
    // are bound with the same Python name
    m.def("update_ghost_mr", &update_ghost_mr_1d, py::arg("field"), "Update ghost cells for multiresolution analysis (1D)");

    m.def("update_ghost_mr", &update_ghost_mr_2d, py::arg("field"), "Update ghost cells for multiresolution analysis (2D)");

    m.def("update_ghost_mr", &update_ghost_mr_3d, py::arg("field"), "Update ghost cells for multiresolution analysis (3D)");

    // Bind update_ghost_mr for VectorField types
    m.def("update_ghost_mr", &update_ghost_mr_vector_2d_2, py::arg("field"), "Update ghost cells for 2D vector field (2 components)");

    m.def("update_ghost_mr", &update_ghost_mr_vector_3d_3, py::arg("field"), "Update ghost cells for 3D vector field (3 components)");

    // Bind make_MRAdapt for all dimensions
    m.def("make_MRAdapt", &make_mr_adapt_1d, py::arg("field"), "Create multiresolution adaptation object (1D)");

    m.def("make_MRAdapt", &make_mr_adapt_2d, py::arg("field"), "Create multiresolution adaptation object (2D)");

    m.def("make_MRAdapt", &make_mr_adapt_3d, py::arg("field"), "Create multiresolution adaptation object (3D)");

    // Bind make_MRAdapt for VectorField types
    m.def("make_MRAdapt", &make_mr_adapt_vector_2d_2, py::arg("field"), "Create multiresolution adaptation object for 2D vector field (2 components)");

    m.def("make_MRAdapt", &make_mr_adapt_vector_3d_3, py::arg("field"), "Create multiresolution adaptation object for 3D vector field (3 components)");

    // ============================================================
    // Create adaptation submodule for organized API access
    // ============================================================
    py::module_ adapt = m.def_submodule("adaptation",
        "Mesh adaptation functions for Samurai AMR simulations\n\n"
        "This submodule provides organized access to AMR/MR adaptation functionality.\n"
        "Both sam.adaptation.MRAdapt and sam.MRAdapt reference the same class.\n\n"
        "Examples:\n"
        "    >>> import samurai_python as sam\n"
        "    >>> # New organized API (recommended)\n"
        "    >>> MRadapt = sam.adaptation.MRAdapt(u)\n"
        "    >>> # Old API (still works)\n"
        "    >>> MRadapt = sam.MRAdapt(u)\n");

    // Reference existing adaptation classes/functions
    adapt.attr("MRAdapt") = m.attr("MRAdapt");
    adapt.attr("make_MRAdapt") = m.attr("make_MRAdapt");
    adapt.attr("update_ghost_mr") = m.attr("update_ghost_mr");
}
