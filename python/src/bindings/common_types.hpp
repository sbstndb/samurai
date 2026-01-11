// Samurai Python Bindings - Common Type Aliases
//
// Centralized type definitions for all Python bindings to eliminate duplication
// and improve maintainability. All binding files should include this header.
//
// Usage:
//   #include "common_types.hpp"
//   using namespace samurai::python::bindings;

#pragma once

#include <samurai/field.hpp>
#include <samurai/mesh_config.hpp>
#include <samurai/mr/mesh.hpp>

namespace samurai::python::bindings
{

    // ============================================================
    // Default interval type used across bindings
    // ============================================================

    using default_interval = samurai::Interval<double, std::size_t>;

    // ============================================================
    // Mesh configuration aliases (template for any dimension)
    // ============================================================

    template <std::size_t dim>
    using Config = samurai::mesh_config<dim>;

    template <std::size_t dim>
    using CompleteConfig = samurai::complete_mesh_config<Config<dim>, samurai::MRMeshId>;

    template <std::size_t dim>
    using MRMesh = samurai::MRMesh<CompleteConfig<dim>>;

    // ============================================================
    // Convenience aliases for specific dimensions (1D, 2D, 3D)
    // ============================================================

    // 1D
    using Config1D         = Config<1>;
    using CompleteConfig1D = CompleteConfig<1>;
    using Mesh1D           = MRMesh<1>;

    // 2D
    using Config2D         = Config<2>;
    using CompleteConfig2D = CompleteConfig<2>;
    using Mesh2D           = MRMesh<2>;

    // 3D
    using Config3D         = Config<3>;
    using CompleteConfig3D = CompleteConfig<3>;
    using Mesh3D           = MRMesh<3>;

    // ============================================================
    // Field type aliases (template for any dimension)
    // ============================================================

    template <std::size_t dim>
    using ScalarField = samurai::ScalarField<MRMesh<dim>, double>;

    template <std::size_t dim, std::size_t n_comp, bool SOA = false>
    using VectorField = samurai::VectorField<MRMesh<dim>, double, n_comp, SOA>;

    // ============================================================
    // Common VectorField types (n_comp == dim for Burgers, etc.)
    // ============================================================

    // 1D VectorFields
    using VectorField1D_2 = VectorField<1, 2, false>;
    using VectorField1D_3 = VectorField<1, 3, false>;

    // 2D VectorFields
    using VectorField2D_2 = VectorField<2, 2, false>;
    using VectorField2D_3 = VectorField<2, 3, false>;

    // 3D VectorFields
    using VectorField3D_2 = VectorField<3, 2, false>;
    using VectorField3D_3 = VectorField<3, 3, false>;

} // namespace samurai::python::bindings
