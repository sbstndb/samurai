#pragma once
#include "stencil.hpp"
#include "csir.hpp" // Include the new CSIR header

namespace samurai
{
    template <class Mesh, class Vector>
    auto
    boundary_layer(const Mesh& mesh, const typename Mesh::lca_type& domain, std::size_t level, const Vector& direction, std::size_t layer_width)
    {
        using mesh_id_t = typename Mesh::mesh_id_t;
        using lca_t = typename Mesh::lca_type;

        auto& cells = mesh[mesh_id_t::cells][level];
        auto translated_domain = translate(self(domain).on(level), -layer_width * direction);

        // CSIR PROTOTYPE
        auto lhs_csir = csir::to_csir_level(cells);
        auto rhs_lca = lca_t(translated_domain);
        auto rhs_csir = csir::to_csir_level(rhs_lca);
        auto result_csir = csir::difference(lhs_csir, rhs_csir);
        auto result_lca = csir::from_csir_level(result_csir);
        return self(result_lca);
    }

    template <class Mesh, class Vector>
    inline auto domain_boundary_layer(const Mesh& mesh, std::size_t level, const Vector& direction, std::size_t layer_width)
    {
        return boundary_layer(mesh, mesh.domain(), level, direction, layer_width);
    }

    template <class Mesh, class Vector>
    inline auto subdomain_boundary_layer(const Mesh& mesh, std::size_t level, const Vector& direction, std::size_t layer_width)
    {
        return boundary_layer(mesh, mesh.subdomain(), level, direction, layer_width);
    }

    template <class Mesh, class Vector>
    inline auto domain_boundary(const Mesh& mesh, std::size_t level, const Vector& direction)
    {
        return domain_boundary_layer(mesh, level, direction, 1);
    }

    template <class Mesh, class Vector>
    inline auto subdomain_boundary(const Mesh& mesh, std::size_t level, const Vector& direction)
    {
        return subdomain_boundary_layer(mesh, level, direction, 1);
    }

    template <class Mesh>
    inline auto domain_boundary(const Mesh& mesh, std::size_t level)
    {
        using mesh_id_t = typename Mesh::mesh_id_t;
        using lca_t = typename Mesh::lca_type;

        auto& cells = mesh[mesh_id_t::cells][level];
        auto contracted_domain = contract(self(mesh.domain()).on(level), 1);

        // CSIR PROTOTYPE
        auto lhs_csir = csir::to_csir_level(cells);
        auto rhs_lca = lca_t(contracted_domain);
        auto rhs_csir = csir::to_csir_level(rhs_lca);
        auto result_csir = csir::difference(lhs_csir, rhs_csir);
        auto result_lca = csir::from_csir_level(result_csir);
        return self(result_lca);
    }

    template <class Mesh>
    auto domain_boundary_outer_layer(const Mesh& mesh, std::size_t level, std::size_t layer_width)
    {
        using lca_t = typename Mesh::lca_type;
        using lcl_t = typename Mesh::lcl_type;

        auto domain = self(mesh.domain()).on(level);

        lcl_t outer_boundary_lcl(level, mesh.origin_point(), mesh.scaling_factor());

        for_each_cartesian_direction<Mesh::dim>(
            [&](const auto& direction)
            {
                auto inner_boundary = domain_boundary(mesh, level, direction);
                for (std::size_t layer = 1; layer <= layer_width; ++layer)
                {
                    auto outer_layer = difference(translate(inner_boundary, layer * direction), domain);
                    outer_layer(
                        [&](const auto& i, const auto& index)
                        {
                            outer_boundary_lcl[index].add_interval({i});
                        });
                }
            });
        return lca_t(outer_boundary_lcl);
    }

    template <class Mesh>
    auto
    domain_boundary_outer_layer(const Mesh& mesh, std::size_t level, const DirectionVector<Mesh::dim>& direction, std::size_t layer_width)
    {
        using lca_t = typename Mesh::lca_type;
        using lcl_t = typename Mesh::lcl_type;

        auto domain = self(mesh.domain()).on(level);

        lcl_t outer_boundary_lcl(level, mesh.origin_point(), mesh.scaling_factor());

        auto inner_boundary = domain_boundary(mesh, level, direction);
        for (std::size_t layer = 1; layer <= layer_width; ++layer)
        {
            auto outer_layer = difference(translate(inner_boundary, layer * direction), domain);
            outer_layer(
                [&](const auto& i, const auto& index)
                {
                    outer_boundary_lcl[index].add_interval({i});
                });
        }
        return lca_t(outer_boundary_lcl);
    }

    template <class Mesh, class Subset, std::size_t stencil_size, class Equation, std::size_t nb_equations, class Func>
    void for_each_stencil_on_boundary(const Mesh& mesh,
                                      const Subset& boundary_region,
                                      const StencilAnalyzer<stencil_size, Mesh::dim>& stencil,
                                      const std::array<Equation, nb_equations>& equations,
                                      Func&& func)
    {        
        using mesh_id_t         = typename Mesh::mesh_id_t;
        using equation_coeffs_t = typename Equation::equation_coeffs_t;
        using lca_t = typename Mesh::lca_type;

        for_each_level(mesh,
                       [&](std::size_t level)
                       {
                           // CSIR PROTOTYPE IMPLEMENTATION
                           // 1. Convert inputs to LevelCellArray
                           auto lhs_lca = mesh[mesh_id_t::cells][level];
                           auto rhs_lca = lca_t(boundary_region.on(level));

                           // 2. Convert LevelCellArray to CSIR_Level
                           auto lhs_csir = csir::to_csir_level(lhs_lca);
                           auto rhs_csir = csir::to_csir_level(rhs_lca);

                           // 3. Perform CSIR intersection
                           auto result_csir = csir::intersection(lhs_csir, rhs_csir);

                           // 4. Convert result back to LevelCellArray and wrap it for the rest of the code
                           auto bdry_lca = csir::from_csir_level(result_csir);
                           auto bdry = self(bdry_lca);

                           std::array<equation_coeffs_t, nb_equations> equations_coeffs;
                           for (std::size_t i = 0; i < nb_equations; ++i)
                           {
                               equations_coeffs[i].ghost_index    = equations[i].ghost_index;
                               equations_coeffs[i].stencil_coeffs = equations[i].get_stencil_coeffs(mesh.cell_length(level));
                               equations_coeffs[i].rhs_coeffs     = equations[i].get_rhs_coeffs(mesh.cell_length(level));
                           }
                           for_each_stencil(mesh,
                                            bdry,
                                            stencil,
                                            [&](auto& cells)
                                            {
                                                func(cells, equations_coeffs);
                                            });
                       });
    }
}
