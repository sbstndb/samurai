#pragma once
#include "stencil.hpp"
#include "csir_unified/src/csir.hpp" // CSIR unified API

namespace samurai
{
    template <class Mesh, class Vector>
    auto
    boundary_layer(const Mesh& mesh, const typename Mesh::lca_type& domain, std::size_t level, const Vector& direction, std::size_t layer_width)
    {
        using mesh_id_t = typename Mesh::mesh_id_t;

        auto& cells = mesh[mesh_id_t::cells][level];
        // CSIR: cells[level] \ translate(domain[level], layer_width * direction)
        auto dom_lvl   = csir::to_csir_level(domain);
        auto dom_on    = csir::project_to_level(dom_lvl, level);
        // Preserve original sign: translate by -layer_width * direction
        std::array<int, Mesh::dim> d{}; for (std::size_t k=0;k<Mesh::dim;++k) d[k] = -direction[k] * static_cast<int>(layer_width);
        auto trans     = csir::translate(dom_on, d);
        auto cells_csir= csir::to_csir_level(cells);
        auto result_csir = csir::difference(cells_csir, trans);
        auto result_lca = csir::from_csir_level(result_csir, mesh.origin_point(), mesh.scaling_factor());
        return result_lca;
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

        auto& cells = mesh[mesh_id_t::cells][level];
        // CSIR: cells[level] \ contract(domain[level], 1)
        auto dom_lvl   = csir::to_csir_level(mesh.domain());
        auto dom_on    = csir::project_to_level(dom_lvl, level);
        std::array<bool, Mesh::dim> mask; mask.fill(true);
        auto contr     = csir::contract(dom_on, 1, mask);
        auto cells_csir= csir::to_csir_level(cells);
        auto result_csir = csir::difference(cells_csir, contr);
        auto result_lca = csir::from_csir_level(result_csir, mesh.origin_point(), mesh.scaling_factor());
        return result_lca;
    }

    template <class Mesh>
    auto domain_boundary_outer_layer(const Mesh& mesh, std::size_t level, std::size_t layer_width)
    {
        using mesh_id_t = typename Mesh::mesh_id_t;
        using lca_t = typename Mesh::lca_type;
        using lcl_t = typename Mesh::lcl_type;

        // Materialize domain[level]
        auto dom_csir   = csir::to_csir_level(mesh.domain());
        auto domain_on  = csir::project_to_level(dom_csir, level);
        auto domain_lca = csir::from_csir_level(domain_on, mesh.origin_point(), mesh.scaling_factor());
        auto domain     = self(domain_lca);

        lcl_t outer_boundary_lcl(level, mesh.origin_point(), mesh.scaling_factor());

        for_each_cartesian_direction<Mesh::dim>(
            [&](const auto& direction)
            {
                // Build inner boundary for this direction using CSIR and keep LCA alive
                auto cells_csir = csir::to_csir_level(mesh[mesh_id_t::cells][level]);
                std::array<int, Mesh::dim> d_in{}; for (std::size_t k=0;k<Mesh::dim;++k) d_in[k] = -direction[k];
                auto dom_shift  = csir::translate(domain_on, d_in);
                auto inner_csir = csir::difference(cells_csir, dom_shift);
                auto inner_lca  = csir::from_csir_level(inner_csir, mesh.origin_point(), mesh.scaling_factor());
                auto inner_boundary = self(inner_lca);

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
        using mesh_id_t = typename Mesh::mesh_id_t;
        using lca_t = typename Mesh::lca_type;
        using lcl_t = typename Mesh::lcl_type;

        auto dom_csir2   = csir::to_csir_level(mesh.domain());
        auto domain_on2  = csir::project_to_level(dom_csir2, level);
        auto domain_lca2 = csir::from_csir_level(domain_on2, mesh.origin_point(), mesh.scaling_factor());
        auto domain      = self(domain_lca2);

        lcl_t outer_boundary_lcl(level, mesh.origin_point(), mesh.scaling_factor());

        // Build inner boundary for this direction using CSIR and keep LCA alive
        auto cells_csir = csir::to_csir_level(mesh[mesh_id_t::cells][level]);
        std::array<int, Mesh::dim> d_in{}; for (std::size_t k=0;k<Mesh::dim;++k) d_in[k] = -direction[k];
        auto dom_shift  = csir::translate(domain_on2, d_in);
        auto inner_csir = csir::difference(cells_csir, dom_shift);
        auto inner_lca  = csir::from_csir_level(inner_csir, mesh.origin_point(), mesh.scaling_factor());
        auto inner_boundary = self(inner_lca);

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
                           // Materialize boundary_region at level
                           lca_t rhs_lca(level, mesh.origin_point(), mesh.scaling_factor());
                           boundary_region.on(level)(
                               [&](const auto& i, const auto& index){ rhs_lca[index].add_interval(i); });

                           // 2. Convert LevelCellArray to CSIR_Level
                           auto lhs_csir = csir::to_csir_level(lhs_lca);
                           auto rhs_csir = csir::to_csir_level(rhs_lca);

                           // 3. Perform CSIR intersection
                           auto result_csir = csir::intersection(lhs_csir, rhs_csir);

                           // 4. Convert result back to LevelCellArray and wrap it for the rest of the code
                           auto bdry_lca = csir::from_csir_level(result_csir, mesh.origin_point(), mesh.scaling_factor());
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
