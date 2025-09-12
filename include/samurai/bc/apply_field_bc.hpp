// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "../boundary.hpp"
#include "../concepts.hpp"
#include "polynomial_extrapolation.hpp"
#include <type_traits>

namespace samurai
{
    template <class Field, class Subset, std::size_t stencil_size, class Vector>
    void __apply_bc_on_subset(Bc<Field>& bc,
                              Field& field,
                              Subset& subset,
                              const StencilAnalyzer<stencil_size, Field::dim>& stencil,
                              const Vector& direction)
    {
        auto bc_function = bc.get_apply_function(std::integral_constant<std::size_t, stencil_size>(), direction);
        if (bc.get_value_type() == BCVType::constant)
        {
            auto value = bc.constant_value();
            for_each_stencil(field.mesh(),
                             subset,
                             stencil,
                             [&, value](auto& cells)
                             {
                                 bc_function(field, cells, value);
                             });
        }
        else if (bc.get_value_type() == BCVType::function)
        {
            assert(stencil.has_origin);
            for_each_stencil(field.mesh(),
                             subset,
                             stencil,
                             [&](auto& cells)
                             {
                                 auto& cell_in    = cells[stencil.origin_index];
                                 auto face_coords = cell_in.face_center(direction);
                                 auto value       = bc.value(direction, cell_in, face_coords);
                                 bc_function(field, cells, value);
                             });
        }
        else
        {
            std::cerr << "Unknown BC type" << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    template <class Field, std::size_t stencil_size>
    void apply_bc_impl(Bc<Field>& bc, std::size_t level, const DirectionVector<Field::dim>& direction, Field& field)
    {
        using mesh_id_t = typename Field::mesh_t::mesh_id_t;

        static constexpr std::size_t dim = Field::dim;

        auto& mesh = field.mesh();

        auto& region            = bc.get_region();
        auto& region_directions = region.first;
        auto& region_lca        = region.second;
        auto stencil_0          = bc.get_stencil(std::integral_constant<std::size_t, stencil_size>());

        for (std::size_t d = 0; d < region_directions.size(); ++d)
        {
            if (region_directions[d] != direction)
            {
                continue;
            }

            bool is_periodic = false;
            for (std::size_t i = 0; i < dim; ++i)
            {
                if (direction(i) != 0 && field.mesh().is_periodic(i))
                {
                    is_periodic = true;
                    break;
                }
            }
            if (!is_periodic)
            {
                bool is_cartesian_direction = is_cartesian(direction);

                if (is_cartesian_direction)
                {
                    auto stencil          = convert_for_direction(stencil_0, direction);
                    auto stencil_analyzer = make_stencil_analyzer(stencil);

                    // Inner cells in the boundary region (CSIR)
                    auto cells_csir   = csir::to_csir_level(mesh[mesh_id_t::cells][level]);
                    auto region_csir  = csir::to_csir_level(region_lca[d]);
                    auto region_on_l  = csir::project_to_level(region_csir, level);
                    auto inter_csir   = csir::intersection(cells_csir, region_on_l);
                    auto inter_lca    = csir::from_csir_level(inter_csir, mesh.origin_point(), mesh.scaling_factor());
                    auto bdry_cells   = self(inter_lca);
                    if (level >= mesh.min_level()) // otherwise there is no cells
                    {
                        __apply_bc_on_subset(bc, field, bdry_cells, stencil_analyzer, direction);
                    }
                }
            }
        }
    }

    template <class Field, std::size_t stencil_size>
    void apply_bc_impl(Bc<Field>& bc, std::size_t level, Field& field)
    {
        static_nested_loop<Field::dim, -1, 2>(
            [&](auto& direction)
            {
                if (xt::any(xt::not_equal(direction, 0))) // direction != {0, ..., 0}
                {
                    apply_bc_impl<Field, stencil_size>(bc, level, direction, field);
                }
            });
    }

    /**
     * Apply polynomial extrapolation on the outside ghosts close to boundary cells
     * @param bc The PolynomialExtrapolation boundary condition
     * @param level Level where to apply the polynomial extrapolation
     * @param field Field to apply the extrapolation on
     * @param direction Direction of the boundary
     * @param bdry_cells subset corresponding to boundary cells where to apply the extrapolation on (center of the BC stencil)
     */
    template <std::size_t stencil_size, class Field, class Subset>
    void __apply_extrapolation_bc__cells(Bc<Field>& bc,
                                         std::size_t level,
                                         Field& field,
                                         const DirectionVector<Field::dim>& direction,
                                         Subset& bdry_cells)
    {
        using mesh_id_t = typename Field::mesh_t::mesh_id_t;

        auto& mesh = field.mesh();

        auto stencil_0        = bc.get_stencil(std::integral_constant<std::size_t, stencil_size>());
        auto stencil          = convert_for_direction(stencil_0, direction);
        auto stencil_analyzer = make_stencil_analyzer(stencil);

        //  We need to check that the furthest ghost exists. It's not always the case for large stencils!
        if constexpr (stencil_size == 2)
        {
            // CSIR: intersection(cells[level], bdry_cells)
            using lca_t = typename Field::mesh_t::lca_type;
            using lcl_t = typename Field::mesh_t::lcl_type;
            lcl_t lcl(level, mesh.origin_point(), mesh.scaling_factor());
            auto bdry_subset = [&]() {
                if constexpr (samurai::IsLCA<std::decay_t<Subset>>)
                {
                    return self(bdry_cells);
                }
                else
                {
                    return bdry_cells;
                }
            }();
            bdry_subset.on(level)(
                [&](const auto& interval, const auto& index)
                {
                    lcl[index].add_interval(interval);
                });
            lca_t bdry_lca(lcl);
            auto cells_csir  = csir::to_csir_level(mesh[mesh_id_t::cells][level]);
            auto bdry_csir   = csir::to_csir_level(bdry_lca);
            auto inter_csir  = csir::intersection(cells_csir, bdry_csir);
            auto inter_lca   = csir::from_csir_level(inter_csir, mesh.origin_point(), mesh.scaling_factor());
            auto cells       = self(inter_lca);

            __apply_bc_on_subset(bc, field, cells, stencil_analyzer, direction);
        }
        else
        {
            using lca_t = typename Field::mesh_t::lca_type;
            using lcl_t = typename Field::mesh_t::lcl_type;
            lca_t trans_lca = translated_outer_neighbours<stencil_size / 2>(mesh, level, direction);
            lcl_t lcl2(level, mesh.origin_point(), mesh.scaling_factor());
            auto bdry_subset2 = [&]() {
                if constexpr (samurai::IsLCA<std::decay_t<Subset>>)
                {
                    return self(bdry_cells);
                }
                else
                {
                    return bdry_cells;
                }
            }();
            bdry_subset2.on(level)(
                [&](const auto& interval, const auto& index)
                {
                    lcl2[index].add_interval(interval);
                });
            lca_t bdry_lca(lcl2);

            auto trans_csir = csir::to_csir_level(trans_lca);
            auto cells_csir = csir::to_csir_level(mesh[mesh_id_t::cells][level]);
            auto bdry_csir  = csir::to_csir_level(bdry_lca);
            auto inter1     = csir::intersection(trans_csir, cells_csir);
            auto inter2     = csir::intersection(inter1, bdry_csir);
            auto inter_lca  = csir::from_csir_level(inter2, mesh.origin_point(), mesh.scaling_factor());
            auto cells      = self(inter_lca);

            __apply_bc_on_subset(bc, field, cells, stencil_analyzer, direction);
        }
    }

    template <std::size_t layers, class Mesh>
    auto translated_outer_neighbours(const Mesh& mesh, std::size_t level, const DirectionVector<Mesh::dim>& direction)
    {
        using mesh_id_t = typename Mesh::mesh_id_t;
        using lca_t     = typename Mesh::lca_type;
        static_assert(layers <= 5, "not implemented for layers > 10");

        // Build base CSIR set for the reference cells at this level
        auto ref_csir = csir::to_csir_level(mesh[mesh_id_t::reference][level]);

        auto translate_steps = [&](int steps)
        {
            std::array<int, Mesh::dim> d{};
            for (std::size_t k = 0; k < Mesh::dim; ++k)
            {
                d[k] = -direction[k] * steps;
            }
            return csir::translate(ref_csir, d);
        };

        auto acc = translate_steps(static_cast<int>(layers));
        for (int s = static_cast<int>(layers) - 1; s >= 1; --s)
        {
            acc = csir::intersection(acc, translate_steps(s));
        }

        return lca_t(csir::from_csir_level(acc, mesh.origin_point(), mesh.scaling_factor()));
    }

    /**
     * Apply polynomial extrapolation on the outside ghosts close to inner ghosts at the boundary
     * (i.e. inner ghosts in the boundary region that have neighbouring ghosts outside the domain)
     * @param bc The PolynomialExtrapolation boundary condition
     * @param level Level where to apply the polynomial extrapolation
     * @param field Field to apply the extrapolation on
     * @param direction Direction of the boundary
     * @param subset subset corresponding to inner ghosts where to apply the extrapolation on (center of the BC stencil)
     */
    template <std::size_t stencil_size, class Field, class Subset>
    void __apply_extrapolation_bc__ghosts(Bc<Field>& bc,
                                          std::size_t level,
                                          Field& field,
                                          const DirectionVector<Field::dim>& direction,
                                          Subset& inner_ghosts_location)
    {
        using mesh_id_t = typename Field::mesh_t::mesh_id_t;

        auto& mesh = field.mesh();

        auto stencil_0        = bc.get_stencil(std::integral_constant<std::size_t, stencil_size>());
        auto stencil          = convert_for_direction(stencil_0, direction);
        auto stencil_analyzer = make_stencil_analyzer(stencil);

        auto translated_outer_nghbr = translated_outer_neighbours<stencil_size / 2>(mesh, level, direction);
        using lca_t = typename Field::mesh_t::lca_type;
        using lcl_t = typename Field::mesh_t::lcl_type;

        // Unify: build lca for translated_outer_nghbr regardless of whether it is already an LCA or a Subset
        lca_t trans_lca = [&]() -> lca_t {
            if constexpr (std::is_same_v<decltype(translated_outer_nghbr), lca_t>)
            {
                return translated_outer_nghbr;
            }
            else
            {
                lcl_t trans_lcl(level, mesh.origin_point(), mesh.scaling_factor());
                translated_outer_nghbr.on(level)(
                    [&](const auto& i, const auto& index)
                    {
                        trans_lcl[index].add_interval(i);
                    });
                return lca_t(trans_lcl);
            }
        }();

        // Materialize inner_ghosts_location (always a Subset)
        lca_t inner_lca = [&]() -> lca_t {
            lcl_t inner_lcl(level, mesh.origin_point(), mesh.scaling_factor());
            inner_ghosts_location.on(level)(
                [&](const auto& i, const auto& index)
                {
                    inner_lcl[index].add_interval(i);
                });
            return lca_t(inner_lcl);
        }();

        auto trans_csir   = csir::to_csir_level(trans_lca);
        auto inner_csir   = csir::to_csir_level(inner_lca);
        auto inter_csir   = csir::intersection(trans_csir, inner_csir);
        auto inter_lca    = csir::from_csir_level(inter_csir, mesh.origin_point(), mesh.scaling_factor());
        auto union_csir   = csir::to_csir_level(mesh.get_union()[level]);
        auto inter2_csir  = csir::intersection(csir::to_csir_level(inter_lca), union_csir);
        auto icg_lca      = csir::from_csir_level(inter2_csir, mesh.origin_point(), mesh.scaling_factor());
        auto cells_csir   = csir::to_csir_level(mesh[mesh_id_t::cells][level]);
        auto diff_csir    = csir::difference(csir::to_csir_level(icg_lca), cells_csir);
        auto igo_lca      = csir::from_csir_level(diff_csir, mesh.origin_point(), mesh.scaling_factor());
        auto inner_ghosts_with_outer_nghbr = self(igo_lca);
        __apply_bc_on_subset(bc, field, inner_ghosts_with_outer_nghbr, stencil_analyzer, direction);
    }

    template <class Field>
        requires IsField<Field>
    void apply_field_bc(std::size_t level, const DirectionVector<Field::dim>& direction, Field& field)
    {
        static constexpr std::size_t max_stencil_size_implemented_BC = Bc<Field>::max_stencil_size_implemented;

        for (auto& bc : field.get_bc())
        {
            static_for<1, max_stencil_size_implemented_BC + 1>::apply( // for (int i=1; i<=max_stencil_size_implemented; i++)
                [&](auto integral_constant_i)
                {
                    static constexpr std::size_t i = decltype(integral_constant_i)::value;

                    if (bc->stencil_size() == i)
                    {
                        apply_bc_impl<Field, i>(*bc.get(), level, direction, field);
                    }
                });
        }
    }

    template <class Field>
        requires IsField<Field>
    void apply_field_bc(Field& field, const DirectionVector<Field::dim>& direction)
    {
        using mesh_id_t = typename Field::mesh_t::mesh_id_t;
        auto& mesh      = field.mesh()[mesh_id_t::reference];

        for (std::size_t level = mesh.min_level(); level <= mesh.max_level(); ++level)
        {
            apply_field_bc(level, direction, field);
        }
    }

    template <class Field>
        requires IsField<Field>
    void apply_field_bc(Field& field, std::size_t direction_index)
    {
        DirectionVector<Field::dim> direction;
        direction.fill(0);

        direction[direction_index] = 1;
        apply_field_bc(field, direction);

        direction[direction_index] = -1;
        apply_field_bc(field, direction);
    }

    template <class Field>
        requires IsField<Field>
    void apply_field_bc(std::size_t level, Field& field, std::size_t direction_index)
    {
        DirectionVector<Field::dim> direction;
        direction.fill(0);

        direction[direction_index] = 1;
        apply_field_bc(level, direction, field);

        direction[direction_index] = -1;
        apply_field_bc(level, direction, field);
    }

    template <class Field>
        requires IsField<Field>
    void apply_field_bc(Field& field)
    {
        for_each_cartesian_direction<Field::dim>(
            [&](const auto& direction)
            {
                apply_field_bc(field, direction);
            });
    }

    template <class Field, class... Fields>
        requires(IsField<Field> && (IsField<Fields> && ...))
    void apply_field_bc(Field& field, Fields&... other_fields)
    {
        apply_field_bc(field, other_fields...);
    }

    template <class Field>
    void update_outer_corners_by_polynomial_extrapolation(std::size_t level, const DirectionVector<Field::dim>& direction, Field& field)
    {
        if constexpr (Field::dim == 1)
        {
            return; // No outer corners in 1D
        }

        static constexpr std::size_t extrap_stencil_size = 2;

        auto& domain = detail::get_mesh(field.mesh());
        PolynomialExtrapolation<Field, extrap_stencil_size> bc(domain, ConstantBc<Field>(), true);

        // CSIR: corner(level, direction) = project_to_level( domain \\ (⋃_k translate(domain, -e_k*dir_k)), level )
        auto dom_csir = csir::to_csir_level(field.mesh().domain());
        // Build per-axis shifts and union them at domain's native level
        auto union_shifts = dom_csir; bool first = true;
        for (std::size_t ax = 0; ax < Field::dim; ++ax) {
            if (direction[ax] == 0) continue;
            std::array<int, Field::dim> s{}; s.fill(0); s[ax] = -direction[ax];
            auto t = csir::translate(dom_csir, s);
            if (first) { union_shifts = t; first = false; }
            else { union_shifts = csir::union_(union_shifts, t); }
        }
        if (first) { // no axes set; nothing to extrapolate
            return;
        }
        auto corner_dom = csir::difference(dom_csir, union_shifts);
        auto corner_cs  = csir::project_to_level(corner_dom, level);
        auto corner_lc = csir::from_csir_level(corner_cs, field.mesh().origin_point(), field.mesh().scaling_factor());
        auto corner    = self(corner_lc);

        __apply_extrapolation_bc__cells<extrap_stencil_size>(bc, level, field, direction, corner);
    }

    template <class Field>
    void update_outer_corners_by_polynomial_extrapolation(std::size_t level, Field& field)
    {
        static constexpr std::size_t dim = Field::dim;

        if constexpr (dim == 1)
        {
            return; // No outer corners in 1D
        }

        for_each_diagonal_direction<dim>(
            [&](const auto& direction)
            {
                bool is_periodic = false;
                for (std::size_t i = 0; i < dim; ++i)
                {
                    if (direction(i) != 0 && field.mesh().is_periodic(i))
                    {
                        is_periodic = true;
                        break;
                    }
                }
                if (!is_periodic)
                {
                    update_outer_corners_by_polynomial_extrapolation(level, direction, field);
                }
            });
    }

    template <class Field>
    void update_further_ghosts_by_polynomial_extrapolation(std::size_t level, const DirectionVector<Field::dim>& direction, Field& field)
    {
        static constexpr std::size_t ghost_width                     = Field::mesh_t::config::ghost_width;
        static constexpr std::size_t max_stencil_size_implemented_PE = PolynomialExtrapolation<Field, 2>::max_stencil_size_implemented_PE;

        // 1. We fill the ghosts that are further than those filled by the B.C. (where there are boundary cells)

        std::size_t ghost_layers_filled_by_bc = 0;
        for (auto& bc : field.get_bc())
        {
            ghost_layers_filled_by_bc = std::max(ghost_layers_filled_by_bc, bc->stencil_size() / 2);
        }

        // We populate the ghosts sequentially from the closest to the farthest.
        for (std::size_t ghost_layer = ghost_layers_filled_by_bc + 1; ghost_layer <= ghost_width; ++ghost_layer)
        {
            std::size_t stencil_s = 2 * ghost_layer;
            static_for<2, std::min(max_stencil_size_implemented_PE, 2 * ghost_width) + 1>::apply(
                [&](auto integral_constant_i)
                {
                    static constexpr std::size_t stencil_size = integral_constant_i();

                    if constexpr (stencil_size % 2 == 0) // (because PolynomialExtrapolation is only implemented for even stencil_size)
                    {
                        if (stencil_s == stencil_size)
                        {
                            auto& domain = detail::get_mesh(field.mesh());
                            PolynomialExtrapolation<Field, stencil_size> bc(domain, ConstantBc<Field>(), true);

                            auto boundary_cells = domain_boundary(field.mesh(), level, direction);
                            __apply_extrapolation_bc__cells<stencil_size>(bc, level, field, direction, boundary_cells);
                        }
                    }
                });
        }

        // 2. We fill the ghosts that are further than those filled by the projection of the B.C. (where there are ghost cells below
        // boundary cells)

        const std::size_t ghost_layers_filled_by_projection_bc = 1;

        for (std::size_t ghost_layer = ghost_layers_filled_by_projection_bc + 1; ghost_layer <= ghost_width; ++ghost_layer)
        {
            std::size_t stencil_s = 2 * ghost_layer;
            static_for<2, std::min(max_stencil_size_implemented_PE, 2 * ghost_width) + 1>::apply(
                [&](auto integral_constant_i)
                {
                    static constexpr std::size_t stencil_size = integral_constant_i();

                    if constexpr (stencil_size % 2 == 0) // (because PolynomialExtrapolation is only implemented for even stencil_size)
                    {
                        if (stencil_s == stencil_size)
                        {
                            auto& domain = detail::get_mesh(field.mesh());
                            PolynomialExtrapolation<Field, stencil_size> bc(domain, ConstantBc<Field>(), true);

                            // CSIR-based boundary ghosts: domain[level] \\ translate(domain[level], -direction)
                            auto dom_csir   = csir::to_csir_level(field.mesh().domain());
                            auto dom_on     = csir::project_to_level(dom_csir, level);
                            std::array<int, Field::dim> d{}; for (std::size_t k = 0; k < Field::dim; ++k) d[k] = -direction[k];
                            auto shifted    = csir::translate(dom_on, d);
                            auto bdry_csir  = csir::difference(dom_on, shifted);
                            auto bdry_lca   = csir::from_csir_level(bdry_csir, field.mesh().origin_point(), field.mesh().scaling_factor());
                            auto boundary_ghosts = self(bdry_lca);
                            __apply_extrapolation_bc__ghosts<stencil_size>(bc, level, field, direction, boundary_ghosts);
                        }
                    }
                });
        }
    }

    template <class Field>
    void update_further_ghosts_by_polynomial_extrapolation(Field& field, const DirectionVector<Field::dim>& direction)
    {
        using mesh_id_t = typename Field::mesh_t::mesh_id_t;
        auto& mesh      = field.mesh()[mesh_id_t::reference];

        for (std::size_t level = mesh.min_level(); level <= mesh.max_level(); ++level)
        {
            update_further_ghosts_by_polynomial_extrapolation(level, direction, field);
        }
    }

    template <class Field>
    void update_further_ghosts_by_polynomial_extrapolation(Field& field)
    {
        for_each_cartesian_direction<Field::dim>(
            [&](const auto& direction)
            {
                update_further_ghosts_by_polynomial_extrapolation(field, direction);
            });
    }

    template <class Field, class... Fields>
    void update_further_ghosts_by_polynomial_extrapolation(Field& field, Fields&... other_fields)
    {
        update_further_ghosts_by_polynomial_extrapolation(field);
        update_further_ghosts_by_polynomial_extrapolation(other_fields...);
    }
}
