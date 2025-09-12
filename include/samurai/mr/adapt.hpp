// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <algorithm>
#include <concepts>

#include <CLI/CLI.hpp>

#include "../algorithm/graduation.hpp"
#include "../algorithm/update.hpp"
#include "../arguments.hpp"
#include "../boundary.hpp"
#include "../csir_unified/src/csir.hpp"
#include "../field.hpp"
#include "../timers.hpp"
#include "config.hpp"
#include "criteria.hpp"
#include "operators.hpp"
#include "rel_detail.hpp"

namespace samurai
{
    struct stencil_graduation
    {
        static auto call(samurai::Dim<1>)
        {
            return xt::xtensor_fixed<int, xt::xshape<2, 1>>{{1}, {-1}};
        }

        static auto call(samurai::Dim<2>)
        {
            return xt::xtensor_fixed<int, xt::xshape<4, 2>>{
                {1,  1 },
                {-1, -1},
                {-1, 1 },
                {1,  -1}
            };
            // return xt::xtensor_fixed<int, xt::xshape<4, 2>> stencil{{ 1,  0},
            //                                                         {-1,  0},
            //                                                         { 0,  1},
            //                                                         { 0,
            //                                                         -1}};
        }

        static auto call(samurai::Dim<3>)
        {
            return xt::xtensor_fixed<int, xt::xshape<8, 3>>{
                {1,  1,  1 },
                {-1, 1,  1 },
                {1,  -1, 1 },
                {-1, -1, 1 },
                {1,  1,  -1},
                {-1, 1,  -1},
                {1,  -1, -1},
                {-1, -1, -1}
            };
            // return xt::xtensor_fixed<int, xt::xshape<6, 3>> stencil{{ 1,  0,
            // 0},
            //                                                         {-1,  0,
            //                                                         0}, { 0,
            //                                                         1,  0},
            //                                                         { 0, -1,
            //                                                         0}, { 0,
            //                                                         0,  1},
            //                                                         { 0,  0,
            //                                                         -1}};
        }
    };

    namespace detail
    {
        template <class... TFields>
        struct get_fields_type
        {
            using fields_t = Field_tuple<TFields...>;
            using mesh_t   = typename fields_t::mesh_t;
            using common_t = typename fields_t::common_t;
            using detail_t = VectorField<mesh_t, common_t, detail::compute_n_comp<TFields...>()>;
        };

        template <class TField>
        struct get_fields_type<TField>
        {
            using fields_t = TField&;
            using mesh_t   = typename TField::mesh_t;
            using detail_t = std::conditional_t<TField::is_scalar,
                                                ScalarField<mesh_t, typename TField::value_type>,
                                                VectorField<mesh_t, typename TField::value_type, TField::n_comp, detail::is_soa_v<TField>>>;
        };
    }

    template <bool enlarge_, class PredictionFn, class TField, class... TFields>
    class Adapt
    {
      public:

        Adapt(PredictionFn&& prediction_fn, TField& field, TFields&... fields);

        template <class... Fields>
        void operator()(mra_config& config, Fields&... other_fields);

        template <class... Fields>
        void operator()(double eps, double regularity, Fields&... other_fields);

      private:

        using inner_fields_type = detail::get_fields_type<TField, TFields...>;
        using fields_t          = typename inner_fields_type::fields_t;
        using mesh_t            = typename inner_fields_type::mesh_t;
        using mesh_id_t         = typename mesh_t::mesh_id_t;
        using detail_t          = typename inner_fields_type::detail_t;
        using tag_t             = ScalarField<mesh_t, int>;

        static constexpr std::size_t dim = mesh_t::dim;
        static constexpr bool enlarge_v  = enlarge_;

        using interval_t    = typename mesh_t::interval_t;
        using coord_index_t = typename interval_t::coord_index_t;
        using cl_type       = typename mesh_t::cl_type;

        template <class... Fields>
        bool harten(std::size_t ite, const mra_config& cfg, Fields&... other_fields);

        PredictionFn m_prediction_fn;
        fields_t m_fields; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)
        detail_t m_detail;
        tag_t m_tag;
    };

    template <bool enlarge_, class PredictionFn, class TField, class... TFields>
    inline Adapt<enlarge_, PredictionFn, TField, TFields...>::Adapt(PredictionFn&& prediction_fn, TField& field, TFields&... fields)
        : m_prediction_fn(std::forward<PredictionFn>(prediction_fn))
        , m_fields(field, fields...)
        , m_detail("detail", field.mesh())
        , m_tag("tag", field.mesh())
    {
    }

    template <bool enlarge_, class PredictionFn, class TField, class... TFields>
    template <class... Fields>
    void Adapt<enlarge_, PredictionFn, TField, TFields...>::operator()(mra_config& cfg, Fields&... other_fields)
    {
        auto& mesh            = m_fields.mesh();
        std::size_t min_level = mesh.min_level();
        std::size_t max_level = mesh.max_level();

        if (min_level == max_level)
        {
            return;
        }

        times::timers.start("mesh adaptation");
        cfg.parse_args();
        for (std::size_t i = 0; i < max_level - min_level; ++i)
        {
            // std::cout << "MR mesh adaptation " << i << std::endl;
            m_detail.resize();
            m_detail.fill(0);
            m_tag.resize();
            m_tag.fill(0);
            if (harten(i, cfg, other_fields...))
            {
                break;
            }
        }
        times::timers.stop("mesh adaptation");
    }

    template <bool enlarge_, class PredictionFn, class TField, class... TFields>
    template <class... Fields>
    [[deprecated("Use mra_config instead of eps and regularity (see advection_2d example for more information)")]]
    void Adapt<enlarge_, PredictionFn, TField, TFields...>::operator()(double eps, double regularity, Fields&... other_fields)
    {
        operator()(mra_config().epsilon(eps).regularity(regularity), other_fields...);
    }

    // TODO: to remove since it is used at several place
    namespace detail
    {

        template <std::size_t dim>
        auto box_dir();

        template <>
        inline auto box_dir<1>()
        {
            return xt::xtensor_fixed<int, xt::xshape<2, 1>>{{-1}, {1}};
        }

        template <>
        inline auto box_dir<2>()
        {
            return xt::xtensor_fixed<int, xt::xshape<4, 2>>{
                {-1, 1 },
                {1,  1 },
                {-1, -1},
                {1,  -1}
            };
        }

        template <>
        inline auto box_dir<3>()
        {
            return xt::xtensor_fixed<int, xt::xshape<8, 3>>{
                {-1, -1, -1},
                {1,  -1, -1},
                {-1, 1,  -1},
                {1,  1,  -1},
                {-1, -1, 1 },
                {1,  -1, 1 },
                {-1, 1,  1 },
                {1,  1,  1 }
            };
        }
    }

    template <class Mesh>
    void keep_boundary_refined(const Mesh& mesh, ScalarField<Mesh, int>& tag, const DirectionVector<Mesh::dim>& direction)
    {
        // Since the adaptation process starts at max_level, we just need to flag to `keep` the boundary cells at max_level only.
        // There will never be boundary cells at lower levels.
        auto bdry = domain_boundary_layer(mesh, mesh.max_level(), direction, Mesh::config::max_stencil_width);
        for_each_cell(mesh,
                      bdry,
                      [&](auto& cell)
                      {
                          tag[cell] = static_cast<int>(CellFlag::keep);
                      });
    }

    template <class Mesh>
    void keep_boundary_refined(const Mesh& mesh, ScalarField<Mesh, int>& tag)
    {
        constexpr std::size_t dim = Mesh::dim;

        DirectionVector<dim> direction;
        direction.fill(0);
        for (std::size_t d = 0; d < dim; ++d)
        {
            direction(d) = 1;
            keep_boundary_refined(mesh, tag, direction);
            direction(d) = -1;
            keep_boundary_refined(mesh, tag, direction);
            direction(d) = 0;
        }
    }

    template <bool enlarge_, class PredictionFn, class TField, class... TFields>
    template <class... Fields>
    bool Adapt<enlarge_, PredictionFn, TField, TFields...>::harten(std::size_t ite, const mra_config& cfg, Fields&... other_fields)
    {
        auto& mesh = m_fields.mesh();

        times::timers.start("mesh adaptation");
        std::size_t min_level = mesh.min_level();
        std::size_t max_level = mesh.max_level();

        for_each_cell(mesh[mesh_id_t::cells],
                      [&](auto& cell)
                      {
                          m_tag[cell] = static_cast<int>(CellFlag::keep);
                      });

        for (std::size_t level = min_level; level <= max_level; ++level)
        {
            update_tag_subdomains(level, m_tag, true);
        }

        times::timers.stop("mesh adaptation");
        update_ghost_mr(m_fields);
        times::timers.start("mesh adaptation");

        //--------------------//
        // Detail computation //
        //--------------------//

        // We compute the detail in the cells and ghosts below the cells, except near the (non-periodic) boundaries, where we compute
        // the detail only in the cells (justification in the comments below).

        bool periodic_in_all_directions = true;
        std::array<bool, dim> contract_directions;
        for (std::size_t d = 0; d < dim; ++d)
        {
            periodic_in_all_directions = periodic_in_all_directions && mesh.is_periodic(d);
            contract_directions[d]     = !mesh.is_periodic(d);
        }

        for (std::size_t level = ((min_level > 0) ? min_level - 1 : 0); level < max_level - ite; ++level)
        {
            // Materialize with CSIR (non-lazy) to remove subset laziness.
            // Build CSIR sets at the appropriate levels
            auto all_lca_lvl      = mesh[mesh_id_t::all_cells][level];
            auto cells_lca_lvlp1  = mesh[mesh_id_t::cells][level + 1];

            auto all_csir         = csir::to_csir_level(all_lca_lvl);
            auto cells_p1_csir    = csir::to_csir_level(cells_lca_lvlp1);
            auto cells_on_lvl_csir = csir::project_to_level(cells_p1_csir, level);

            // 1. detail computation in the cells (at level+1):
            auto ghosts_below_cells_csir = csir::intersection(all_csir, cells_on_lvl_csir);
            auto ghosts_below_cells_lca  = csir::from_csir_level(ghosts_below_cells_csir, mesh.origin_point(), mesh.scaling_factor());
            auto ghosts_below_cells      = self(ghosts_below_cells_lca);
            ghosts_below_cells.apply_op(compute_detail(m_detail, m_fields));

            // 2. detail computation in the ghosts below cells (at level)
            if (level >= min_level)
            {
                if (periodic_in_all_directions)
                {
                    auto all_lca_lvlm1      = mesh[mesh_id_t::all_cells][level - 1];
                    auto all_lvlm1_csir     = csir::to_csir_level(all_lca_lvlm1);
                    auto gbc_on_lvlm1_csir  = csir::project_to_level(ghosts_below_cells_csir, level - 1);
                    auto ghosts2_csir       = csir::intersection(all_lvlm1_csir, gbc_on_lvlm1_csir);
                    auto ghosts2_lca        = csir::from_csir_level(ghosts2_csir, mesh.origin_point(), mesh.scaling_factor());
                    auto ghosts_2_levels_below_cells = self(ghosts2_lca);
                    ghosts_2_levels_below_cells.apply_op(compute_detail(m_detail, m_fields));
                }
                else
                {
                    // Remove up to 4 boundary layers in non-periodic directions at level+1
                    auto domain_lca         = mesh.domain();
                    auto domain_csir        = csir::to_csir_level(domain_lca);
                    auto domain_on_lvlp1    = csir::project_to_level(domain_csir, level + 1);
                    auto domain_wo_bdry_csir = csir::contract(domain_on_lvlp1, 4, contract_directions);

                    auto cells_wo_bdry_csir = csir::intersection(cells_p1_csir, domain_wo_bdry_csir);
                    auto cells_wo_on_lvl     = csir::project_to_level(cells_wo_bdry_csir, level);
                    auto ghosts_below_cells2_csir = csir::intersection(all_csir, cells_wo_on_lvl);

                    auto all_lca_lvlm1      = mesh[mesh_id_t::all_cells][level - 1];
                    auto all_lvlm1_csir     = csir::to_csir_level(all_lca_lvlm1);
                    auto ghosts2_on_lvlm1   = csir::project_to_level(ghosts_below_cells2_csir, level - 1);
                    auto ghosts2_csir       = csir::intersection(all_lvlm1_csir, ghosts2_on_lvlm1);
                    auto ghosts2_lca        = csir::from_csir_level(ghosts2_csir, mesh.origin_point(), mesh.scaling_factor());
                    auto ghosts_2_levels_below_cells = self(ghosts2_lca);
                    ghosts_2_levels_below_cells.apply_op(compute_detail(m_detail, m_fields));
                }
            }
        }

        if (cfg.relative_detail())
        {
            compute_relative_detail(m_detail, m_fields);
        }

        update_ghost_subdomains(m_detail);

        for (std::size_t level = min_level; level <= max_level - ite; ++level)
        {
            std::size_t exponent = dim * (max_level - level);
            double eps_l         = cfg.epsilon() / (1 << exponent);

            double regularity_to_use = cfg.regularity() + dim;

            // subset_1: intersection(cells[level], all_cells[level-1]).on(level-1)
            {
                auto cells_lvl_lca    = mesh[mesh_id_t::cells][level];
                auto all_lvlm1_lca    = mesh[mesh_id_t::all_cells][level - 1];
                auto cells_lvl_csir   = csir::to_csir_level(cells_lvl_lca);
                auto all_lvlm1_csir   = csir::to_csir_level(all_lvlm1_lca);
                auto cells_on_lvlm1   = csir::project_to_level(cells_lvl_csir, level - 1);
                auto inter_csir       = csir::intersection(cells_on_lvlm1, all_lvlm1_csir);
                auto inter_lca        = csir::from_csir_level(inter_csir, mesh.origin_point(), mesh.scaling_factor());
                auto subset_1         = self(inter_lca);
                subset_1.apply_op(to_coarsen_mr(m_detail, m_tag, eps_l, min_level),
                                  to_refine_mr(m_detail,
                                               m_tag,
                                               (pow(2.0, regularity_to_use)) * eps_l,
                                               max_level));
            }
            update_tag_subdomains(level, m_tag, true);
        }

        if (args::refine_boundary) // cppcheck-suppress knownConditionTrueFalse
        {
            keep_boundary_refined(mesh, m_tag);
        }

        for (std::size_t level = min_level; level <= max_level - ite; ++level)
        {
            // subset_2: cells at current level
            auto subset_2 = self(mesh[mesh_id_t::cells][level]);

            subset_2.apply_op(keep_around_refine(m_tag));

            if constexpr (enlarge_v)
            {
                auto subset_3 = self(mesh[mesh_id_t::cells_and_ghosts][level]);
                subset_2.apply_op(enlarge(m_tag));
                subset_3.apply_op(tag_to_keep<0>(m_tag, CellFlag::enlarge));
            }

            update_tag_periodic(level, m_tag);
            update_tag_subdomains(level, m_tag);
        }

        for (std::size_t level = max_level; level > 0; --level)
        {
            auto cells_lvl_lca    = mesh[mesh_id_t::cells][level];
            auto all_lvlm1_lca    = mesh[mesh_id_t::all_cells][level - 1];
            auto cells_lvl_csir   = csir::to_csir_level(cells_lvl_lca);
            auto all_lvlm1_csir   = csir::to_csir_level(all_lvlm1_lca);
            auto cells_on_lvlm1   = csir::project_to_level(cells_lvl_csir, level - 1);
            auto inter_csir       = csir::intersection(cells_on_lvlm1, all_lvlm1_csir);
            auto inter_lca        = csir::from_csir_level(inter_csir, mesh.origin_point(), mesh.scaling_factor());
            auto keep_subset      = self(inter_lca);

            update_tag_periodic(level, m_tag);
            update_tag_subdomains(level, m_tag);

            keep_subset.apply_op(maximum(m_tag));
        }
        using ca_type = typename mesh_t::ca_type;

        ca_type new_ca = update_cell_array_from_tag(mesh[mesh_id_t::cells], m_tag);
        make_graduation(new_ca,
                        mesh.domain(),
                        mesh.mpi_neighbourhood(),
                        mesh.periodicity(),
                        mesh_t::config::graduation_width,
                        mesh_t::config::max_stencil_width);
        mesh_t new_mesh{new_ca, mesh};
#ifdef SAMURAI_WITH_MPI
        mpi::communicator world;
        if (mpi::all_reduce(world, mesh == new_mesh, std::logical_and()))
#else
        if (mesh == new_mesh)
#endif // SAMURAI_WITH_MPI
        {
            return true;
        }

        times::timers.stop("mesh adaptation");
        update_ghost_mr(other_fields...);
        times::timers.start("mesh adaptation");

        update_fields(std::forward<PredictionFn>(m_prediction_fn), new_mesh, m_fields, other_fields...);
        m_fields.mesh().swap(new_mesh);
        return false;
    }

    template <class... TFields>
        requires(IsField<TFields> && ...)
    auto make_MRAdapt(TFields&... fields)
    {
        using prediction_fn_t = decltype(default_config::default_prediction_fn);
        return Adapt<false, prediction_fn_t, TFields...>(std::forward<prediction_fn_t>(default_config::default_prediction_fn), fields...);
    }

    template <class Prediction_fn, class... TFields>
        requires(!IsField<Prediction_fn>) && (IsField<TFields> && ...)
    auto make_MRAdapt(Prediction_fn&& prediction_fn, TFields&... fields)
    {
        std::cout << "Use custom prediction function for MRAdapt" << std::endl;
        return Adapt<false, Prediction_fn, TFields...>(std::forward<Prediction_fn>(prediction_fn), fields...);
    }

    template <bool enlarge_, class... TFields>
        requires(IsField<TFields> && ...)
    auto make_MRAdapt(TFields&... fields)
    {
        using prediction_fn_t = decltype(default_config::default_prediction_fn);
        return Adapt<enlarge_, prediction_fn_t, TFields...>(std::forward<prediction_fn_t>(default_config::default_prediction_fn), fields...);
    }

    template <bool enlarge_, class Prediction_fn, class... TFields>
        requires(!IsField<Prediction_fn>) && (IsField<TFields> && ...)
    auto make_MRAdapt(Prediction_fn&& prediction_fn, TFields&... fields)
    {
        return Adapt<enlarge_, Prediction_fn, TFields...>(std::forward<Prediction_fn>(prediction_fn), fields...);
    }
} // namespace samurai
