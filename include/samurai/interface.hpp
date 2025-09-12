#pragma once
#include "boundary.hpp"
#include "stencil.hpp"
#include <array>

namespace samurai
{
    template <Get get_type, class interface_iterator_t, class stencil_iterator_t, class Func>
    inline void apply_on_interval(const typename interface_iterator_t::mesh_interval_t& mesh_interval,
                                  interface_iterator_t& interface_it,
                                  stencil_iterator_t& comput_stencil_it,
                                  Func&& f)
    {
        comput_stencil_it.init(mesh_interval);
        interface_it.init(mesh_interval);

        if constexpr (get_type == Get::Intervals)
        {
            f(interface_it, comput_stencil_it);
        }
        else if constexpr (get_type == Get::Cells)
        {
            for (std::size_t ii = 0; ii < mesh_interval.i.size(); ++ii)
            {
                f(interface_it.cells(), comput_stencil_it.cells());
                interface_it.move_next();
                comput_stencil_it.move_next();
            }
        }
    }

    /**
     * Iterates over the interfaces of same level only (no level jump).
     * Same parameters as the preceding function.
     */
    template <Run run_type = Run::Sequential, Get get_type = Get::Cells, bool include_periodic = true, class Mesh, std::size_t comput_stencil_size, class Func>
    void for_each_interior_interface__same_level(const Mesh& mesh,
                                                 std::size_t level,
                                                 const DirectionVector<Mesh::dim>& direction,
                                                 const StencilAnalyzer<comput_stencil_size, Mesh::dim>& comput_stencil,
                                                 Func&& f)
    {
        static constexpr std::size_t dim = Mesh::dim;
        using mesh_id_t                  = typename Mesh::mesh_id_t;
        using mesh_interval_t            = typename Mesh::mesh_interval_t;

        Stencil<2, dim> interface_stencil_ = in_out_stencil<dim>(direction);
        auto interface_stencil             = make_stencil_analyzer(interface_stencil_);

#ifdef SAMURAI_WITH_OPENMP
        std::size_t num_threads = static_cast<std::size_t>(omp_get_max_threads());
        std::vector<IteratorStencil<Mesh, 2>> interface_its;
        std::vector<IteratorStencil<Mesh, comput_stencil_size>> comput_stencil_its;
        for (std::size_t i = 0; i < num_threads; ++i)
        {
            interface_its.push_back(make_stencil_iterator(mesh, interface_stencil));
            comput_stencil_its.push_back(make_stencil_iterator(mesh, comput_stencil));
        }
#else
        auto interface_it      = make_stencil_iterator(mesh, interface_stencil);
        auto comput_stencil_it = make_stencil_iterator(mesh, comput_stencil);
#endif

        auto apply_on_interface = [&](const auto& lhs_cells,
                                     const auto& rhs_cells,
                                     const std::array<int, dim>& lhs_shift,
                                     const std::array<int, dim>& rhs_shift)
        {
            using lca_t = typename Mesh::lca_type;
            lca_t lhs_lca(lhs_cells);
            lca_t rhs_lca(rhs_cells);
            auto lhs_csir = csir::to_csir_level(lhs_lca);
            auto rhs_csir = csir::to_csir_level(rhs_lca);
            // Apply requested shifts in CSIR
            auto lhs_t = lhs_csir;
            auto rhs_t = rhs_csir;
            // Detect non-zero shifts
            bool lhs_nonzero = false, rhs_nonzero = false;
            for (std::size_t i = 0; i < dim; ++i) { lhs_nonzero = lhs_nonzero || (lhs_shift[i] != 0); rhs_nonzero = rhs_nonzero || (rhs_shift[i] != 0); }
            if (lhs_nonzero) lhs_t = csir::translate(lhs_csir, lhs_shift);
            if (rhs_nonzero) rhs_t = csir::translate(rhs_csir, rhs_shift);
            auto inter    = csir::intersection(lhs_t, rhs_t);
            auto inter_l  = csir::from_csir_level(inter, mesh.origin_point(), mesh.scaling_factor());
            auto intersect = self(inter_l).on(level);

            for_each_meshinterval<mesh_interval_t, run_type>(
                intersect,
                [&](auto mesh_interval)
                {
#ifdef SAMURAI_WITH_OPENMP
                    std::size_t thread      = static_cast<std::size_t>(omp_get_thread_num());
                    auto& interface_it      = interface_its[thread];
                    auto& comput_stencil_it = comput_stencil_its[thread];
#endif
                    apply_on_interval<get_type>(mesh_interval, interface_it, comput_stencil_it, std::forward<Func>(f));
                });
        };

        // Base case: intersect cells with cells shifted by -direction
        std::array<int, dim> zero{}; zero.fill(0);
        std::array<int, dim> neg_dir{}; for (std::size_t k = 0; k < dim; ++k) neg_dir[k] = -direction[k];
        apply_on_interface(mesh[mesh_id_t::cells][level], mesh[mesh_id_t::cells][level], zero, neg_dir);

        auto d = find_direction_index(direction);
        if constexpr (include_periodic)
        {
            if (mesh.periodicity()[d])
            {
                auto shift = get_periodic_shift(mesh.domain(), level, d);
                std::array<int, dim> sh{}; sh.fill(0); sh[d] = static_cast<int>(shift[d]);
                // RHS shift = shift then -direction (combined)
                std::array<int, dim> rhs_shift = sh; for (std::size_t k = 0; k < dim; ++k) rhs_shift[k] -= direction[k];
                apply_on_interface(mesh[mesh_id_t::cells][level], mesh[mesh_id_t::cells][level], zero, rhs_shift);
                // LHS shift = -shift; RHS shift = -direction
                std::array<int, dim> lhs_shift = sh; for (auto& v : lhs_shift) v = -v;
                apply_on_interface(mesh[mesh_id_t::cells][level], mesh[mesh_id_t::cells][level], lhs_shift, neg_dir);
            }
        }
#ifdef SAMURAI_WITH_MPI
        for (const auto& neigh : mesh.mpi_neighbourhood())
        {
            apply_on_interface(mesh[mesh_id_t::cells][level], neigh.mesh[mesh_id_t::cells][level], zero, neg_dir);
            apply_on_interface(neigh.mesh[mesh_id_t::cells][level], mesh[mesh_id_t::cells][level], zero, neg_dir);
            if constexpr (include_periodic)
            {
                if (mesh.periodicity()[d])
                {
                    auto shift = get_periodic_shift(mesh.domain(), level, d);
                    std::array<int, dim> sh{}; sh.fill(0); sh[d] = static_cast<int>(shift[d]);
                    std::array<int, dim> rhs_shift = sh; for (std::size_t k = 0; k < dim; ++k) rhs_shift[k] -= direction[k];
                    apply_on_interface(mesh[mesh_id_t::cells][level], neigh.mesh[mesh_id_t::cells][level], zero, rhs_shift);
                    std::array<int, dim> lhs_shift_n = sh; for (auto& v : lhs_shift_n) v = -v;
                    apply_on_interface(neigh.mesh[mesh_id_t::cells][level], mesh[mesh_id_t::cells][level], lhs_shift_n, neg_dir);
                }
            }
        }
#endif
    }

    /**
     * Iterates over the level jumps (level --> level+1) that occur in the chosen direction.
     *
     *         |__|   l+1
     *    |____|      l
     *    --------->
     *    direction
     *
     * The provided callback @param f has the following signature:
     *           void f(auto& interface_cells, auto& comput_cells)
     * where
     *       'interface_cells' = [cell_{l}, cell_{l+1}].
     */
    template <Run run_type = Run::Sequential, Get get_type = Get::Cells, bool include_periodic = true, class Mesh, std::size_t comput_stencil_size, class Func>
    void for_each_interior_interface__level_jump_direction(const Mesh& mesh,
                                                           std::size_t level,
                                                           const DirectionVector<Mesh::dim>& direction,
                                                           const StencilAnalyzer<comput_stencil_size, Mesh::dim>& comput_stencil,
                                                           Func&& f)
    {
        static constexpr std::size_t dim = Mesh::dim;
        using mesh_id_t       = typename Mesh::mesh_id_t;
        using mesh_interval_t = typename Mesh::mesh_interval_t;

        if (level >= mesh.max_level())
        {
            return;
        }

        int direction_index_int = comput_stencil.find(direction);
        auto direction_index    = static_cast<std::size_t>(direction_index_int);
#ifdef SAMURAI_WITH_OPENMP
        std::size_t num_threads = static_cast<std::size_t>(omp_get_max_threads());
        std::vector<IteratorStencil<Mesh, comput_stencil_size>> comput_stencil_its;
        comput_stencil_its.reserve(num_threads);
        std::vector<LevelJumpIterator<0, Mesh, comput_stencil_size>> interface_its;
        interface_its.reserve(num_threads);
        for (std::size_t i = 0; i < num_threads; ++i)
        {
            comput_stencil_its.emplace_back(mesh, comput_stencil);
            interface_its.emplace_back(comput_stencil_its[i], direction_index);
        }
#else
        auto comput_stencil_it = make_stencil_iterator(mesh, comput_stencil);
        auto interface_it      = make_leveljump_iterator<0>(comput_stencil_it, direction_index);
#endif

        auto apply_on_interface = [&](const auto& coarse_cells,
                                     const auto& fine_cells,
                                     const std::array<int, dim>& coarse_shift,
                                     const std::array<int, dim>& fine_preshift)
        {
            using lca_t = typename Mesh::lca_type;
            lca_t coarse_lca(coarse_cells);
            lca_t fine_lca(fine_cells);
            auto coarse_csir = csir::to_csir_level(coarse_lca);
            // optional shift on coarse (before projecting up)
            bool c_nonzero = false; for (std::size_t k = 0; k < dim; ++k) c_nonzero = c_nonzero || (coarse_shift[k] != 0);
            if (c_nonzero) { coarse_csir = csir::translate(coarse_csir, coarse_shift); }
            auto coarse_on_fine = csir::project_to_level(coarse_csir, level + 1);
            auto fine_csir   = csir::to_csir_level(fine_lca);
            // combine preshift with -direction
            std::array<int, dim> rhs_shift = fine_preshift; for (std::size_t k = 0; k < dim; ++k) rhs_shift[k] -= direction[k];
            bool f_nonzero = false; for (std::size_t k = 0; k < dim; ++k) f_nonzero = f_nonzero || (rhs_shift[k] != 0);
            if (f_nonzero) { fine_csir = csir::translate(fine_csir, rhs_shift); }
            auto inter_csir  = csir::intersection(coarse_on_fine, fine_csir);
            auto inter_lca   = csir::from_csir_level(inter_csir, mesh.origin_point(), mesh.scaling_factor());
            auto fine_intersect = self(inter_lca).on(level + 1);

            for_each_meshinterval<mesh_interval_t, run_type>(
                fine_intersect,
                [&](auto fine_mesh_interval)
                {
#ifdef SAMURAI_WITH_OPENMP
                    std::size_t thread      = static_cast<std::size_t>(omp_get_thread_num());
                    auto& interface_it      = interface_its[thread];
                    auto& comput_stencil_it = comput_stencil_its[thread];
#endif
                    apply_on_interval<get_type>(fine_mesh_interval, interface_it, comput_stencil_it, std::forward<Func>(f));
                });
        };

        std::array<int, dim> zeros{}; zeros.fill(0);
        apply_on_interface(mesh[mesh_id_t::cells][level], mesh[mesh_id_t::cells][level + 1], zeros, zeros);

        auto d = find_direction_index(direction);
        if constexpr (include_periodic)
        {
            if (mesh.periodicity()[d])
            {
                auto shift = get_periodic_shift(mesh.domain(), level + 1, d);
                std::array<int, dim> sh1{}; sh1.fill(0); sh1[d] = static_cast<int>(shift[d]);
                apply_on_interface(mesh[mesh_id_t::cells][level], mesh[mesh_id_t::cells][level + 1], zeros, sh1);
                shift = get_periodic_shift(mesh.domain(), level, d);
                std::array<int, dim> sh2{}; sh2.fill(0); sh2[d] = -static_cast<int>(shift[d]);
                apply_on_interface(mesh[mesh_id_t::cells][level], mesh[mesh_id_t::cells][level + 1], sh2, zeros);
            }
        }
#ifdef SAMURAI_WITH_MPI
        for (const auto& neigh : mesh.mpi_neighbourhood())
        {
            apply_on_interface(mesh[mesh_id_t::cells][level], neigh.mesh[mesh_id_t::cells][level + 1], zeros, zeros);
            apply_on_interface(neigh.mesh[mesh_id_t::cells][level], mesh[mesh_id_t::cells][level + 1], zeros, zeros);
            if constexpr (include_periodic)
            {
                if (mesh.periodicity()[d])
                {
                    auto shift = get_periodic_shift(mesh.domain(), level + 1, d);
                    std::array<int, dim> sh1{}; sh1.fill(0); sh1[d] = static_cast<int>(shift[d]);
                    apply_on_interface(mesh[mesh_id_t::cells][level], neigh.mesh[mesh_id_t::cells][level + 1], zeros, sh1);
                    shift = get_periodic_shift(mesh.domain(), level, d);
                    std::array<int, dim> sh2{}; sh2.fill(0); sh2[d] = -static_cast<int>(shift[d]);
                    apply_on_interface(neigh.mesh[mesh_id_t::cells][level], mesh[mesh_id_t::cells][level + 1], sh2, zeros);
                }
            }
        }
#endif
    }

    /**
     * Iterates over the level jumps (level --> level+1) that occur in the OPPOSITE direction of @param direction.
     *
     *    |__|        l+1
     *       |____|   l
     *    --------->
     *    direction
     *
     * The provided callback @param f has the following signature:
     *           void f(auto& interface_cells, auto& comput_cells)
     * where
     *       'interface_cells' = [cell_{l+1}, cell_{l}].
     */
    template <Run run_type = Run::Sequential, Get get_type = Get::Cells, bool include_periodic = true, class Mesh, std::size_t comput_stencil_size, class Func>
    void for_each_interior_interface__level_jump_opposite_direction(const Mesh& mesh,
                                                                    std::size_t level,
                                                                    const DirectionVector<Mesh::dim>& direction,
                                                                    const StencilAnalyzer<comput_stencil_size, Mesh::dim>& comput_stencil,
                                                                    Func&& f)
    {
        static constexpr std::size_t dim = Mesh::dim;
        using mesh_id_t                  = typename Mesh::mesh_id_t;
        using mesh_interval_t            = typename Mesh::mesh_interval_t;

        if (level >= mesh.max_level())
        {
            return;
        }

        Stencil<comput_stencil_size, dim> minus_comput_stencil_ = comput_stencil.stencil - direction;
        auto minus_comput_stencil                               = make_stencil_analyzer(minus_comput_stencil_);
        DirectionVector<dim> minus_direction                    = -direction;
        int minus_direction_index_int                           = minus_comput_stencil.find(minus_direction);
        auto minus_direction_index                              = static_cast<std::size_t>(minus_direction_index_int);

#ifdef SAMURAI_WITH_OPENMP
        std::size_t num_threads = static_cast<std::size_t>(omp_get_max_threads());
        std::vector<IteratorStencil<Mesh, comput_stencil_size>> comput_stencil_its;
        comput_stencil_its.reserve(num_threads);
        std::vector<LevelJumpIterator<1, Mesh, comput_stencil_size>> interface_its;
        interface_its.reserve(num_threads);
        for (std::size_t i = 0; i < num_threads; ++i)
        {
            comput_stencil_its.emplace_back(mesh, minus_comput_stencil);
            interface_its.emplace_back(comput_stencil_its[i], minus_direction_index);
        }
#else
        auto minus_comput_stencil_it = make_stencil_iterator(mesh, minus_comput_stencil);
        auto interface_it            = make_leveljump_iterator<1>(minus_comput_stencil_it, minus_direction_index);
#endif

        auto apply_on_interface = [&](const auto& coarse_cells,
                                     const auto& fine_cells,
                                     const std::array<int, dim>& coarse_shift,
                                     const std::array<int, dim>& fine_preshift)
        {
            using lca_t = typename Mesh::lca_type;
            lca_t coarse_lca(coarse_cells);
            lca_t fine_lca(fine_cells);
            auto coarse_csir = csir::to_csir_level(coarse_lca);
            bool c_nonzero = false; for (std::size_t k = 0; k < dim; ++k) c_nonzero = c_nonzero || (coarse_shift[k] != 0);
            if (c_nonzero) coarse_csir = csir::translate(coarse_csir, coarse_shift);
            auto coarse_on_fine = csir::project_to_level(coarse_csir, level + 1);
            auto fine_csir   = csir::to_csir_level(fine_lca);
            // combine preshift with +direction (opposite variant)
            std::array<int, dim> rhs_shift = fine_preshift; for (std::size_t k = 0; k < dim; ++k) rhs_shift[k] += direction[k];
            bool f_nonzero = false; for (std::size_t k = 0; k < dim; ++k) f_nonzero = f_nonzero || (rhs_shift[k] != 0);
            if (f_nonzero) fine_csir = csir::translate(fine_csir, rhs_shift);
            auto inter_csir  = csir::intersection(coarse_on_fine, fine_csir);
            auto inter_lca   = csir::from_csir_level(inter_csir, mesh.origin_point(), mesh.scaling_factor());
            auto fine_intersect = self(inter_lca).on(level + 1);

            for_each_meshinterval<mesh_interval_t, run_type>(
                fine_intersect,
                [&](auto fine_mesh_interval)
                {
#ifdef SAMURAI_WITH_OPENMP
                    std::size_t thread            = static_cast<std::size_t>(omp_get_thread_num());
                    auto& interface_it            = interface_its[thread];
                    auto& minus_comput_stencil_it = comput_stencil_its[thread];
#endif
                    apply_on_interval<get_type>(fine_mesh_interval, interface_it, minus_comput_stencil_it, std::forward<Func>(f));
                });
        };

        std::array<int, dim> zeros_op{}; zeros_op.fill(0);
        apply_on_interface(mesh[mesh_id_t::cells][level], mesh[mesh_id_t::cells][level + 1], zeros_op, zeros_op);

        auto d = find_direction_index(direction);
        if constexpr (include_periodic)
        {
            if (mesh.periodicity()[d])
            {
                auto shift = get_periodic_shift(mesh.domain(), level + 1, d);
                std::array<int, dim> sh1{}; sh1.fill(0); sh1[d] = -static_cast<int>(shift[d]);
                apply_on_interface(mesh[mesh_id_t::cells][level], mesh[mesh_id_t::cells][level + 1], zeros_op, sh1);
                shift = get_periodic_shift(mesh.domain(), level, d);
                std::array<int, dim> sh2{}; sh2.fill(0); sh2[d] = static_cast<int>(shift[d]);
                apply_on_interface(mesh[mesh_id_t::cells][level], mesh[mesh_id_t::cells][level + 1], sh2, zeros_op);
            }
        }
#ifdef SAMURAI_WITH_MPI
        for (const auto& neigh : mesh.mpi_neighbourhood())
        {
            apply_on_interface(mesh[mesh_id_t::cells][level], neigh.mesh[mesh_id_t::cells][level + 1], zeros_op, zeros_op);
            apply_on_interface(neigh.mesh[mesh_id_t::cells][level], mesh[mesh_id_t::cells][level + 1], zeros_op, zeros_op);
            if constexpr (include_periodic)
            {
                if (mesh.periodicity()[d])
                {
                    auto shift = get_periodic_shift(mesh.domain(), level + 1, d);
                    std::array<int, dim> sh1{}; sh1.fill(0); sh1[d] = -static_cast<int>(shift[d]);
                    apply_on_interface(mesh[mesh_id_t::cells][level], neigh.mesh[mesh_id_t::cells][level + 1], zeros_op, sh1);
                    shift = get_periodic_shift(mesh.domain(), level, d);
                    std::array<int, dim> sh2{}; sh2.fill(0); sh2[d] = static_cast<int>(shift[d]);
                    apply_on_interface(neigh.mesh[mesh_id_t::cells][level], mesh[mesh_id_t::cells][level + 1], sh2, zeros_op);
                }
            }
        }
#endif

        // DirectionVector<Mesh::dim> opposite_direction    = -direction;
        // decltype(comput_stencil) opposite_comput_stencil = comput_stencil - direction;
        // for_each_interior_interface__level_jump_direction<run_type, get_type>(mesh,
        //                                                                       level,
        //                                                                       opposite_direction,
        //                                                                       opposite_comput_stencil,
        //                                                                       std::forward<Func>(f));
    }

    /**
     * Iterates over the interior interfaces of the mesh level in the chosen direction.
     * @param level: the browsed interfaces will be defined by two cells of same level,
     *               or one cell of that level and another one level higher.
     * @param direction: positive Cartesian direction defining, for each cell, which neighbour defines the desired interface.
     *                   In 2D: {1,0} to browse horizontal interfaces, {0,1} to browse vertical interfaces.
     *
     * The provided callback @param f has the following signature:
     *           void f(auto& interface_cells, auto& comput_cells)
     * where
     *       'interface_cells' is an array containing the two real cells on both sides of the interface (might be of different levels).
     *       'comput_cells'    is an array containing the set of cells/ghosts defined by @param comput_stencil (all of same level).
     */
    template <Run run_type = Run::Sequential, Get get_type = Get::Cells, bool include_periodic = true, class Mesh, std::size_t comput_stencil_size, class Func>
    void for_each_interior_interface(const Mesh& mesh,
                                     std::size_t level,
                                     const DirectionVector<Mesh::dim>& direction,
                                     const StencilAnalyzer<comput_stencil_size, Mesh::dim>& comput_stencil,
                                     Func&& f)
    {
        for_each_interior_interface__same_level<run_type, get_type, include_periodic>(mesh,
                                                                                      level,
                                                                                      direction,
                                                                                      comput_stencil,
                                                                                      std::forward<Func>(f));
        for_each_interior_interface__level_jump_direction<run_type, get_type, include_periodic>(mesh,
                                                                                                level,
                                                                                                direction,
                                                                                                comput_stencil,
                                                                                                std::forward<Func>(f));
        for_each_interior_interface__level_jump_opposite_direction<run_type, get_type, include_periodic>(mesh,
                                                                                                         level,
                                                                                                         direction,
                                                                                                         comput_stencil,
                                                                                                         std::forward<Func>(f));
    }

    /**
     * Iterates over the interior interfaces of the mesh in the chosen direction.
     * @param direction: positive Cartesian direction defining, for each cell, which neighbour defines the desired interface.
     *                   In 2D: {1,0} to browse horizontal interfaces, {0,1} to browse vertical interfaces.
     * @param comput_stencil: the computational stencil, defining the set of cells (of same level)
     *                        captured in second argument of the callback function.
     *
     * The provided callback @param f has the following signature:
     *           void f(auto& interface_cells, auto& comput_cells)
     * where
     *       'interface_cells' is an array containing the two real cells on both sides of the interface (might be of different levels),
     *       'comput_cells'    is an array containing the set of cells/ghosts defined by @param comput_stencil (all of same level).
     */
    template <Run run_type = Run::Sequential, Get get_type = Get::Cells, bool include_periodic = true, class Mesh, std::size_t comput_stencil_size, class Func>
    void for_each_interior_interface(const Mesh& mesh,
                                     const DirectionVector<Mesh::dim>& direction,
                                     const StencilAnalyzer<comput_stencil_size, Mesh::dim>& comput_stencil,
                                     Func&& f)
    {
        for_each_level(mesh,
                       [&](auto level)
                       {
                           for_each_interior_interface<run_type, get_type, include_periodic>(mesh,
                                                                                             level,
                                                                                             direction,
                                                                                             comput_stencil,
                                                                                             std::forward<Func>(f));
                       });

        // for (std::size_t level = 0; level < mesh.max_level(); ++level)
        // {
        //     for_each_interior_interface<run_type, get_type>(mesh, level, direction, comput_stencil, std::forward<Func>(f));
        // }
    }

    /**
     * Iterates over the interior interfaces of the mesh in the chosen direction.
     * @param direction: positive Cartesian direction defining, for each cell, which neighbour defines the desired interface.
     *                   In 2D: {1,0} to browse horizontal interfaces, {0,1} to browse vertical interfaces.
     *
     * The provided callback @param f has the following signature:
     *           void f(auto& interface_cells, auto& comput_cells)
     * where
     *       'interface_cells' is an array containing the two real cells on both sides of the interface,
     *       'comput_cells'    is an array containing the two cells that must be used for the computation.
     * If there is no level jump, then 'interface_cells' = 'comput_cells'.
     * In case of level jump l/l+1, the cells of 'interface_cells' are of different levels,
     * while both cells of 'comput_cells' are at level l+1 and one of them is a ghost.
     */
    template <Run run_type = Run::Sequential, Get get_type = Get::Cells, bool include_periodic, class Mesh, class Func>
    void for_each_interior_interface(const Mesh& mesh, const DirectionVector<Mesh::dim>& direction, Func&& f)
    {
        static constexpr std::size_t dim = Mesh::dim;

        Stencil<2, dim> comput_stencil = in_out_stencil<dim>(direction);
        auto stencil_analyzer          = make_stencil_analyzer(comput_stencil);
        for_each_interior_interface<run_type, get_type, include_periodic>(mesh, direction, stencil_analyzer, std::forward<Func>(f));
    }

    /**
     * Iterates over the interior interfaces of the mesh.
     * The provided callback @param f has the following signature:
     *           void f(auto& interface_cells, auto& comput_cells)
     * where
     *       'interface_cells' is an array containing the two real cells on both sides of the interface,
     *       'comput_cells'    is an array containing the two cells that must be used for the computation.
     * If there is no level jump, then 'interface_cells' = 'comput_cells'.
     * In case of level jump l/l+1, the cells of 'interface_cells' are of different levels,
     * while both cells of 'comput_cells' are at level l+1 and one of them is a ghost.
     */
    template <Run run_type = Run::Sequential, Get get_type = Get::Cells, bool include_periodic = true, class Mesh, class Func>
    void for_each_interior_interface(const Mesh& mesh, Func&& f)
    {
        static constexpr std::size_t dim = Mesh::dim;

        for (std::size_t d = 0; d < dim; ++d)
        {
            DirectionVector<Mesh::dim> direction;
            direction.fill(0);
            direction[d] = 1;
            for_each_interior_interface<run_type, get_type, include_periodic>(mesh, direction, std::forward<Func>(f));
        }
    }

    //----------------------------------------------------------------------------------------------------------------------------------

    template <Run run_type = Run::Sequential, Get get_type = Get::Cells, class Mesh, std::size_t comput_stencil_size, class Func>
    void for_each_boundary_interface__direction(const Mesh& mesh,
                                                std::size_t level,
                                                const DirectionVector<Mesh::dim>& direction,
                                                const StencilAnalyzer<comput_stencil_size, Mesh::dim>& comput_stencil,
                                                Func&& f)
    {
        static constexpr std::size_t dim = Mesh::dim;
        using mesh_interval_t            = typename Mesh::mesh_interval_t;

        Stencil<2, dim> interface_stencil_ = in_out_stencil<dim>(direction);
        auto interface_stencil             = make_stencil_analyzer(interface_stencil_);

#ifdef SAMURAI_WITH_OPENMP
        std::size_t num_threads = static_cast<std::size_t>(omp_get_max_threads());
        std::vector<IteratorStencil<Mesh, 2>> interface_its;
        std::vector<IteratorStencil<Mesh, comput_stencil_size>> comput_stencil_its;
        for (std::size_t i = 0; i < num_threads; ++i)
        {
            interface_its.push_back(make_stencil_iterator(mesh, interface_stencil));
            comput_stencil_its.push_back(make_stencil_iterator(mesh, comput_stencil));
        }
#else
        auto interface_it      = make_stencil_iterator(mesh, interface_stencil);
        auto comput_stencil_it = make_stencil_iterator(mesh, comput_stencil);
#endif

        auto bdry_lca = domain_boundary(mesh, level, direction);
        auto bdry = self(bdry_lca);
        for_each_meshinterval<mesh_interval_t, run_type>(bdry,
                                                         [&](auto mesh_interval)
                                                         {
#ifdef SAMURAI_WITH_OPENMP
                                                             std::size_t thread      = static_cast<std::size_t>(omp_get_thread_num());
                                                             auto& interface_it      = interface_its[thread];
                                                             auto& comput_stencil_it = comput_stencil_its[thread];
#endif
                                                             interface_it.init(mesh_interval);
                                                             comput_stencil_it.init(mesh_interval);
                                                             if constexpr (get_type == Get::Intervals)
                                                             {
                                                                 f(interface_it, comput_stencil_it);
                                                             }
                                                             else if constexpr (get_type == Get::Cells)
                                                             {
                                                                 for (std::size_t ii = 0; ii < mesh_interval.i.size(); ++ii)
                                                                 {
                                                                     f(interface_it.cells()[0], comput_stencil_it.cells());
                                                                     interface_it.move_next();
                                                                     comput_stencil_it.move_next();
                                                                 }
                                                             }
                                                         });
    }

    template <Run run_type = Run::Sequential, Get get_type = Get::Cells, class Mesh, std::size_t comput_stencil_size, class Func>
    void for_each_boundary_interface__opposite_direction(const Mesh& mesh,
                                                         std::size_t level,
                                                         const DirectionVector<Mesh::dim>& direction,
                                                         const StencilAnalyzer<comput_stencil_size, Mesh::dim>& comput_stencil_analyzer,
                                                         Func&& f)
    {
        DirectionVector<Mesh::dim> opposite_direction                   = -direction;
        Stencil<comput_stencil_size, Mesh::dim> opposite_comput_stencil = comput_stencil_analyzer.stencil - direction;
        auto opposite_comput_stencil_analyzer                           = make_stencil_analyzer(opposite_comput_stencil);
        for_each_boundary_interface__direction<run_type, get_type>(mesh,
                                                                   level,
                                                                   opposite_direction,
                                                                   opposite_comput_stencil_analyzer,
                                                                   std::forward<Func>(f));
    }

    template <Run run_type = Run::Sequential, Get get_type = Get::Cells, class Mesh, std::size_t comput_stencil_size, class Func>
    void for_each_boundary_interface__direction(const Mesh& mesh,
                                                const DirectionVector<Mesh::dim>& direction,
                                                const StencilAnalyzer<comput_stencil_size, Mesh::dim>& comput_stencil,
                                                Func&& f)
    {
        for_each_level(
            mesh,
            [&](auto level)
            {
                for_each_boundary_interface__direction<run_type, get_type>(mesh, level, direction, comput_stencil, std::forward<Func>(f));
            });
    }

    template <Run run_type = Run::Sequential, Get get_type = Get::Cells, class Mesh, std::size_t comput_stencil_size, class Func>
    void for_each_boundary_interface__direction(const Mesh& mesh,
                                                const DirectionVector<Mesh::dim>& direction,
                                                const Stencil<comput_stencil_size, Mesh::dim>& comput_stencil,
                                                Func&& f)
    {
        auto stencil_analyzer = make_stencil_analyzer(comput_stencil);
        for_each_boundary_interface__direction<run_type, get_type>(mesh, direction, stencil_analyzer, std::forward<Func>(f));
    }

    template <Run run_type = Run::Sequential, Get get_type = Get::Cells, class Mesh, class Func>
    void for_each_boundary_interface__direction(const Mesh& mesh, const DirectionVector<Mesh::dim>& direction, Func&& f)
    {
        static constexpr std::size_t dim = Mesh::dim;

        Stencil<2, dim> comput_stencil = in_out_stencil<dim>(direction);
        auto compute_stencil_analyzer  = make_stencil_analyzer(comput_stencil);
        for_each_boundary_interface__direction<run_type, get_type>(mesh, direction, compute_stencil_analyzer, std::forward<Func>(f));
    }

    template <Run run_type = Run::Sequential, Get get_type = Get::Cells, class Mesh, std::size_t comput_stencil_size, class Func>
    void for_each_boundary_interface__both_directions(const Mesh& mesh,
                                                      std::size_t level,
                                                      const DirectionVector<Mesh::dim>& direction,
                                                      const StencilAnalyzer<comput_stencil_size, Mesh::dim>& comput_stencil,
                                                      Func&& f)
    {
        for_each_boundary_interface__direction<run_type, get_type>(mesh, level, direction, comput_stencil, std::forward<Func>(f));
        for_each_boundary_interface__opposite_direction<run_type, get_type>(mesh, level, direction, comput_stencil, std::forward<Func>(f));
    }

    /**
     * Iterates over the boundary interfaces in a given direction and its opposite direction.
     * @param direction: positive Cartesian direction defining, for each cell, which neighbour defines the desired interface.
     *                   In 2D: {1,0} to browse horizontal interfaces, {0,1} to browse vertical interfaces.
     * @param comput_stencil: the computational stencil, defining the set of cells (of same level)
     *                        captured in second argument of the callback function.
     *
     * The provided callback @param f has the following signature:
     *           void f(auto& cell, auto& comput_cells)
     * where
     *       'cell'         is the inner cell at the boundary.
     *       'comput cells' is the set of cells/ghosts defined by @param comput_stencil.
     */
    template <Run run_type = Run::Sequential, Get get_type = Get::Cells, class Mesh, std::size_t comput_stencil_size, class Func>
    void for_each_boundary_interface__both_directions(const Mesh& mesh,
                                                      const DirectionVector<Mesh::dim>& direction,
                                                      const StencilAnalyzer<comput_stencil_size, Mesh::dim>& comput_stencil,
                                                      Func&& f)
    {
        for_each_level(
            mesh,
            [&](auto level)
            {
                for_each_boundary_interface__both_directions<run_type, get_type>(mesh, level, direction, comput_stencil, std::forward<Func>(f));
            });
    }

    /**
     * Iterates over the boundary interfaces in a given direction and its opposite direction.
     * @param direction: positive Cartesian direction defining, for each cell, which neighbour defines the desired interface.
     *                   In 2D: {1,0} to browse horizontal interfaces, {0,1} to browse vertical interfaces.
     * @param comput_stencil: the computational stencil, defining the set of cells (of same level)
     *                        captured in second argument of the callback function.
     *
     * The provided callback @param f has the following signature:
     *           void f(auto& cell, auto& comput_cells)
     * where
     *       'cell'         is the inner cell at the boundary.
     *       'comput cells' is the set of cells/ghosts defined by @param comput_stencil.
     */
    template <Run run_type = Run::Sequential, Get get_type = Get::Cells, class Mesh, std::size_t comput_stencil_size, class Func>
    void for_each_boundary_interface__both_directions(const Mesh& mesh,
                                                      const DirectionVector<Mesh::dim>& direction,
                                                      const Stencil<comput_stencil_size, Mesh::dim>& comput_stencil,
                                                      Func&& f)
    {
        auto compute_stencil_analyzer = make_stencil_analyzer(comput_stencil);
        for_each_boundary_interface__both_directions<run_type, get_type>(mesh, direction, compute_stencil_analyzer, std::forward<Func>(f));
    }

    /**
     * Iterates over the boundary interfaces in a given direction and its opposite direction.
     * @param direction: positive Cartesian direction defining, for each cell, which neighbour defines the desired interface.
     *                   In 2D: {1,0} to browse horizontal interfaces, {0,1} to browse vertical interfaces.
     *
     * The provided callback @param f has the following signature:
     *           void f(auto& cell, auto& comput_cells)
     * where
     *       'cell'         is the inner cell at the boundary.
     *       'comput cells' is the array containing the inner cell and the outside ghost.
     */
    template <Run run_type = Run::Sequential, Get get_type = Get::Cells, class Mesh, class Func>
    void for_each_boundary_interface__both_directions(const Mesh& mesh, const DirectionVector<Mesh::dim>& direction, Func&& f)
    {
        static constexpr std::size_t dim = Mesh::dim;

        Stencil<2, dim> comput_stencil = in_out_stencil<dim>(direction);
        auto compute_stencil_analyzer  = make_stencil_analyzer(comput_stencil);
        for_each_boundary_interface__both_directions<run_type, get_type>(mesh, direction, compute_stencil_analyzer, std::forward<Func>(f));
    }

    /**
     * Iterates over the boundary interfaces.
     *
     * The provided callback @param f has the following signature:
     *           void f(auto& cell, auto& comput_cells)
     * where
     *       'cell'         is the inner cell at the boundary.
     *       'comput cells' is the array containing the inner cell and the outside ghost.
     */
    template <Run run_type = Run::Sequential, Get get_type = Get::Cells, class Mesh, class Func>
    void for_each_boundary_interface(const Mesh& mesh, Func&& f)
    {
        static constexpr std::size_t dim = Mesh::dim;

        for (std::size_t d = 0; d < dim; ++d)
        {
            if (!mesh.periodicity()[d])
            {
                DirectionVector<Mesh::dim> direction;
                direction.fill(0);
                direction[d] = 1;
                for_each_boundary_interface__both_directions<run_type, get_type>(mesh, direction, std::forward<Func>(f));
            }
        }
    }

}
