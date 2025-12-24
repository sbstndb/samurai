// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <fmt/format.h>

#include "../box.hpp"
#include "../mesh.hpp"
#include "../samurai_config.hpp"
#include "../static_algorithm.hpp"

namespace samurai
{
    //==========================================================================
    //                           UniformMeshId
    //==========================================================================

    enum class UniformMeshId
    {
        cells            = 0,
        cells_and_ghosts = 1,
        count            = 2,
        reference        = cells_and_ghosts
    };

    //==========================================================================
    //                          UniformMesh
    //==========================================================================

    template <class Config>
    class UniformMesh : public Mesh_base<UniformMesh<Config>, Config>
    {
      public:

        using base_type   = Mesh_base<UniformMesh<Config>, Config>;
        using self_type   = UniformMesh<Config>;
        using config      = typename base_type::config;

        static constexpr std::size_t dim = config::dim;

        using mesh_id_t       = typename base_type::mesh_id_t;
        using interval_t      = typename base_type::interval_t;
        using cl_type         = typename base_type::cl_type;
        using lcl_type        = typename base_type::lcl_type;
        using ca_type         = typename base_type::ca_type;
        using lca_type        = typename base_type::lca_type;
        using mpi_subdomain_t = typename base_type::mpi_subdomain_t;

        using base_type::ghost_width;
        using base_type::max_stencil_radius;

        // Modern constructors (via mesh_config)
        UniformMesh() = default;
        UniformMesh(const ca_type& ca, const self_type& ref_mesh);
        UniformMesh(const cl_type& cl, const self_type& ref_mesh);
        UniformMesh(const cl_type& cl, const mesh_config<Config::dim>& config);
        UniformMesh(const ca_type& ca, const mesh_config<Config::dim>& config);
        UniformMesh(const Box<double, dim>& b, const mesh_config<Config::dim>& config);
        UniformMesh(const DomainBuilder<dim>& domain_builder, const mesh_config<Config::dim>& config);

        // CRTP implementation required by Mesh_base
        void update_sub_mesh_impl();
    };

    //==========================================================================
    //                     Constructor Implementations
    //==========================================================================

    template <class Config>
    inline UniformMesh<Config>::UniformMesh(const ca_type& ca, const self_type& ref_mesh)
        : base_type(ca, ref_mesh)
    {
    }

    template <class Config>
    inline UniformMesh<Config>::UniformMesh(const cl_type& cl, const self_type& ref_mesh)
        : base_type(cl, ref_mesh)
    {
    }

    template <class Config>
    inline UniformMesh<Config>::UniformMesh(const cl_type& cl, const mesh_config<Config::dim>& config)
        : base_type(cl, config)
    {
    }

    template <class Config>
    inline UniformMesh<Config>::UniformMesh(const ca_type& ca, const mesh_config<Config::dim>& config)
        : base_type(ca, config)
    {
    }

    template <class Config>
    inline UniformMesh<Config>::UniformMesh(const Box<double, dim>& b, const mesh_config<Config::dim>& config)
        : base_type(b, config)
    {
    }

    template <class Config>
    inline UniformMesh<Config>::UniformMesh(const DomainBuilder<dim>& domain_builder, const mesh_config<Config::dim>& config)
        : base_type(domain_builder, config)
    {
    }

    //==========================================================================
    //                     update_sub_mesh_impl
    //==========================================================================

    template <class Config>
    inline void UniformMesh<Config>::update_sub_mesh_impl()
    {
        // For a uniform mesh: single level, just add ghosts around cells
        // This is much simpler than MRA (~15 lines vs ~350 lines)

        cl_type cell_list;

        for_each_interval(this->cells()[mesh_id_t::cells],
                          [&](std::size_t level, const auto& interval, const auto& index_yz)
                          {
                              lcl_type& lcl = cell_list[level];
                              static_nested_loop<dim - 1>(
                                  -max_stencil_radius(),
                                  max_stencil_radius() + 1,
                                  [&](auto stencil)
                                  {
                                      auto index = xt::eval(index_yz + stencil);
                                      lcl[index].add_interval({interval.start - max_stencil_radius(),
                                                               interval.end + max_stencil_radius()});
                                  });
                          });

        this->cells()[mesh_id_t::cells_and_ghosts] = {cell_list, false};
    }

    //==========================================================================
    //                    Factory Functions (namespace uniform)
    //==========================================================================

    namespace uniform
    {
        /**
         * @brief Create an empty uniform mesh (for delayed initialization)
         */
        template <class mesh_config_t, class complete_mesh_config_t = complete_mesh_config<mesh_config_t, UniformMeshId>>
        auto make_empty_mesh(const mesh_config_t&)
        {
            return UniformMesh<complete_mesh_config_t>();
        }

        /**
         * @brief Create a uniform mesh from a CellList
         */
        template <class mesh_config_t, class complete_mesh_config_t = complete_mesh_config<mesh_config_t, UniformMeshId>>
        auto make_mesh(const typename UniformMesh<complete_mesh_config_t>::cl_type& cl, const mesh_config_t& cfg)
        {
            auto mesh_cfg = cfg;
            mesh_cfg.parse_args();

            return UniformMesh<complete_mesh_config_t>(cl, mesh_cfg);
        }

        /**
         * @brief Create a uniform mesh from a CellArray
         */
        template <class mesh_config_t, class complete_mesh_config_t = complete_mesh_config<mesh_config_t, UniformMeshId>>
        auto make_mesh(const typename UniformMesh<complete_mesh_config_t>::ca_type& ca, const mesh_config_t& cfg)
        {
            auto mesh_cfg = cfg;
            mesh_cfg.parse_args();

            return UniformMesh<complete_mesh_config_t>(ca, mesh_cfg);
        }

        /**
         * @brief Create a uniform mesh from a Box
         */
        template <class mesh_config_t>
        auto make_mesh(const Box<double, mesh_config_t::dim>& b, const mesh_config_t& cfg)
        {
            using complete_cfg_t = complete_mesh_config<mesh_config_t, UniformMeshId>;

            auto mesh_cfg = cfg;
            mesh_cfg.parse_args();

            return UniformMesh<complete_cfg_t>(b, mesh_cfg);
        }

        /**
         * @brief Create a uniform mesh from a DomainBuilder
         */
        template <class mesh_config_t>
        auto make_mesh(const DomainBuilder<mesh_config_t::dim>& domain_builder, const mesh_config_t& cfg)
        {
            using complete_cfg_t = complete_mesh_config<mesh_config_t, UniformMeshId>;

            auto mesh_cfg = cfg;
            mesh_cfg.parse_args();

            return UniformMesh<complete_cfg_t>(domain_builder, mesh_cfg);
        }

    } // namespace uniform

} // namespace samurai

//==========================================================================
//                    fmt formatter for UniformMeshId
//==========================================================================

template <>
struct fmt::formatter<samurai::UniformMeshId> : formatter<string_view>
{
    template <typename FormatContext>
    auto format(samurai::UniformMeshId c, FormatContext& ctx) const
    {
        string_view name = "unknown";
        switch (c)
        {
            case samurai::UniformMeshId::cells:
                name = "cells";
                break;
            case samurai::UniformMeshId::cells_and_ghosts:
                name = "cells and ghosts";
                break;
            case samurai::UniformMeshId::count:
                name = "count";
                break;
        }
        return formatter<string_view>::format(name, ctx);
    }
};
