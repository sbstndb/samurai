// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <array>
#include <fmt/format.h>

#include "box.hpp"
#include "level_cell_array.hpp"
#include "level_cell_list.hpp"
#include "mesh.hpp"
#include "samurai_config.hpp"

namespace samurai
{
    enum class UniformMeshId
    {
        cells            = 0,
        cells_and_ghosts = 1,
        count            = 2,
        reference        = cells_and_ghosts
    };

    template <std::size_t dim_, int ghost_width_ = default_config::ghost_width, class TInterval = default_config::interval_t>
    struct UniformConfig
    {
        static constexpr std::size_t dim = dim_;
        static constexpr int ghost_width = ghost_width_;
        static constexpr std::size_t max_refinement_level = default_config::max_level;
        static constexpr std::size_t max_stencil_width    = default_config::ghost_width;
        static constexpr int prediction_order             = default_config::prediction_order;
        using interval_t                 = TInterval;
        using mesh_id_t                  = UniformMeshId;
    };
    template <class Config>
    class UniformMesh : public samurai::Mesh_base<UniformMesh<Config>, Config>
    {
      public:

        using base_type                  = samurai::Mesh_base<UniformMesh<Config>, Config>;
        using self_type                  = UniformMesh<Config>;
        using config                     = typename base_type::config;
        static constexpr std::size_t dim = config::dim;

        using mesh_id_t  = typename base_type::mesh_id_t;
        using interval_t = typename base_type::interval_t;
        using ca_type    = typename base_type::ca_type;
        using cl_type    = typename base_type::cl_type;
        using lca_type   = typename base_type::lca_type;

        using base_type::cell_length;
        using base_type::get_cell;
        using base_type::get_index;
        using base_type::get_interval;
        using base_type::nb_cells;
        using base_type::operator[];
        using base_type::origin_point;
        using base_type::scaling_factor;
        using base_type::set_origin_point;
        using base_type::set_scaling_factor;
        using base_type::swap;

        UniformMesh() = default;
        explicit UniformMesh(const cl_type& cl);
        explicit UniformMesh(const ca_type& ca);
        UniformMesh(const Box<double, dim>& b,
                    std::size_t level,
                    double approx_box_tol = lca_type::default_approx_box_tol,
                    double scaling_factor = 0);
        UniformMesh(const Box<double, dim>& b,
                    std::size_t level,
                    const std::array<bool, dim>& periodic,
                    double approx_box_tol = lca_type::default_approx_box_tol,
                    double scaling_factor = 0);

        void update_sub_mesh_impl();
        void to_stream(std::ostream& os) const;
    };

    template <class Config>
    inline UniformMesh<Config>::UniformMesh(const typename UniformMesh<Config>::cl_type& cl)
        : UniformMesh<Config>::base_type(cl, cl.level(), cl.level())
    {
    }

    template <class Config>
    inline UniformMesh<Config>::UniformMesh(const typename UniformMesh<Config>::ca_type& ca)
        : UniformMesh<Config>::base_type(ca, ca.min_level(), ca.max_level())
    {
    }

    template <class Config>
    inline UniformMesh<Config>::UniformMesh(const Box<double, UniformMesh<Config>::dim>& b,
                                            std::size_t level,
                                            double approx_box_tol,
                                            double scaling_factor_)
        : UniformMesh<Config>::base_type(b, level, level, level, approx_box_tol, scaling_factor_)
    {
    }

    template <class Config>
    inline UniformMesh<Config>::UniformMesh(const Box<double, UniformMesh<Config>::dim>& b,
                                            std::size_t level,
                                            const std::array<bool, UniformMesh<Config>::dim>& periodic,
                                            double approx_box_tol,
                                            double scaling_factor_)
        : UniformMesh<Config>::base_type(b, level, level, level, periodic, approx_box_tol, scaling_factor_)
    {
    }

    template <class Config>
    inline void UniformMesh<Config>::update_sub_mesh_impl()
    {
        auto& cells_array = this->cells();
        const std::size_t level = this->min_level();

        using cl_type      = typename UniformMesh<Config>::cl_type;
        using ca_type      = typename UniformMesh<Config>::ca_type;
        using mesh_id_t    = typename UniformMesh<Config>::mesh_id_t;
        using config_type  = typename UniformMesh<Config>::config;
        constexpr auto dim = UniformMesh<Config>::dim;

        cl_type cl(this->origin_point(), this->scaling_factor());
        for (std::size_t lvl = 0; lvl <= config_type::max_refinement_level; ++lvl)
        {
            cl[lvl].clear();
        }

        for_each_interval(cells_array[mesh_id_t::cells],
                          [&](std::size_t lvl, const auto& interval, const auto& index_yz)
                          {
                              if (lvl != level)
                              {
                                  return;
                              }
                              static_nested_loop<dim - 1, -config_type::ghost_width, config_type::ghost_width + 1>(
                                  [&](auto stencil)
                                  {
                                      auto index = xt::eval(index_yz + stencil);
                                      cl[level][index].add_interval({interval.start - config_type::ghost_width, interval.end + config_type::ghost_width});
                                  });
                          });

        cells_array[mesh_id_t::cells_and_ghosts] = ca_type(cl);
        this->update_meshid_neighbour(mesh_id_t::cells_and_ghosts);
    }

    template <class Config>
    inline void UniformMesh<Config>::to_stream(std::ostream& os) const
    {
        using mesh_id_t = typename UniformMesh<Config>::mesh_id_t;

        for (std::size_t id = 0; id < static_cast<std::size_t>(mesh_id_t::count); ++id)
        {
            auto mt = static_cast<mesh_id_t>(id);

            os << fmt::format(fmt::emphasis::bold, "{}\n{:─^50}", mt, "") << std::endl;
            os << (*this)[mt];
        }
    }

    template <class Config>
    inline bool operator==(const UniformMesh<Config>& mesh1, const UniformMesh<Config>& mesh2)
    {
        using mesh_id_t = typename UniformMesh<Config>::mesh_id_t;

        return (mesh1[mesh_id_t::cells] == mesh2[mesh_id_t::cells]);
    }

    template <class Config>
    inline std::ostream& operator<<(std::ostream& out, const UniformMesh<Config>& mesh)
    {
        mesh.to_stream(out);
        return out;
    }
} // namespace samurai

template <>
struct fmt::formatter<samurai::UniformMeshId> : formatter<string_view>
{
    // parse is inherited from formatter<string_view>.
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
