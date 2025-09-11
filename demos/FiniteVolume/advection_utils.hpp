// Lightweight fallback header for init/save utilities used in advection demos.
// This mirrors the API of the experimental C++20 module `samurai.advection_utils`.

#pragma once

#include <filesystem>
#include <string>

#include <xtensor/xfixed.hpp>

#include <samurai/algorithm.hpp>
#include <samurai/bc.hpp>
#include <samurai/field.hpp>
#include <samurai/io/hdf5.hpp>
#include <samurai/io/restart.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/samurai.hpp>
#include <samurai/stencil_field.hpp>
#include <samurai/subset/node.hpp>

namespace fs = std::filesystem;

namespace samurai_advection_utils
{
    template <class Field>
    inline void init(Field& u)
    {
        auto& mesh = u.mesh();
        u.resize();

        samurai::for_each_cell(
            mesh,
            [&](auto& cell)
            {
                auto center           = cell.center();
                const double radius   = .2;
                const double x_center = 0.3;
                const double y_center = 0.3;
                if (((center[0] - x_center) * (center[0] - x_center) + (center[1] - y_center) * (center[1] - y_center)) <= radius * radius)
                {
                    u[cell] = 1;
                }
                else
                {
                    u[cell] = 0;
                }
            });
    }

    template <class Field>
    inline void save(const fs::path& path, const std::string& filename, const Field& u, const std::string& suffix = "")
    {
        auto& mesh = u.mesh();

#ifdef SAMURAI_WITH_MPI
        mpi::communicator world;
        samurai::save(path, fmt::format("{}_size_{}{}", filename, world.size(), suffix), mesh, u);
#else
        samurai::save(path, fmt::format("{}{}", filename, suffix), mesh, u);
        samurai::dump(path, fmt::format("{}_restart{}", filename, suffix), mesh, u);
#endif
    }
}

// Provide unqualified names like in the module for easy drop-in replacement.
template <class Field>
inline void init(Field& u)
{
    samurai_advection_utils::init(u);
}

template <class Field>
inline void save(const fs::path& path, const std::string& filename, const Field& u, const std::string& suffix = "")
{
    samurai_advection_utils::save(path, filename, u, suffix);
}

