module;

// Global module fragment: include headers used by the implementation.
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

export module samurai.advection_utils;

namespace fs = std::filesystem;

export
{
    template <class Field>
    void init(Field& u)
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
    void save(const fs::path& path, const std::string& filename, const Field& u, const std::string& suffix = "")
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

