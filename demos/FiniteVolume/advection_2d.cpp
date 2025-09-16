// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#include <algorithm>
#include <array>
#include <type_traits>

#include <xtensor/xfixed.hpp>

#include <samurai/algorithm.hpp>
#include <samurai/bc.hpp>
#include <samurai/field.hpp>
#include <samurai/io/hdf5.hpp>
#include <samurai/io/restart.hpp>
#include <samurai/samurai.hpp>
#include <samurai/stencil.hpp>
#include <samurai/stencil_field.hpp>
#include <samurai/uniform_mesh.hpp>

#include <filesystem>
namespace fs = std::filesystem;

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

template <class Field>
void update_ghost_uniform(Field& field)
{
    using mesh_t    = typename Field::mesh_t;
    using mesh_id_t = typename mesh_t::mesh_id_t;
    using value_t   = typename mesh_t::interval_t::value_t;

    static_assert(std::is_same_v<mesh_id_t, samurai::UniformMeshId>, "Uniform ghost update expects a uniform mesh");
    static_assert(Field::dim == 2, "Uniform ghost update currently implemented for 2D fields.");

    auto& mesh          = field.mesh();
    auto& bc_container  = field.get_bc();
    if (bc_container.empty())
    {
        return;
    }

    const auto& cells              = mesh[mesh_id_t::cells];
    const auto& cells_and_ghosts   = mesh[mesh_id_t::cells_and_ghosts];
    const auto interior_minmax     = cells.minmax_indices();
    const auto ghost_minmax        = cells_and_ghosts.minmax_indices();
    const value_t i_min            = interior_minmax[0].first;
    const value_t i_max            = interior_minmax[0].second;
    const value_t j_min            = interior_minmax[1].first;
    const value_t j_max            = interior_minmax[1].second;
    const value_t ghost_i_min      = ghost_minmax[0].first;
    const value_t ghost_i_max      = ghost_minmax[0].second;
    const value_t ghost_j_min      = ghost_minmax[1].first;
    const value_t ghost_j_max      = ghost_minmax[1].second;

    auto direction_matches = [](const auto& bc, const samurai::DirectionVector<Field::dim>& direction)
    {
        const auto& dirs = bc.get_region().first;
        return std::any_of(dirs.begin(), dirs.end(),
                           [&](const auto& candidate)
                           {
                               for (std::size_t d = 0; d < Field::dim; ++d)
                               {
                                   if (candidate[d] != direction[d])
                                   {
                                       return false;
                                   }
                               }
                               return true;
                           });
    };

    auto boundary_value = [&](auto& bc, const samurai::DirectionVector<Field::dim>& direction, const auto& interior_cell)
    {
        if (bc.get_value_type() == samurai::BCVType::constant)
        {
            return bc.constant_value();
        }
        auto coords = interior_cell.face_center(direction);
        return bc.value(direction, interior_cell, coords);
    };

    auto assign_value = [&](auto& bc, const samurai::DirectionVector<Field::dim>& direction, value_t i, value_t j)
    {
        const bool in_domain = (i >= i_min && i < i_max && j >= j_min && j < j_max);
        if (in_domain)
        {
            return;
        }

        const value_t interior_i = std::clamp(i, i_min, static_cast<value_t>(i_max - 1));
        const value_t interior_j = std::clamp(j, j_min, static_cast<value_t>(j_max - 1));
        auto interior_cell       = cells.get_cell(interior_i, interior_j);
        auto ghost_cell          = cells_and_ghosts.get_cell(i, j);
        field[ghost_cell]        = boundary_value(bc, direction, interior_cell);
    };

    auto apply_direction = [&](const samurai::DirectionVector<Field::dim>& direction,
                               value_t i_begin,
                               value_t i_end,
                               value_t j_begin,
                               value_t j_end)
    {
        bool applied = false;
        for (auto& bc_ptr : bc_container)
        {
            auto& bc = *bc_ptr;
            if (!direction_matches(bc, direction))
            {
                continue;
            }

            applied = true;
            for (value_t i = i_begin; i < i_end; ++i)
            {
                for (value_t j = j_begin; j < j_end; ++j)
                {
                    assign_value(bc, direction, i, j);
                }
            }
        }

        if (!applied)
        {
            auto& bc = *bc_container.front();
            for (value_t i = i_begin; i < i_end; ++i)
            {
                for (value_t j = j_begin; j < j_end; ++j)
                {
                    assign_value(bc, direction, i, j);
                }
            }
        }
    };

    samurai::DirectionVector<Field::dim> direction;
    direction.fill(0);
    direction[0] = -1;
    apply_direction(direction, ghost_i_min, i_min, ghost_j_min, ghost_j_max);

    direction.fill(0);
    direction[0] = 1;
    apply_direction(direction, i_max, ghost_i_max, ghost_j_min, ghost_j_max);

    if constexpr (Field::dim >= 2)
    {
        direction.fill(0);
        direction[1] = -1;
        apply_direction(direction, ghost_i_min, ghost_i_max, ghost_j_min, j_min);

        direction.fill(0);
        direction[1] = 1;
        apply_direction(direction, ghost_i_min, ghost_i_max, j_max, ghost_j_max);
    }
}

int main(int argc, char* argv[])
{
    auto& app = samurai::initialize("Finite volume example for the advection equation in 2d using a uniform mesh", argc, argv);

    constexpr std::size_t dim = 2;
    using Config              = samurai::UniformConfig<dim>;
    using Mesh                = samurai::UniformMesh<Config>;
    using mesh_id_t           = typename Mesh::mesh_id_t;

    // Simulation parameters
    xt::xtensor_fixed<double, xt::xshape<dim>> min_corner = {0., 0.};
    xt::xtensor_fixed<double, xt::xshape<dim>> max_corner = {1., 1.};
    std::array<double, dim> a{
        {1, 1}
    };
    double Tf  = .1;
    double cfl = 0.5;
    double t   = 0.;
    std::string restart_file;

    // Mesh parameters
    std::size_t level = 10;

    // Output parameters
    fs::path path        = fs::current_path();
    std::string filename = "FV_advection_2d";
    std::size_t nfiles   = 1;

    app.add_option("--min-corner", min_corner, "The min corner of the box")->capture_default_str()->group("Simulation parameters");
    app.add_option("--max-corner", max_corner, "The max corner of the box")->capture_default_str()->group("Simulation parameters");
    app.add_option("--velocity", a, "The velocity of the advection equation")->capture_default_str()->group("Simulation parameters");
    app.add_option("--cfl", cfl, "The CFL")->capture_default_str()->group("Simulation parameters");
    app.add_option("--Ti", t, "Initial time")->capture_default_str()->group("Simulation parameters");
    app.add_option("--Tf", Tf, "Final time")->capture_default_str()->group("Simulation parameters");
    app.add_option("--restart-file", restart_file, "Restart file")->capture_default_str()->group("Simulation parameters");
    app.add_option("--level", level, "Refinement level of the uniform mesh")->capture_default_str()->group("Mesh");
    app.add_option("--path", path, "Output path")->capture_default_str()->group("Output");
    app.add_option("--filename", filename, "File name prefix")->capture_default_str()->group("Output");
    app.add_option("--nfiles", nfiles, "Number of output files")->capture_default_str()->group("Output");

    SAMURAI_PARSE(argc, argv);

    const samurai::Box<double, dim> box(min_corner, max_corner);
    Mesh mesh;
    auto u = samurai::make_scalar_field<double>("u", mesh);

    if (restart_file.empty())
    {
        mesh = Mesh{box, level};
        init(u);
    }
    else
    {
        samurai::load(restart_file, mesh, u);
        level = mesh[mesh_id_t::cells].level();
    }
    samurai::make_bc<samurai::Dirichlet<1>>(u, 0.);

    double dt            = cfl * mesh.cell_length(level);
    const double dt_save = Tf / static_cast<double>(nfiles);

    auto unp1 = samurai::make_scalar_field<double>("unp1", mesh);
    save(path, filename, u, "_init");

    std::size_t nsave = 1;
    std::size_t nt    = 0;

    while (t != Tf)
    {
        t += dt;
        if (t > Tf)
        {
            dt += Tf - t;
            t = Tf;
        }

        std::cout << fmt::format("iteration {}: t = {}, dt = {}", nt++, t, dt) << std::endl;

        update_ghost_uniform(u);
        unp1.resize();
        unp1 = u - dt * samurai::upwind(a, u);

        std::swap(u.array(), unp1.array());

        if (t >= static_cast<double>(nsave + 1) * dt_save || t == Tf)
        {
            const std::string suffix = (nfiles != 1) ? fmt::format("_ite_{}", nsave++) : "";
            save(path, filename, u, suffix);
        }
    }
    samurai::finalize();
    return 0;
}
