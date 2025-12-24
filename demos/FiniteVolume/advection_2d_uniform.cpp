// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#include <array>

#include <xtensor/containers/xfixed.hpp>

#include <samurai/algorithm.hpp>
#include <samurai/algorithm/update.hpp>
#include <samurai/bc.hpp>
#include <samurai/field.hpp>
#include <samurai/io/hdf5.hpp>
#include <samurai/samurai.hpp>
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
    samurai::save(path, fmt::format("{}{}", filename, suffix), mesh, u);
}

int main(int argc, char* argv[])
{
    auto& app = samurai::initialize("Finite volume example for the advection equation in 2d using uniform mesh", argc, argv);

    constexpr std::size_t dim = 2;

    // Simulation parameters
    xt::xtensor_fixed<double, xt::xshape<dim>> min_corner = {0., 0.};
    xt::xtensor_fixed<double, xt::xshape<dim>> max_corner = {1., 1.};
    std::array<double, dim> a{
        {1, 1}
    };
    double Tf           = .1;
    double cfl          = 0.5;
    double t            = 0.;
    std::size_t level   = 7;

    // Output parameters
    fs::path path        = fs::current_path();
    std::string filename = "FV_advection_2d_uniform";
    std::size_t nfiles   = 1;

    app.add_option("--min-corner", min_corner, "The min corner of the box")->capture_default_str()->group("Simulation parameters");
    app.add_option("--max-corner", max_corner, "The max corner of the box")->capture_default_str()->group("Simulation parameters");
    app.add_option("--velocity", a, "The velocity of the advection equation")->capture_default_str()->group("Simulation parameters");
    app.add_option("--level", level, "The uniform mesh level")->capture_default_str()->group("Simulation parameters");
    app.add_option("--cfl", cfl, "The CFL")->capture_default_str()->group("Simulation parameters");
    app.add_option("--Ti", t, "Initial time")->capture_default_str()->group("Simulation parameters");
    app.add_option("--Tf", Tf, "Final time")->capture_default_str()->group("Simulation parameters");
    app.add_option("--path", path, "Output path")->capture_default_str()->group("Output");
    app.add_option("--filename", filename, "File name prefix")->capture_default_str()->group("Output");
    app.add_option("--nfiles", nfiles, "Number of output files")->capture_default_str()->group("Output");

    SAMURAI_PARSE(argc, argv);

    const samurai::Box<double, dim> box(min_corner, max_corner);

    // Create uniform mesh using the new API
    auto config = samurai::mesh_config<dim>()
                      .level(level)
                      .max_stencil_size(2);

    auto mesh = samurai::uniform::make_mesh(box, config);

    std::cout << "Uniform mesh created with " << mesh.nb_cells() << " cells at level " << level << std::endl;

    auto u = samurai::make_scalar_field<double>("u", mesh);
    init(u);

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

        // Use the new update_ghost_uniform function
        samurai::update_ghost_uniform(u);

        unp1.resize();
        unp1 = u - dt * samurai::upwind(a, u);

        std::swap(u.array(), unp1.array());

        if (t >= static_cast<double>(nsave) * dt_save || t == Tf)
        {
            const std::string suffix = (nfiles != 1) ? fmt::format("_ite_{}", nsave++) : "";
            save(path, filename, u, suffix);
        }
    }

    std::cout << "Simulation completed successfully!" << std::endl;

    samurai::finalize();
    return 0;
}
