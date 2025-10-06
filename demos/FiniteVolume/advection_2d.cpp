// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#include <algorithm>
#include <array>
#include <cmath>

#include <xtensor/xfixed.hpp>

#include <samurai/algorithm/graduation.hpp>
#include <samurai/algorithm/update.hpp>
#include <samurai/bc.hpp>
#include <samurai/cell_flag.hpp>
#include <samurai/field.hpp>
#include <samurai/io/hdf5.hpp>
#include <samurai/io/restart.hpp>
#include <samurai/amr/mesh.hpp>
#include <samurai/samurai.hpp>
#include <samurai/schemes/fv/operators/gradient.hpp>
#include <samurai/stencil_field.hpp>
#include <samurai/subset/node.hpp>

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

template <class Field, class Tag>
void AMR_criterion(const Field& field,
                   Tag& tag,
                   double refine_threshold,
                   double coarsen_ratio,
                   bool allow_refine,
                   bool allow_coarsen)
{
    static_assert(Field::dim == 2, "AMR criterion implemented for 2d advection demo");

    auto& mesh      = field.mesh();
    using mesh_id_t = typename Field::mesh_t::mesh_id_t;
    const auto min_l = mesh.min_level();
    const auto max_l = mesh.max_level();
    const double coarsen_threshold = refine_threshold * coarsen_ratio;

    auto gradient_op = samurai::make_gradient_order2<Field>();
    auto& mutable_field = const_cast<Field&>(field); // gradient scheme updates ghost layers
    auto gradient       = gradient_op(mutable_field);

    samurai::for_each_cell(mesh[mesh_id_t::cells],
                           [&](auto cell)
                           {
                               const std::size_t level = cell.level;
                               const double dx         = mesh.cell_length(level);

                               const auto grad_val = gradient[cell];
                               double max_component = 0.;
                               for (std::size_t d = 0; d < Field::dim; ++d)
                               {
                                   const double component = grad_val(static_cast<std::size_t>(d));
                                   max_component           = std::max(max_component, std::abs(component));
                               }

                               const double indicator = max_component * dx;

                               if (allow_refine && indicator >= refine_threshold && level < max_l)
                               {
                                   tag[cell] = static_cast<int>(samurai::CellFlag::refine);
                               }
                               else if (allow_coarsen && indicator <= coarsen_threshold && level > min_l)
                               {
                                   tag[cell] = static_cast<int>(samurai::CellFlag::coarsen);
                               }
                               else
                               {
                                   tag[cell] = static_cast<int>(samurai::CellFlag::keep);
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

int main(int argc, char* argv[])
{
    auto& app = samurai::initialize("Finite volume example for the advection equation in 2d using AMR", argc, argv);

    constexpr std::size_t dim = 2;
    using Config              = samurai::amr::Config<dim, samurai::default_config::ghost_width, samurai::default_config::graduation_width, samurai::default_config::max_level, 0>;

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

    // AMR parameters
    std::size_t min_level            = 4;
    std::size_t max_level            = 10;
    std::size_t start_level          = max_level;
    double amr_refine_threshold      = 1e-4;
    double amr_coarsen_ratio         = 0.5;
    bool amr_allow_coarsen           = true;

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
    app.add_option("--start-level", start_level, "Start level of the AMR")->capture_default_str()->group("AMR");
    app.add_option("--min-level", min_level, "Minimum level of the AMR")->capture_default_str()->group("AMR");
    app.add_option("--max-level", max_level, "Maximum level of the AMR")->capture_default_str()->group("AMR");
    app.add_option("--amr-threshold", amr_refine_threshold, "Variation threshold triggering refinement")
        ->capture_default_str()
        ->group("AMR");
    app.add_option("--amr-coarsen-ratio",
                   amr_coarsen_ratio,
                   "Ratio applied to the refine threshold to trigger coarsening")
        ->capture_default_str()
        ->group("AMR");
    app.add_option("--amr-allow-coarsen", amr_allow_coarsen, "Allow coarsening during AMR adaptation")
        ->capture_default_str()
        ->group("AMR");
    app.add_option("--path", path, "Output path")->capture_default_str()->group("Output");
    app.add_option("--filename", filename, "File name prefix")->capture_default_str()->group("Output");
    app.add_option("--nfiles", nfiles, "Number of output files")->capture_default_str()->group("Output");

    SAMURAI_PARSE(argc, argv);

    max_level = std::max(max_level, min_level);
    start_level = std::clamp(start_level, min_level, max_level);
    amr_refine_threshold = std::max(amr_refine_threshold, 0.);
    amr_coarsen_ratio    = std::max(amr_coarsen_ratio, 0.);

    const samurai::Box<double, dim> box(min_corner, max_corner);
    samurai::amr::Mesh<Config> mesh;
    auto u = samurai::make_scalar_field<double>("u", mesh);

    if (restart_file.empty())
    {
        mesh = {box, start_level, min_level, max_level};
        init(u);
    }
    else
    {
        samurai::load(restart_file, mesh, u);
        min_level = mesh.min_level();
        max_level = mesh.max_level();
    }
    samurai::make_bc<samurai::Dirichlet<1>>(u, 0.);

    double dt            = cfl * mesh.cell_length(max_level);
    const double dt_save = Tf / static_cast<double>(nfiles);

    auto unp1 = samurai::make_scalar_field<double>("unp1", mesh);
    auto tag  = samurai::make_scalar_field<int>("tag", mesh);
    const xt::xtensor_fixed<int, xt::xshape<4, dim>> stencil_grad{
        {1, 0},
        {-1, 0},
        {0, 1},
        {0, -1}
    };

    auto adapt_mesh = [&]()
    {
        const auto run_round = [&](bool allow_coarsen_pass)
        {
            std::size_t adaptation_iteration = 0;
            constexpr std::size_t max_adaptation_iterations = 200;
            while (true)
            {
                std::cout << fmt::format("\tmesh adaptation ({}): {}",
                                         allow_coarsen_pass ? "coarsen" : "refine",
                                         adaptation_iteration++)
                          << std::endl;
                samurai::update_ghost(u);
                tag.resize();
                tag.fill(static_cast<int>(samurai::CellFlag::keep));
                AMR_criterion(u,
                              tag,
                              amr_refine_threshold,
                              amr_coarsen_ratio,
                              !allow_coarsen_pass,
                              allow_coarsen_pass && amr_allow_coarsen);
                samurai::graduation(tag, stencil_grad);
                if (samurai::update_field(tag, u))
                {
                    break;
                }
                if (adaptation_iteration >= max_adaptation_iterations)
                {
                    std::cout << "\tmaximum number of AMR adaptation iterations reached" << std::endl;
                    break;
                }
            }
        };

        run_round(false); // refine until stability
        if (amr_allow_coarsen)
        {
            run_round(true); // optional coarsening pass
        }
    };

    adapt_mesh();
    save(path, filename, u, "_init");

    std::size_t nsave = 1;
    std::size_t nt    = 0;

    while (t != Tf)
    {
        adapt_mesh();

        t += dt;
        if (t > Tf)
        {
            dt += Tf - t;
            t = Tf;
        }

        std::cout << fmt::format("iteration {}: t = {}, dt = {}", nt++, t, dt) << std::endl;

        samurai::update_ghost(u);
        unp1.resize();
        unp1 = u - dt * samurai::upwind(a, u);

        std::swap(u.array(), unp1.array());

        if (t >= static_cast<double>(nsave) * dt_save || t == Tf)
        {
            const std::string suffix = (nfiles != 1) ? fmt::format("_ite_{}", nsave++) : "";
            save(path, filename, u, suffix);
        }
    }
    samurai::finalize();
    return 0;
}
