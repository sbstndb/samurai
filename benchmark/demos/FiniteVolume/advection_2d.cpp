
#include <benchmark/benchmark.h>

#include <array>

#include <xtensor/xfixed.hpp>

#include <samurai/algorithm.hpp>
#include <samurai/bc.hpp>
#include <samurai/field.hpp>
#include <samurai/hdf5.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/samurai.hpp>
#include <samurai/stencil_field.hpp>
#include <samurai/subset/subset_op.hpp>

#include <filesystem>
namespace fs = std::filesystem;

template <class Mesh>
auto init(Mesh& mesh)
{
    auto u = samurai::make_field<double, 1>("u", mesh);

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

    return u;
}

template <class Field>
void flux_correction(double dt, const std::array<double, 2>& a, const Field& u, Field& unp1)
{
    using mesh_t              = typename Field::mesh_t;
    using mesh_id_t           = typename mesh_t::mesh_id_t;
    using interval_t          = typename mesh_t::interval_t;
    constexpr std::size_t dim = Field::dim;

    auto mesh = u.mesh();

    for (std::size_t level = mesh.min_level(); level < mesh.max_level(); ++level)
    {
        xt::xtensor_fixed<int, xt::xshape<dim>> stencil;

        stencil = {
            {-1, 0}
        };

        auto subset_right = samurai::intersection(samurai::translate(mesh[mesh_id_t::cells][level + 1], stencil),
                                                  mesh[mesh_id_t::cells][level])
                                .on(level);

        subset_right(
            [&](const auto& i, const auto& index)
            {
                auto j          = index[0];
                const double dx = samurai::cell_length(level);

                unp1(level, i, j) = unp1(level, i, j)
                                  + dt / dx
                                        * (samurai::upwind_op<dim, interval_t>(level, i, j).right_flux(a, u)
                                           - .5 * samurai::upwind_op<dim, interval_t>(level + 1, 2 * i + 1, 2 * j).right_flux(a, u)
                                           - .5 * samurai::upwind_op<dim, interval_t>(level + 1, 2 * i + 1, 2 * j + 1).right_flux(a, u));
            });

        stencil = {
            {1, 0}
        };

        auto subset_left = samurai::intersection(samurai::translate(mesh[mesh_id_t::cells][level + 1], stencil),
                                                 mesh[mesh_id_t::cells][level])
                               .on(level);

        subset_left(
            [&](const auto& i, const auto& index)
            {
                auto j          = index[0];
                const double dx = samurai::cell_length(level);

                unp1(level, i, j) = unp1(level, i, j)
                                  - dt / dx
                                        * (samurai::upwind_op<dim, interval_t>(level, i, j).left_flux(a, u)
                                           - .5 * samurai::upwind_op<dim, interval_t>(level + 1, 2 * i, 2 * j).left_flux(a, u)
                                           - .5 * samurai::upwind_op<dim, interval_t>(level + 1, 2 * i, 2 * j + 1).left_flux(a, u));
            });

        stencil = {
            {0, -1}
        };

        auto subset_up = samurai::intersection(samurai::translate(mesh[mesh_id_t::cells][level + 1], stencil), mesh[mesh_id_t::cells][level])
                             .on(level);

        subset_up(
            [&](const auto& i, const auto& index)
            {
                auto j          = index[0];
                const double dx = samurai::cell_length(level);

                unp1(level, i, j) = unp1(level, i, j)
                                  + dt / dx
                                        * (samurai::upwind_op<dim, interval_t>(level, i, j).up_flux(a, u)
                                           - .5 * samurai::upwind_op<dim, interval_t>(level + 1, 2 * i, 2 * j + 1).up_flux(a, u)
                                           - .5 * samurai::upwind_op<dim, interval_t>(level + 1, 2 * i + 1, 2 * j + 1).up_flux(a, u));
            });

        stencil = {
            {0, 1}
        };

        auto subset_down = samurai::intersection(samurai::translate(mesh[mesh_id_t::cells][level + 1], stencil),
                                                 mesh[mesh_id_t::cells][level])
                               .on(level);

        subset_down(
            [&](const auto& i, const auto& index)
            {
                auto j          = index[0];
                const double dx = samurai::cell_length(level);

                unp1(level, i, j) = unp1(level, i, j)
                                  - dt / dx
                                        * (samurai::upwind_op<dim, interval_t>(level, i, j).down_flux(a, u)
                                           - .5 * samurai::upwind_op<dim, interval_t>(level + 1, 2 * i, 2 * j).down_flux(a, u)
                                           - .5 * samurai::upwind_op<dim, interval_t>(level + 1, 2 * i + 1, 2 * j).down_flux(a, u));
            });
    }
}


int advection_2d(std::size_t max_level, double Tf)
{
    samurai::initialize();

    constexpr std::size_t dim = 2;
    using Config              = samurai::MRConfig<dim>;

    // Simulation parameters
    xt::xtensor_fixed<double, xt::xshape<dim>> min_corner = {0., 0.};
    xt::xtensor_fixed<double, xt::xshape<dim>> max_corner = {1., 1.};
    std::array<double, dim> a{
        {1, 1}
    };
    double cfl = 0.5;

    // Multiresolution parameters
    std::size_t min_level = 4;
    double mr_epsilon     = 2.e-4; // Threshold used by multiresolution
    double mr_regularity  = 1.;    // Regularity guess for multiresolution
    bool correction       = false;

    // Output parameters
    fs::path path        = fs::current_path();
    std::string filename = "FV_advection_2d";
    std::size_t nfiles   = 1;


    const samurai::Box<double, dim> box(min_corner, max_corner);
    samurai::MRMesh<Config> mesh{box, min_level, max_level};

    double dt            = cfl / (1 << max_level);
    const double dt_save = Tf / static_cast<double>(nfiles);
    double t             = 0.;

    auto u = init(mesh);
    samurai::make_bc<samurai::Dirichlet<1>>(u, 0.);
    auto unp1 = samurai::make_field<double, 1>("unp1", mesh);

    auto MRadaptation = samurai::make_MRAdapt(u);
    MRadaptation(mr_epsilon, mr_regularity);

    std::size_t nsave = 1;
    std::size_t nt    = 0;

    while (t != Tf)
    {
        MRadaptation(mr_epsilon, mr_regularity);

        t += dt;
        if (t > Tf)
        {
            dt += Tf - t;
            t = Tf;
        }


        samurai::update_ghost_mr(u);
        unp1.resize();
        unp1 = u - dt * samurai::upwind(a, u);
        if (correction)
        {
            flux_correction(dt, a, u, unp1);
        }

        std::swap(u.array(), unp1.array());

    }
    samurai::finalize();
    return 0;
}



static void CASE_advection_2d_little(benchmark::State& state){
        for (auto _ : state){
                advection_2d(7, 5.0);
        }
}

static void CASE_advection_2d_medium(benchmark::State& state){
        for (auto _ : state){
                advection_2d(10, 0.05);
        }
}

static void CASE_advection_2d_large(benchmark::State& state){
        for (auto _ : state){
                advection_2d(13, 0.003);
        }
}



BENCHMARK(CASE_advection_2d_little)
        ->Repetitions(1)
        ->ComputeStatistics("max", [](const std::vector<double>& v) -> double {
                return *(std::max_element(std::begin(v), std::end(v)));
        });


BENCHMARK(CASE_advection_2d_medium)
        ->Repetitions(1)
        ->ComputeStatistics("max", [](const std::vector<double>& v) -> double {
                return *(std::max_element(std::begin(v), std::end(v)));
        });

BENCHMARK(CASE_advection_2d_large)
        ->Repetitions(1)
        ->ComputeStatistics("max", [](const std::vector<double>& v) -> double {
                return *(std::max_element(std::begin(v), std::end(v)));
        });



