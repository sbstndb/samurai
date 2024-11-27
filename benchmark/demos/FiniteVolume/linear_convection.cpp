
#include <benchmark/benchmark.h>

#include <samurai/hdf5.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>
//#include <samurai/petsc.hpp>
#include <samurai/schemes/fv.hpp>
#include <samurai/samurai.hpp>

#include <filesystem>
namespace fs = std::filesystem;


int linear_convection(std::size_t max_level, double Tf) 
{

    char program_name[] = "program_name";
    char* argv[] = { program_name, nullptr };	
    samurai::initialize();

    static constexpr std::size_t dim = 2;
    using Config                     = samurai::MRConfig<dim, 1>;
    using Box                        = samurai::Box<double, dim>;
    using point_t                    = typename Box::point_t;

    //--------------------//
    // Program parameters //
    //--------------------//

    // Simulation parameters
    double left_box  = -1;
    double right_box = 1;

    // Time integration
//    double Tf  = 0.05;
    double dt  = 0;
    double cfl = 0.95;

    // Multiresolution parameters
    std::size_t min_level = 1;
//    std::size_t max_level = 10;
    double mr_epsilon     = 1e-4; // Threshold used by multiresolution
    double mr_regularity  = 1.;   // Regularity guess for multiresolution

    // Output parameters
    fs::path path        = fs::current_path();
    std::string filename = "linear_convection_" + std::to_string(dim) + "D";
    std::size_t nfiles   = 0;

    //--------------------//
    // Problem definition //
    //--------------------//

    point_t box_corner1, box_corner2;
    box_corner1.fill(left_box);
    box_corner2.fill(right_box);
    Box box(box_corner1, box_corner2);
    std::array<bool, dim> periodic;
    periodic.fill(true);
    samurai::MRMesh<Config> mesh{box, min_level, max_level, periodic};

    // Initial solution
    auto u = samurai::make_field<1>("u",
                                    mesh,
                                    [](const auto& coords)
                                    {
                                        if constexpr (dim == 1)
                                        {
                                            auto& x = coords(0);
                                            return (x >= -0.8 && x <= -0.3) ? 1. : 0.;
                                        }
                                        else
                                        {
                                            auto& x = coords(0);
                                            auto& y = coords(1);
                                            return (x >= -0.8 && x <= -0.3 && y >= 0.3 && y <= 0.8) ? 1. : 0.;
                                        }
                                    });

    auto unp1 = samurai::make_field<1>("unp1", mesh);
    // Intermediary fields for the RK3 scheme
    auto u1 = samurai::make_field<1>("u1", mesh);
    auto u2 = samurai::make_field<1>("u2", mesh);

    unp1.fill(0);
    u1.fill(0);
    u2.fill(0);

    // Convection operator
    samurai::VelocityVector<dim> velocity;
    velocity.fill(1);
    if constexpr (dim == 2)
    {
        velocity(1) = -1;
    }
    auto conv = samurai::make_convection_upwind<decltype(u)>(velocity);

    //--------------------//
    //   Time iteration   //
    //--------------------//

    if (dt == 0)
    {
        double dx             = samurai::cell_length(max_level);
        auto a                = xt::abs(velocity);
        double sum_velocities = xt::sum(xt::abs(velocity))();
        dt                    = cfl * dx / sum_velocities;
    }

    auto MRadaptation = samurai::make_MRAdapt(u);
    MRadaptation(mr_epsilon, mr_regularity);

    double t = 0;
    while (t != Tf)
    {
        // Move to next timestep
        t += dt;
        if (t > Tf)
        {
            dt += Tf - t;
            t = Tf;
        }

        // Mesh adaptation
        MRadaptation(mr_epsilon, mr_regularity);
        samurai::update_ghost_mr(u);
        unp1.resize();
        u1.resize();
        u2.resize();
        u1.fill(0);
        u2.fill(0);

        unp1 = u - dt * conv(u);

        // TVD-RK3 (SSPRK3)
        //u1 = u - dt * conv(u);
        //samurai::update_ghost_mr(u1);
        //u2 = 3. / 4 * u + 1. / 4 * (u1 - dt * conv(u1));
        //samurai::update_ghost_mr(u2);
        //unp1 = 1. / 3 * u + 2. / 3 * (u2 - dt * conv(u2));

        // u <-- unp1
        std::swap(u.array(), unp1.array());
//	benchmark::DoNotOptimize(u);
    }

    samurai::finalize();
    return 0;
}



static void CASE_linear_convection_little(benchmark::State& state){
	for (auto _ : state){
		linear_convection(7, 2.0);
	}
}

static void CASE_linear_convection_medium(benchmark::State& state){
        for (auto _ : state){
                linear_convection(10, 0.05);
	}
}

static void CASE_linear_convection_large(benchmark::State& state){
        for (auto _ : state){
                linear_convection(13, 0.001);
        }
}




BENCHMARK(CASE_linear_convection_little)
	->Repetitions(1)
	->ComputeStatistics("max", [](const std::vector<double>& v) -> double {
		return *(std::max_element(std::begin(v), std::end(v)));
	});


BENCHMARK(CASE_linear_convection_medium)
        ->Repetitions(1)
        ->ComputeStatistics("max", [](const std::vector<double>& v) -> double {
                return *(std::max_element(std::begin(v), std::end(v)));
	});

BENCHMARK(CASE_linear_convection_large)
        ->Repetitions(1)
        ->ComputeStatistics("max", [](const std::vector<double>& v) -> double {
                return *(std::max_element(std::begin(v), std::end(v)));
        });


