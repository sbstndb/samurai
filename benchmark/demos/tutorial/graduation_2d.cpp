
#include <benchmark/benchmark.h>

#include <filesystem>

#include <xtensor/xfixed.hpp>
#include <xtensor/xmasked_view.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xview.hpp>

#include <samurai/box.hpp>
#include <samurai/cell_array.hpp>
#include <samurai/field.hpp>
#include <samurai/hdf5.hpp>
#include <samurai/samurai.hpp>
#include <samurai/subset/subset_op.hpp>

namespace fs = std::filesystem;

auto generate_mesh(std::size_t min_level, std::size_t max_level, std::size_t nsamples = 10000)
{
    constexpr std::size_t dim = 2;
    xt::random::seed(42);

    samurai::CellList<dim> cl;
    cl[0][{0}].add_point(0);

    for (std::size_t s = 0; s < nsamples; ++s)
    {
        auto level = xt::random::randint<std::size_t>({1}, min_level, max_level)[0];
        auto x     = xt::random::randint<int>({1}, 0, (1 << level) - 1)[0];
        auto y     = xt::random::randint<int>({1}, 0, (1 << level) - 1)[0];

        cl[level][{y}].add_point(x);
    }

    return samurai::CellArray<dim>(cl, true);
}

int graduation_2d(std::size_t max_level, std::size_t nsamples)
{
    samurai::initialize();

    constexpr std::size_t dim = 2;
    std::size_t min_level     = 1;
    bool with_corner          = false;

    // Output parameters
    fs::path path        = fs::current_path();
    std::string filename = "graduation_case_2";



    auto ca = generate_mesh(min_level, max_level, nsamples);


    std::size_t ite = 0;
    while (true)
    {
        auto tag = samurai::make_field<bool, 1>("tag", ca);
        tag.fill(false);

        for (std::size_t level = ca.min_level() + 1; level <= ca.max_level(); ++level)
        {
            for (std::size_t level_below = ca.min_level(); level_below < level; ++level_below)
            {
                auto set = samurai::intersection(ca[level], ca[level_below]).on(level_below);
                set(
                    [&](const auto& i, const auto& index)
                    {
                        tag(level_below, i, index[0]) = true;
                    });
            }
        }

        samurai::CellList<dim> cl;
        samurai::for_each_cell(ca,
                               [&](auto cell)
                               {
                                   auto i = cell.indices[0];
                                   auto j = cell.indices[1];
                                   if (tag[cell])
                                   {
                                       cl[cell.level + 1][{2 * j}].add_interval({2 * i, 2 * i + 2});
                                       cl[cell.level + 1][{2 * j + 1}].add_interval({2 * i, 2 * i + 2});
                                   }
                                   else
                                   {
                                       cl[cell.level][{j}].add_point(i);
                                   }
                               });
        samurai::CellArray<dim> new_ca = {cl, true};

        if (new_ca == ca)
        {
            break;
        }

        std::swap(ca, new_ca);
    }


    xt::xtensor_fixed<int, xt::xshape<4, dim>> stencil;
    if (with_corner)
    {
        stencil = {
            {1,  0 },
            {-1, 0 },
            {0,  1 },
            {0,  -1}
        };
    }
    else
    {
        stencil = {
            {1,  1 },
            {-1, -1},
            {-1, 1 },
            {1,  -1}
        };
    }

    ite = 0;
    while (true)
    {
        auto tag = samurai::make_field<bool, 1>("tag", ca);
        tag.fill(false);

        for (std::size_t level = ca.min_level() + 2; level <= ca.max_level(); ++level)
        {
            for (std::size_t level_below = ca.min_level(); level_below < level - 1; ++level_below)
            {
                for (std::size_t is = 0; is < stencil.shape()[0]; ++is)
                {
                    auto s   = xt::view(stencil, is);
                    auto set = samurai::intersection(samurai::translate(ca[level], s), ca[level_below]).on(level_below);
                    set(
                        [&](const auto& i, const auto& index)
                        {
                            tag(level_below, i, index[0]) = true;
                        });
                }
            }
        }

        samurai::CellList<dim> cl;
        samurai::for_each_cell(ca,
                               [&](auto cell)
                               {
                                   auto i = cell.indices[0];
                                   auto j = cell.indices[1];
                                   if (tag[cell])
                                   {
                                       cl[cell.level + 1][{2 * j}].add_interval({2 * i, 2 * i + 2});
                                       cl[cell.level + 1][{2 * j + 1}].add_interval({2 * i, 2 * i + 2});
                                   }
                                   else
                                   {
                                       cl[cell.level][{j}].add_point(i);
                                   }
                               });
        samurai::CellArray<dim> new_ca = {cl, true};

        if (new_ca == ca)
        {
            break;
        }

        std::swap(ca, new_ca);
    }

    samurai::finalize();
    return 0;
}

static void CASE_graduation_2d_medium(benchmark::State& state){
        for (auto _ : state){
                graduation_2d(14, 100000);
        }
}

static void CASE_graduation_2d_large(benchmark::State& state){
        for (auto _ : state){
                graduation_2d(16, 100000);
        }
}

BENCHMARK(CASE_graduation_2d_medium)
        ->Repetitions(1)
        ->ComputeStatistics("max", [](const std::vector<double>& v) -> double {
                return *(std::max_element(std::begin(v), std::end(v)));
        });

BENCHMARK(CASE_graduation_2d_large)
        ->Repetitions(1)
        ->ComputeStatistics("max", [](const std::vector<double>& v) -> double {
                return *(std::max_element(std::begin(v), std::end(v)));
        });
