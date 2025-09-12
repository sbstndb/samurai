#include <benchmark/benchmark.h>

#include "../../level_cell_array.hpp"
#include "../../subset/node.hpp"
#include <xtensor/xfixed.hpp>

#include <random>

namespace
{

    static samurai::LevelCellArray<2> make_full_slice_2d(std::size_t level, int min_coord, int max_coord)
    {
        samurai::LevelCellArray<2> lca(level);
        using value_t = typename samurai::LevelCellArray<2>::value_t;
        xt::xtensor_fixed<value_t, xt::xshape<1>> yz;
        for (int y = min_coord; y < max_coord; ++y)
        {
            yz[0] = y;
            lca.add_interval_back({min_coord, max_coord}, yz);
        }
        return lca;
    }

    static samurai::LevelCellArray<3> create_cube_lca(std::size_t level, int min_coord, int max_coord)
    {
        samurai::LevelCellArray<3> lca(level);
        using value_t = typename samurai::LevelCellArray<3>::value_t;
        xt::xtensor_fixed<value_t, xt::xshape<2>> yz;
        for (int z = min_coord; z < max_coord; ++z)
        {
            yz[1] = z; // yz = (y, z)
            for (int y = min_coord; y < max_coord; ++y)
            {
                yz[0] = y;
                lca.add_interval_back({min_coord, max_coord}, yz);
            }
        }
        return lca;
    }

    static samurai::LevelCellArray<3> create_fragmented_3d_lca(std::size_t level, int size, float density, unsigned seed)
    {
        samurai::LevelCellArray<3> lca(level);
        using value_t = typename samurai::LevelCellArray<3>::value_t;
        xt::xtensor_fixed<value_t, xt::xshape<2>> yz;

        std::mt19937 gen(seed);
        std::uniform_real_distribution<> dis(0.0, 1.0);

        for (int z = 0; z < size; ++z)
        {
            yz[1] = z;
            for (int y = 0; y < size; ++y)
            {
                if (dis(gen) < density)
                {
                    yz[0]      = y;
                    int last_x = -1;
                    for (int x = 0; x < size; ++x)
                    {
                        if (dis(gen) < density)
                        {
                            if (last_x == -1)
                            {
                                last_x = x;
                            }
                        }
                        else
                        {
                            if (last_x != -1)
                            {
                                lca.add_interval_back({last_x, x}, yz);
                                last_x = -1;
                            }
                        }
                    }
                    if (last_x != -1)
                    {
                        lca.add_interval_back({last_x, size}, yz);
                    }
                }
            }
        }
        return lca;
    }

} // namespace

static void BM_SAMURAI3D_Intersection_Solid(benchmark::State& state)
{
    int size = state.range(0);
    auto a   = create_cube_lca(2, 0, size);
    auto b   = create_cube_lca(2, size / 2, size + size / 2);
    for (auto _ : state)
    {
        auto subset       = samurai::intersection(a, b).on(a.level());
        volatile int flag = 0;
        samurai::apply(subset,
                       [&](const auto& interval, const auto& yz)
                       {
                           flag ^= (interval.end - interval.start) + yz[0] + yz[1];
                           return; // ignored
                       });
        benchmark::DoNotOptimize(flag);
    }
}

BENCHMARK(BM_SAMURAI3D_Intersection_Solid)->RangeMultiplier(2)->Range(16, 64);

static void BM_SAMURAI3D_Intersection_Fragmented(benchmark::State& state)
{
    int size = state.range(0);
    auto a   = create_fragmented_3d_lca(2, size, 0.2f, 100);
    auto b   = create_fragmented_3d_lca(2, size, 0.2f, 200);
    for (auto _ : state)
    {
        auto subset       = samurai::intersection(a, b).on(a.level());
        volatile int flag = 0;
        samurai::apply(subset,
                       [&](const auto& interval, const auto& yz)
                       {
                           flag ^= interval.start + yz[0] + yz[1];
                           return; // ignored
                       });
        benchmark::DoNotOptimize(flag);
    }
}

BENCHMARK(BM_SAMURAI3D_Intersection_Fragmented)->RangeMultiplier(2)->Range(16, 64);

static void BM_SAMURAI3D_ProjectUp(benchmark::State& state)
{
    int size = state.range(0);
    auto s   = create_cube_lca(2, 0, size);
    for (auto _ : state)
    {
        auto subset       = samurai::self(s).on(s.level() + 1);
        volatile int flag = 0;
        samurai::apply(subset,
                       [&](const auto& interval, const auto& yz)
                       {
                           flag ^= (interval.end - interval.start) + yz[0] + yz[1];
                           return; // ignored
                       });
        benchmark::DoNotOptimize(flag);
    }
}

BENCHMARK(BM_SAMURAI3D_ProjectUp)->RangeMultiplier(2)->Range(16, 64);

static void BM_SAMURAI3D_ProjectDown(benchmark::State& state)
{
    int size = state.range(0);
    auto s   = create_cube_lca(3, 0, size * 2);
    for (auto _ : state)
    {
        auto subset       = samurai::self(s).on(s.level() - 1);
        volatile int flag = 0;
        samurai::apply(subset,
                       [&](const auto& interval, const auto& yz)
                       {
                           flag ^= (interval.end - interval.start) + yz[0] + yz[1];
                           return; // ignored
                       });
        benchmark::DoNotOptimize(flag);
    }
}

BENCHMARK(BM_SAMURAI3D_ProjectDown)->RangeMultiplier(2)->Range(16, 64);

// Turbo combo: 4 sets with projection/difference/union/intersection using Samurai subset (3D)
static void BM_SAMURAI3D_TurboCombo(benchmark::State& state)
{
    int size = state.range(0);
    auto A   = create_fragmented_3d_lca(2, size, 0.15f, 11);
    auto B   = create_cube_lca(2, 0, size);
    auto C   = create_fragmented_3d_lca(2, size, 0.25f, 22);
    auto D   = create_cube_lca(2, size / 3, size + size / 3);

    xt::xtensor_fixed<int, xt::xshape<3>> t;
    t[0] = 2;
    t[1] = -1;
    t[2] = 3;

    for (auto _ : state)
    {
        auto A_proj       = samurai::self(A).on(2);
        auto D_proj       = samurai::self(D).on(2);
        auto C_t          = samurai::translate(C, t).on(2);
        auto B_minus_Ct   = samurai::difference(B, C_t).on(2);
        auto inter        = samurai::intersection(A_proj, B_minus_Ct).on(2);
        auto final_subset = samurai::union_(inter, D_proj).on(2);

        volatile int flag = 0;
        samurai::apply(final_subset,
                       [&](const auto& interval, const auto& yz)
                       {
                           flag ^= (interval.end - interval.start) + yz[0] + yz[1];
                           return; // force eval
                       });
        benchmark::DoNotOptimize(flag);
    }
}

BENCHMARK(BM_SAMURAI3D_TurboCombo)->RangeMultiplier(2)->Range(16, 64);
