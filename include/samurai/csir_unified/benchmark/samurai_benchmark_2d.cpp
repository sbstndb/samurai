#include <benchmark/benchmark.h>

#include "../../level_cell_array.hpp"
#include "../../subset/node.hpp"
#include <random>
#include <xtensor/xfixed.hpp>

namespace
{

    static samurai::LevelCellArray<2> create_square_mesh_lca(int size, std::size_t level)
    {
        samurai::LevelCellArray<2> lca(level);
        using value_t = typename samurai::LevelCellArray<2>::value_t;
        xt::xtensor_fixed<value_t, xt::xshape<1>> yz;
        for (int y = 0; y < size; ++y)
        {
            yz[0] = y;
            lca.add_interval_back({0, size}, yz);
        }
        return lca;
    }

    static samurai::LevelCellArray<2> create_fragmented_mesh_lca(int size, float density, std::size_t level, unsigned seed)
    {
        samurai::LevelCellArray<2> lca(level);
        using value_t = typename samurai::LevelCellArray<2>::value_t;
        xt::xtensor_fixed<value_t, xt::xshape<1>> yz;

        std::mt19937 gen(seed);
        std::uniform_real_distribution<> dis(0.0, 1.0);

        for (int y = 0; y < size; ++y)
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
        return lca;
    }

    static samurai::LevelCellArray<2> create_checkerboard_mesh_lca(int size, std::size_t level)
    {
        samurai::LevelCellArray<2> lca(level);
        using value_t = typename samurai::LevelCellArray<2>::value_t;
        xt::xtensor_fixed<value_t, xt::xshape<1>> yz;
        for (int y = 0; y < size; ++y)
        {
            yz[0] = y;
            for (int x = 0; x < size; x += 2)
            {
                lca.add_interval_back({x, x + 1}, yz);
            }
        }
        return lca;
    }

} // namespace

static void BM_SAMURAI2D_Intersection_Solid(benchmark::State& state)
{
    int size = state.range(0);
    auto a   = create_square_mesh_lca(size, 5);
    auto b   = create_square_mesh_lca(size, 5);
    for (auto _ : state)
    {
        auto subset       = samurai::intersection(a, b).on(a.level());
        volatile int flag = 0;
        samurai::apply(subset,
                       [&](const auto& interval, const auto& yz)
                       {
                           flag ^= (interval.end - interval.start) + yz[0];
                           return; // ignored
                       });
        benchmark::DoNotOptimize(flag);
    }
}

BENCHMARK(BM_SAMURAI2D_Intersection_Solid)->RangeMultiplier(2)->Range(64, 512);

static void BM_SAMURAI2D_Intersection_Fragmented(benchmark::State& state)
{
    int size = state.range(0);
    auto a   = create_fragmented_mesh_lca(size, 0.2f, 5, 1337);
    auto b   = create_fragmented_mesh_lca(size, 0.2f, 5, 4242);
    for (auto _ : state)
    {
        auto subset       = samurai::intersection(a, b).on(a.level());
        volatile int flag = 0;
        samurai::apply(subset,
                       [&](const auto& interval, const auto& yz)
                       {
                           flag ^= interval.start + yz[0];
                           return; // ignored
                       });
        benchmark::DoNotOptimize(flag);
    }
}

BENCHMARK(BM_SAMURAI2D_Intersection_Fragmented)->RangeMultiplier(2)->Range(64, 512);

static void BM_SAMURAI2D_Intersection_Checkerboard(benchmark::State& state)
{
    int size = state.range(0);
    auto a   = create_checkerboard_mesh_lca(size, 5);
    auto b   = create_checkerboard_mesh_lca(size, 5);
    for (auto _ : state)
    {
        auto subset       = samurai::intersection(a, b).on(a.level());
        volatile int flag = 0;
        samurai::apply(subset,
                       [&](const auto& interval, const auto& yz)
                       {
                           flag ^= interval.end + yz[0];
                           return; // ignored
                       });
        benchmark::DoNotOptimize(flag);
    }
}

BENCHMARK(BM_SAMURAI2D_Intersection_Checkerboard)->RangeMultiplier(2)->Range(64, 512);

static void BM_SAMURAI2D_ProjectUp(benchmark::State& state)
{
    int size = state.range(0);
    auto s   = create_checkerboard_mesh_lca(size, 4);
    for (auto _ : state)
    {
        auto subset       = samurai::self(s).on(s.level() + 1);
        volatile int flag = 0;
        samurai::apply(subset,
                       [&](const auto& interval, const auto& yz)
                       {
                           flag ^= (interval.end - interval.start) + yz[0];
                           return; // ignored
                       });
        benchmark::DoNotOptimize(flag);
    }
}

BENCHMARK(BM_SAMURAI2D_ProjectUp)->RangeMultiplier(2)->Range(64, 512);

static void BM_SAMURAI2D_ProjectDown(benchmark::State& state)
{
    int size = state.range(0);
    auto s   = create_checkerboard_mesh_lca(size, 5);
    for (auto _ : state)
    {
        auto subset       = samurai::self(s).on(s.level() - 1);
        volatile int flag = 0;
        samurai::apply(subset,
                       [&](const auto& interval, const auto& yz)
                       {
                           flag ^= (interval.end - interval.start) + yz[0];
                           return; // ignored
                       });
        benchmark::DoNotOptimize(flag);
    }
}

BENCHMARK(BM_SAMURAI2D_ProjectDown)->RangeMultiplier(2)->Range(64, 512);

// Turbo combo: 4 sets with projection/difference/union/intersection using Samurai subset
static void BM_SAMURAI2D_TurboCombo(benchmark::State& state)
{
    int size = state.range(0);
    auto A   = create_fragmented_mesh_lca(size, 0.15f, 5, 111);
    auto B   = create_checkerboard_mesh_lca(size, 5);
    auto C   = create_square_mesh_lca(size, 5);
    auto D   = create_fragmented_mesh_lca(size, 0.25f, 5, 222);

    xt::xtensor_fixed<int, xt::xshape<2>> t;
    t[0] = 3;
    t[1] = -2;

    for (auto _ : state)
    {
        auto A_proj       = samurai::self(A).on(5);
        auto D_proj       = samurai::self(D).on(5);
        auto C_t          = samurai::translate(C, t).on(5);
        auto B_minus_Ct   = samurai::difference(B, C_t).on(5);
        auto inter        = samurai::intersection(A_proj, B_minus_Ct).on(5);
        auto final_subset = samurai::union_(inter, D_proj).on(5);

        volatile int flag = 0;
        samurai::apply(final_subset,
                       [&](const auto& interval, const auto& yz)
                       {
                           flag ^= (interval.end - interval.start) + yz[0];
                           return; // force eval
                       });
        benchmark::DoNotOptimize(flag);
    }
}

BENCHMARK(BM_SAMURAI2D_TurboCombo)->RangeMultiplier(2)->Range(64, 512);
