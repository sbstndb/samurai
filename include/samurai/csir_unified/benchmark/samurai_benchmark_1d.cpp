#include <benchmark/benchmark.h>

#include <random>
#include <algorithm>

#include "../../level_cell_array.hpp"
#include "../../subset/node.hpp"
#include <xtensor/xfixed.hpp>

namespace {

// Build a random 1D LevelCellArray at a given level
static samurai::LevelCellArray<1> make_random_lca_1d(std::size_t level,
                                                     int x_min,
                                                     int x_max,
                                                     int n_intervals,
                                                     int min_len = 2,
                                                     int max_len = 15,
                                                     unsigned seed = 42)
{
    using interval_t = typename samurai::LevelCellArray<1>::interval_t;
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist_start(x_min, std::max(x_min, x_max - min_len - 1));
    std::uniform_int_distribution<int> dist_len(min_len, max_len);

    // Generate, sort and merge random intervals
    std::vector<interval_t> tmp;
    tmp.reserve(static_cast<std::size_t>(n_intervals));
    for (int k = 0; k < n_intervals; ++k)
    {
        int s = dist_start(rng);
        int e = std::min(x_max, s + dist_len(rng));
        if (s < e) tmp.emplace_back(s, e);
    }
    std::sort(tmp.begin(), tmp.end(), [](const interval_t& a, const interval_t& b){ return a.start < b.start; });

    std::vector<interval_t> merged;
    merged.reserve(tmp.size());
    for (auto itv : tmp)
    {
        if (merged.empty() || itv.start > merged.back().end) merged.push_back(itv);
        else merged.back().end = std::max(merged.back().end, itv.end);
    }

    samurai::LevelCellArray<1> lca(level);
    xt::xtensor_fixed<int, xt::xshape<0>> yz; // empty yz for dim=1
    for (const auto& itv : merged)
    {
        lca.add_interval_back(itv, yz);
    }
    return lca;
}

} // namespace

static void BM_SAMURAI1D_Union(benchmark::State& state)
{
    int n = state.range(0);
    auto a = make_random_lca_1d(6, 0, n * 5, n, 2, 5, 1337);
    auto b = make_random_lca_1d(6, 0, n * 5, n, 2, 5, 4242);
    for (auto _ : state)
    {
        auto subset = samurai::union_(a, b).on(a.level());
        volatile int flag = 0;
        samurai::apply(subset, [&](const auto& interval, const auto&) {
            flag ^= (interval.end - interval.start);
            return; // ignored
        });
        benchmark::DoNotOptimize(flag);
    }
}
BENCHMARK(BM_SAMURAI1D_Union)->RangeMultiplier(2)->Range(1<<10, 1<<17);

static void BM_SAMURAI1D_Intersection(benchmark::State& state)
{
    int n = state.range(0);
    auto a = make_random_lca_1d(6, 0, n * 5, n, 2, 5, 1337);
    auto b = make_random_lca_1d(6, 0, n * 5, n, 2, 5, 4242);
    for (auto _ : state)
    {
        auto subset = samurai::intersection(a, b).on(a.level());
        volatile int flag = 0;
        samurai::apply(subset, [&](const auto& interval, const auto&) {
            flag ^= interval.start;
            return; // ignored
        });
        benchmark::DoNotOptimize(flag);
    }
}
BENCHMARK(BM_SAMURAI1D_Intersection)->RangeMultiplier(2)->Range(1<<10, 1<<17);

static void BM_SAMURAI1D_Difference(benchmark::State& state)
{
    int n = state.range(0);
    auto a = make_random_lca_1d(6, 0, n * 5, n, 2, 5, 1337);
    auto b = make_random_lca_1d(6, 0, n * 5, n, 2, 5, 4242);
    for (auto _ : state)
    {
        auto subset = samurai::difference(a, b).on(a.level());
        volatile int flag = 0;
        samurai::apply(subset, [&](const auto& interval, const auto&) {
            flag ^= interval.end;
            return; // ignored
        });
        benchmark::DoNotOptimize(flag);
    }
}
BENCHMARK(BM_SAMURAI1D_Difference)->RangeMultiplier(2)->Range(1<<10, 1<<17);

static void BM_SAMURAI1D_Translate(benchmark::State& state)
{
    int n = state.range(0);
    auto s = make_random_lca_1d(6, 0, n * 5, n, 2, 5, 1337);
    for (auto _ : state)
    {
        xt::xtensor_fixed<int, xt::xshape<1>> t; t[0] = 10;
        auto subset = samurai::translate(s, t).on(s.level());
        volatile int flag = 0;
        samurai::apply(subset, [&](const auto& interval, const auto&) {
            flag ^= interval.start;
            return; // ignored
        });
        benchmark::DoNotOptimize(flag);
    }
}
BENCHMARK(BM_SAMURAI1D_Translate)->RangeMultiplier(2)->Range(1<<10, 1<<17);

static void BM_SAMURAI1D_ProjectUp(benchmark::State& state)
{
    int n = state.range(0);
    auto s = make_random_lca_1d(4, 0, n * 5, n, 2, 5, 1337);
    for (auto _ : state)
    {
        auto subset = samurai::self(s).on(s.level() + 2);
        volatile int flag = 0;
        samurai::apply(subset, [&](const auto& interval, const auto&) {
            flag ^= (interval.end - interval.start);
            return; // ignored
        });
        benchmark::DoNotOptimize(flag);
    }
}
BENCHMARK(BM_SAMURAI1D_ProjectUp)->RangeMultiplier(2)->Range(1<<10, 1<<17);

static void BM_SAMURAI1D_ProjectDown(benchmark::State& state)
{
    int n = state.range(0);
    auto s = make_random_lca_1d(6, 0, n * 5, n, 2, 5, 1337);
    for (auto _ : state)
    {
        auto subset = samurai::self(s).on(s.level() - 2);
        volatile int flag = 0;
        samurai::apply(subset, [&](const auto& interval, const auto&) {
            flag ^= interval.start;
            return; // ignored
        });
        benchmark::DoNotOptimize(flag);
    }
}
BENCHMARK(BM_SAMURAI1D_ProjectDown)->RangeMultiplier(2)->Range(1<<10, 1<<17);
