#include "src/csir.hpp"
#include <algorithm>
#include <benchmark/benchmark.h>
#include <random>

namespace
{

    static csir::CSIR_Level_1D
    make_random_1d(std::size_t level, int x_min, int x_max, int n_intervals, int min_len = 2, int max_len = 15, unsigned seed = 42)
    {
        using namespace csir;
        std::mt19937 rng(seed);
        std::uniform_int_distribution<int> dist_start(x_min, std::max(x_min, x_max - min_len - 1));
        std::uniform_int_distribution<int> dist_len(min_len, max_len);

        std::vector<Interval> tmp;
        tmp.reserve(n_intervals);
        for (int k = 0; k < n_intervals; ++k)
        {
            int s = dist_start(rng);
            int e = std::min(x_max, s + dist_len(rng));
            if (s < e)
            {
                tmp.push_back({s, e});
            }
        }
        // tri + fusion
        std::sort(tmp.begin(),
                  tmp.end(),
                  [](const Interval& a, const Interval& b)
                  {
                      return a.start < b.start;
                  });
        std::vector<Interval> merged;
        for (auto itv : tmp)
        {
            if (merged.empty() || itv.start > merged.back().end)
            {
                merged.push_back(itv);
            }
            else
            {
                merged.back().end = std::max(merged.back().end, itv.end);
            }
        }
        csir::CSIR_Level_1D out;
        out.level     = level;
        out.intervals = std::move(merged);
        return out;
    }

}

static void BM_CSIR1D_Union(benchmark::State& state)
{
    int n  = state.range(0);
    auto a = make_random_1d(6, 0, n * 5, n, 2, 5, 1337);
    auto b = make_random_1d(6, 0, n * 5, n, 2, 5, 4242);
    for (auto _ : state)
    {
        auto u = csir::union_(a, b);
        benchmark::DoNotOptimize(u);
    }
}

BENCHMARK(BM_CSIR1D_Union)->RangeMultiplier(2)->Range(1 << 10, 1 << 17);

static void BM_CSIR1D_Intersection(benchmark::State& state)
{
    int n  = state.range(0);
    auto a = make_random_1d(6, 0, n * 5, n, 2, 5, 1337);
    auto b = make_random_1d(6, 0, n * 5, n, 2, 5, 4242);
    for (auto _ : state)
    {
        auto i = csir::intersection(a, b);
        benchmark::DoNotOptimize(i);
    }
}

BENCHMARK(BM_CSIR1D_Intersection)->RangeMultiplier(2)->Range(1 << 10, 1 << 17);

static void BM_CSIR1D_Difference(benchmark::State& state)
{
    int n  = state.range(0);
    auto a = make_random_1d(6, 0, n * 5, n, 2, 5, 1337);
    auto b = make_random_1d(6, 0, n * 5, n, 2, 5, 4242);
    for (auto _ : state)
    {
        auto d = csir::difference(a, b);
        benchmark::DoNotOptimize(d);
    }
}

BENCHMARK(BM_CSIR1D_Difference)->RangeMultiplier(2)->Range(1 << 10, 1 << 17);

static void BM_CSIR1D_Translate(benchmark::State& state)
{
    int n  = state.range(0);
    auto s = make_random_1d(6, 0, n * 5, n, 2, 5, 1337);
    for (auto _ : state)
    {
        auto t = csir::translate(s, 10);
        benchmark::DoNotOptimize(t);
    }
}

BENCHMARK(BM_CSIR1D_Translate)->RangeMultiplier(2)->Range(1 << 10, 1 << 17);

static void BM_CSIR1D_ProjectUp(benchmark::State& state)
{
    int n  = state.range(0);
    auto s = make_random_1d(4, 0, n * 5, n, 2, 5, 1337);
    for (auto _ : state)
    {
        auto p = csir::project_to_level(s, s.level + 2);
        benchmark::DoNotOptimize(p);
    }
}

BENCHMARK(BM_CSIR1D_ProjectUp)->RangeMultiplier(2)->Range(1 << 10, 1 << 17);

static void BM_CSIR1D_ProjectDown(benchmark::State& state)
{
    int n  = state.range(0);
    auto s = make_random_1d(6, 0, n * 5, n, 2, 5, 1337);
    for (auto _ : state)
    {
        auto p = csir::project_to_level(s, s.level - 2);
        benchmark::DoNotOptimize(p);
    }
}

BENCHMARK(BM_CSIR1D_ProjectDown)->RangeMultiplier(2)->Range(1 << 10, 1 << 17);
