#include <array>
#include <benchmark/benchmark.h>
#include <experimental/random>

#include <xtensor/xfixed.hpp>
#include <xtensor/xrandom.hpp>

#include <samurai/algorithm.hpp>
#include <samurai/box.hpp>
#include <samurai/cell_array.hpp>
#include <samurai/cell_list.hpp>
#include <samurai/field.hpp>
#include <samurai/list_of_intervals.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/static_algorithm.hpp>
#include <samurai/uniform_mesh.hpp>

// Observation : Cela prend environ 10ns par intervalle
// si on compare 2 intervalles de taillen, cela prendra environ 2n * 10ns

///////////////////////////////////////////////////////////////////
// Fonctions utilitaires pour la génération d'intervalles
///////////////////////////////////////////////////////////////////

template <unsigned int delta_level>
auto gen_same_intervals = [](auto& cl1, auto& cl2, int index, unsigned int)
{
    cl1[0][{}].add_interval({2 * index, 2 * index + 1});
    cl2[delta_level][{}].add_interval({pow(2, delta_level + 1) * index, pow(2, delta_level + 1) * index + pow(2, delta_level)});
};

template <unsigned int delta_level>
auto gen_different_intervals = [](auto& cl1, auto& cl2, int index, unsigned int)
{
    cl1[0][{}].add_interval({2 * index, 2 * index + 1});
    cl2[delta_level][{}].add_interval({pow(2, delta_level + 1) * index + pow(2, delta_level), pow(2, delta_level + 2) * index});
};

template <unsigned int delta_level>
auto gen_n1_intervals = [](auto& cl1, auto& cl2, int index, unsigned int)
{
    cl1[0][{}].add_interval({2 * index, 2 * index + 1});
    if (index == 0)
    {
        cl2[delta_level][{}].add_interval({static_cast<int>(0), static_cast<int>(pow(2, delta_level))});
    }
};

///////////////////////////////////////////////////////////////////
// Opérations ensemblistes
///////////////////////////////////////////////////////////////////

auto op_difference = [](const auto& a, const auto& b)
{
    return samurai::difference(a, b);
};

auto op_intersection = [](const auto& a, const auto& b)
{
    return samurai::intersection(a, b);
};

auto op_union = [](const auto& a, const auto& b)
{
    return samurai::union_(a, b);
};

///////////////////////////////////////////////////////////////////
// Fonction de benchmark unifiée
///////////////////////////////////////////////////////////////////

template <unsigned int dim, unsigned int delta_level, typename IntervalGenerator, typename Operation>
void SUBSET_unified_benchmark(benchmark::State& state, IntervalGenerator&& gen_intervals, Operation&& operation)
{
    samurai::CellList<dim> cl1, cl2;
    for (int64_t i = 0; i < state.range(0); i++)
    {
        int index = static_cast<int>(i);
        gen_intervals(cl1, cl2, index, delta_level);
    }
    samurai::CellArray<dim> ca1(cl1);
    samurai::CellArray<dim> ca2(cl2);
    for (auto _ : state)
    {
        auto total_cells = 0;
        if constexpr (delta_level == 0)
        {
            auto subset = operation(ca1[0], ca2[0]);
            subset(
                [&total_cells](const auto&, const auto&)
                {
                    total_cells = 1;
                });
            benchmark::DoNotOptimize(total_cells);
            benchmark::DoNotOptimize(subset);
        }
    }
}

///////////////////////////////////////////////////////////////////
// Benchmarks pour les opérations ensemblistes
///////////////////////////////////////////////////////////////////

template <unsigned int dim, unsigned int delta_level>
void SUBSET_set_diff_identical(benchmark::State& state)
{
    SUBSET_unified_benchmark<dim, delta_level>(state, gen_same_intervals<delta_level>, op_difference);
}

template <unsigned int dim, unsigned int delta_level>
void SUBSET_set_diff_disjoint(benchmark::State& state)
{
    SUBSET_unified_benchmark<dim, delta_level>(state, gen_different_intervals<delta_level>, op_difference);
}

template <unsigned int dim, unsigned int delta_level>
void SUBSET_set_diff_single(benchmark::State& state)
{
    SUBSET_unified_benchmark<dim, delta_level>(state, gen_n1_intervals<delta_level>, op_difference);
}

template <unsigned int dim, unsigned int delta_level>
void SUBSET_set_intersect_identical(benchmark::State& state)
{
    SUBSET_unified_benchmark<dim, delta_level>(state, gen_same_intervals<delta_level>, op_intersection);
}

template <unsigned int dim, unsigned int delta_level>
void SUBSET_set_intersect_disjoint(benchmark::State& state)
{
    SUBSET_unified_benchmark<dim, delta_level>(state, gen_different_intervals<delta_level>, op_intersection);
}

template <unsigned int dim, unsigned int delta_level>
void SUBSET_set_intersect_single(benchmark::State& state)
{
    SUBSET_unified_benchmark<dim, delta_level>(state, gen_n1_intervals<delta_level>, op_intersection);
}

template <unsigned int dim, unsigned int delta_level>
void SUBSET_set_union_identical(benchmark::State& state)
{
    SUBSET_unified_benchmark<dim, delta_level>(state, gen_same_intervals<delta_level>, op_union);
}

template <unsigned int dim, unsigned int delta_level>
void SUBSET_set_union_disjoint(benchmark::State& state)
{
    SUBSET_unified_benchmark<dim, delta_level>(state, gen_different_intervals<delta_level>, op_union);
}

template <unsigned int dim, unsigned int delta_level>
void SUBSET_set_union_single(benchmark::State& state)
{
    SUBSET_unified_benchmark<dim, delta_level>(state, gen_n1_intervals<delta_level>, op_union);
}

///////////////////////////////////////////////////////////////////
// Benchmarks pour les opérations géométriques
///////////////////////////////////////////////////////////////////

template <unsigned int dim>
void SUBSET_translate(benchmark::State& state)
{
    samurai::CellList<dim> cl;
    for (int64_t i = 0; i < state.range(0); i++)
    {
        int index = static_cast<int>(i);
        cl[0][{}].add_interval({2 * index, 2 * index + 1});
    }
    samurai::CellArray<dim> ca(cl);
    xt::xtensor_fixed<int, xt::xshape<dim>> stencil;
    if constexpr (dim == 1)
    {
        stencil = xt::xtensor_fixed<int, xt::xshape<1>>({1});
    }
    else if constexpr (dim == 2)
    {
        stencil = xt::xtensor_fixed<int, xt::xshape<2>>({1, 1});
    }
    else if constexpr (dim == 3)
    {
        stencil = xt::xtensor_fixed<int, xt::xshape<3>>({1, 1, 1});
    }
    for (auto _ : state)
    {
        auto total_cells = 0;
        auto subset      = samurai::translate(ca[0], stencil);
        subset(
            [&total_cells](const auto&, const auto&)
            {
                total_cells = 1;
            });
        benchmark::DoNotOptimize(total_cells);
        benchmark::DoNotOptimize(subset);
    }
}

///////////////////////////////////////////////////////////////////
// Enregistrement des benchmarks
///////////////////////////////////////////////////////////////////

// Benchmarks pour les opérations ensemblistes en 1D
BENCHMARK_TEMPLATE(SUBSET_set_diff_identical, 1, 0)->RangeMultiplier(8)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(SUBSET_set_diff_disjoint, 1, 0)->RangeMultiplier(8)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(SUBSET_set_diff_single, 1, 0)->RangeMultiplier(8)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(SUBSET_set_intersect_identical, 1, 0)->RangeMultiplier(8)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(SUBSET_set_intersect_disjoint, 1, 0)->RangeMultiplier(8)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(SUBSET_set_intersect_single, 1, 0)->RangeMultiplier(8)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(SUBSET_set_union_identical, 1, 0)->RangeMultiplier(8)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(SUBSET_set_union_disjoint, 1, 0)->RangeMultiplier(8)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(SUBSET_set_union_single, 1, 0)->RangeMultiplier(8)->Range(1 << 1, 1 << 10);

// Benchmarks pour les opérations ensemblistes en 2D
BENCHMARK_TEMPLATE(SUBSET_set_diff_identical, 2, 0)->RangeMultiplier(8)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(SUBSET_set_diff_disjoint, 2, 0)->RangeMultiplier(8)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(SUBSET_set_diff_single, 2, 0)->RangeMultiplier(8)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(SUBSET_set_intersect_identical, 2, 0)->RangeMultiplier(8)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(SUBSET_set_intersect_disjoint, 2, 0)->RangeMultiplier(8)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(SUBSET_set_intersect_single, 2, 0)->RangeMultiplier(8)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(SUBSET_set_union_identical, 2, 0)->RangeMultiplier(8)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(SUBSET_set_union_disjoint, 2, 0)->RangeMultiplier(8)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(SUBSET_set_union_single, 2, 0)->RangeMultiplier(8)->Range(1 << 1, 1 << 10);

// Benchmarks pour les opérations ensemblistes en 3D
BENCHMARK_TEMPLATE(SUBSET_set_diff_identical, 3, 0)->RangeMultiplier(8)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(SUBSET_set_diff_disjoint, 3, 0)->RangeMultiplier(8)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(SUBSET_set_diff_single, 3, 0)->RangeMultiplier(8)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(SUBSET_set_intersect_identical, 3, 0)->RangeMultiplier(8)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(SUBSET_set_intersect_disjoint, 3, 0)->RangeMultiplier(8)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(SUBSET_set_intersect_single, 3, 0)->RangeMultiplier(8)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(SUBSET_set_union_identical, 3, 0)->RangeMultiplier(8)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(SUBSET_set_union_disjoint, 3, 0)->RangeMultiplier(8)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(SUBSET_set_union_single, 3, 0)->RangeMultiplier(8)->Range(1 << 1, 1 << 10);

// Benchmarks pour les opérations géométriques
BENCHMARK_TEMPLATE(SUBSET_translate, 1)->RangeMultiplier(8)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(SUBSET_translate, 2)->RangeMultiplier(8)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(SUBSET_translate, 3)->RangeMultiplier(8)->Range(1 << 1, 1 << 10);
