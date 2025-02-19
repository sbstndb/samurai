#include <array>
#include <benchmark/benchmark.h>
#include <experimental/random>

#include <xtensor/xfixed.hpp>
#include <xtensor/xrandom.hpp>

#include <samurai/algorithm.hpp>
#include <samurai/cell_array.hpp>
#include <samurai/cell_list.hpp>
#include <samurai/static_algorithm.hpp>
#include <samurai/list_of_intervals.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/box.hpp>
#include <samurai/field.hpp>
#include <samurai/uniform_mesh.hpp>



template <unsigned int dim>
void LEVELCELLARRAY_default(benchmark::State& state){
	using TInterval = samurai::default_config::interval_t;
        for (auto _ : state){
		auto lcl = samurai::LevelCellArray<dim, TInterval>() ; 
                benchmark::DoNotOptimize(lcl);
        }
}

template <unsigned int dim>
void LEVELCELLARRAY_empty_lcl_to_lca(benchmark::State& state){
        using TInterval = samurai::default_config::interval_t;
	samurai::LevelCellList<dim> lcl ;
        for (auto _ : state){
                auto lca = samurai::LevelCellArray<dim, TInterval>(lcl) ;
                benchmark::DoNotOptimize(lca);
        }
}


template <unsigned int dim>
void LEVELCELLARRAY_lcl_to_lca(benchmark::State& state){
	samurai::LevelCellList<dim> lcl;
        using TInterval = samurai::default_config::interval_t;
	for (int64_t i = 0 ; i < state.range(0) ; i++){
		int index = static_cast<int>(i) ; 
		lcl[{0}].add_interval({2*index,2*index +1});
	}
        for (auto _ : state){
                auto lca = samurai::LevelCellArray<dim, TInterval>(lcl) ;
                benchmark::DoNotOptimize(lca);
        }
}


template <unsigned int dim>
void LEVELCELLARRAY_begin(benchmark::State& state){
        samurai::LevelCellList<dim> lcl;
        using TInterval = samurai::default_config::interval_t;
        for (int64_t i = 0 ; i < state.range(0) ; i++){
                int index = static_cast<int>(i) ;
                lcl[{0}].add_interval({2*index,2*index +1});
        }
        auto lca = samurai::LevelCellArray<dim, TInterval>(lcl) ;
        for (auto _ : state){
		auto begin = lca.begin() ; 
                benchmark::DoNotOptimize(begin);
        }
}

template <unsigned int dim>
void LEVELCELLARRAY_end(benchmark::State& state){
        samurai::LevelCellList<dim> lcl;
        using TInterval = samurai::default_config::interval_t;
        for (int64_t i = 0 ; i < state.range(0) ; i++){
                int index = static_cast<int>(i) ;
                lcl[{0}].add_interval({2*index,2*index +1});
        }
        auto lca = samurai::LevelCellArray<dim, TInterval>(lcl) ;
        for (auto _ : state){
                auto end = lca.end() ;
                benchmark::DoNotOptimize(end);
        }
}


template <unsigned int dim>
void LEVELCELLARRAY_shape(benchmark::State& state){
        samurai::LevelCellList<dim> lcl;
        using TInterval = samurai::default_config::interval_t;
        for (int64_t i = 0 ; i < state.range(0) ; i++){
                int index = static_cast<int>(i) ;
                lcl[{0}].add_interval({2*index,2*index +1});
        }
        auto lca = samurai::LevelCellArray<dim, TInterval>(lcl) ;
        for (auto _ : state){
                auto shape = lca.shape() ;
                benchmark::DoNotOptimize(shape);
        }
}


template <unsigned int dim>
void LEVELCELLARRAY_nb_intervals(benchmark::State& state){
        samurai::LevelCellList<dim> lcl;
        using TInterval = samurai::default_config::interval_t;
        for (int64_t i = 0 ; i < state.range(0) ; i++){
                int index = static_cast<int>(i) ;
                lcl[{0}].add_interval({2*index,2*index +1});
        }
        auto lca = samurai::LevelCellArray<dim, TInterval>(lcl) ;
        for (auto _ : state){
                auto nb = lca.nb_intervals() ;
                benchmark::DoNotOptimize(nb);
        }
}



template <unsigned int dim>
void LEVELCELLARRAY_nb_cells(benchmark::State& state){
        samurai::LevelCellList<dim> lcl;
        using TInterval = samurai::default_config::interval_t;
        for (int64_t i = 0 ; i < state.range(0) ; i++){
                int index = static_cast<int>(i) ;
                lcl[{0}].add_interval({2*index,2*index +1});
        }
        auto lca = samurai::LevelCellArray<dim, TInterval>(lcl) ;
        for (auto _ : state){
                auto nb = lca.nb_cells() ;
                benchmark::DoNotOptimize(nb);
        }
}

template <unsigned int dim>
void LEVELCELLARRAY_cell_length(benchmark::State& state){
        samurai::LevelCellList<dim> lcl;
        using TInterval = samurai::default_config::interval_t;
        for (int64_t i = 0 ; i < state.range(0) ; i++){
                int index = static_cast<int>(i) ;
                lcl[{0}].add_interval({2*index,2*index +1});
        }
        auto lca = samurai::LevelCellArray<dim, TInterval>(lcl) ;
        for (auto _ : state){
                auto length = lca.cell_length() ;
                benchmark::DoNotOptimize(length);
        }
}


template <unsigned int dim>
void LEVELCELLARRAY_max_indices(benchmark::State& state){
        samurai::LevelCellList<dim> lcl;
        using TInterval = samurai::default_config::interval_t;
        for (int64_t i = 0 ; i < state.range(0) ; i++){
                int index = static_cast<int>(i) ;
                lcl[{0}].add_interval({2*index,2*index +1});
        }
        auto lca = samurai::LevelCellArray<dim, TInterval>(lcl) ;
        for (auto _ : state){
                auto max = lca.max_indices() ;
                benchmark::DoNotOptimize(max);
        }
}

template <unsigned int dim>
void LEVELCELLARRAY_min_indices(benchmark::State& state){
        samurai::LevelCellList<dim> lcl;
        using TInterval = samurai::default_config::interval_t;
        for (int64_t i = 0 ; i < state.range(0) ; i++){
                int index = static_cast<int>(i) ;
                lcl[{0}].add_interval({2*index,2*index +1});
        }
        auto lca = samurai::LevelCellArray<dim, TInterval>(lcl) ;
        for (auto _ : state){
                auto min = lca.min_indices() ;
                benchmark::DoNotOptimize(min);
        }
}

template <unsigned int dim>
void LEVELCELLARRAY_minmax_indices(benchmark::State& state){
        samurai::LevelCellList<dim> lcl;
        using TInterval = samurai::default_config::interval_t;
        for (int64_t i = 0 ; i < state.range(0) ; i++){
                int index = static_cast<int>(i) ;
                lcl[{0}].add_interval({2*index,2*index +1});
        }
        auto lca = samurai::LevelCellArray<dim, TInterval>(lcl) ;
        for (auto _ : state){
                auto minmax = lca.minmax_indices() ;
                benchmark::DoNotOptimize(minmax);
        }
}



template <unsigned int dim>
void LEVELCELLARRAY_equal(benchmark::State& state){
        samurai::LevelCellList<dim> lcl;
        using TInterval = samurai::default_config::interval_t;
        for (int64_t i = 0 ; i < state.range(0) ; i++){
                int index = static_cast<int>(i) ;
                lcl[{0}].add_interval({2*index,2*index +1});
        }
        auto lca = samurai::LevelCellArray<dim, TInterval>(lcl) ;
	auto lca2 = lca ; 
        for (auto _ : state){
                auto is_equal = lca == lca2 ;
                benchmark::DoNotOptimize(is_equal);
        }
}




// manque les LevelCellList_iterator 



BENCHMARK_TEMPLATE(LEVELCELLARRAY_default, 1);
BENCHMARK_TEMPLATE(LEVELCELLARRAY_default, 2);
BENCHMARK_TEMPLATE(LEVELCELLARRAY_default, 3);

BENCHMARK_TEMPLATE(LEVELCELLARRAY_empty_lcl_to_lca, 1);
BENCHMARK_TEMPLATE(LEVELCELLARRAY_empty_lcl_to_lca, 2);
BENCHMARK_TEMPLATE(LEVELCELLARRAY_empty_lcl_to_lca, 3);



BENCHMARK_TEMPLATE(LEVELCELLARRAY_lcl_to_lca,1)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(LEVELCELLARRAY_lcl_to_lca,2)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(LEVELCELLARRAY_lcl_to_lca,3)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);


BENCHMARK_TEMPLATE(LEVELCELLARRAY_begin,1)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(LEVELCELLARRAY_begin,2)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(LEVELCELLARRAY_begin,3)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);

BENCHMARK_TEMPLATE(LEVELCELLARRAY_end,1)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(LEVELCELLARRAY_end,2)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(LEVELCELLARRAY_end,3)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);


BENCHMARK_TEMPLATE(LEVELCELLARRAY_shape,1)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(LEVELCELLARRAY_shape,2)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(LEVELCELLARRAY_shape,3)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);

BENCHMARK_TEMPLATE(LEVELCELLARRAY_nb_intervals,1)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(LEVELCELLARRAY_nb_intervals,2)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(LEVELCELLARRAY_nb_intervals,3)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);

BENCHMARK_TEMPLATE(LEVELCELLARRAY_nb_cells,1)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(LEVELCELLARRAY_nb_cells,2)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(LEVELCELLARRAY_nb_cells,3)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);


BENCHMARK_TEMPLATE(LEVELCELLARRAY_cell_length,1)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(LEVELCELLARRAY_cell_length,2)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(LEVELCELLARRAY_cell_length,3)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);

BENCHMARK_TEMPLATE(LEVELCELLARRAY_max_indices,1)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(LEVELCELLARRAY_max_indices,2)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(LEVELCELLARRAY_max_indices,3)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);


BENCHMARK_TEMPLATE(LEVELCELLARRAY_min_indices,1)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(LEVELCELLARRAY_min_indices,2)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(LEVELCELLARRAY_min_indices,3)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);

BENCHMARK_TEMPLATE(LEVELCELLARRAY_minmax_indices,1)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(LEVELCELLARRAY_minmax_indices,2)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(LEVELCELLARRAY_minmax_indices,3)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);

BENCHMARK_TEMPLATE(LEVELCELLARRAY_equal,1)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(LEVELCELLARRAY_equal,2)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);
BENCHMARK_TEMPLATE(LEVELCELLARRAY_equal,3)->RangeMultiplier(2)->Range(1 << 1, 1 << 10);



