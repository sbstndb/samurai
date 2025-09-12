#include <benchmark/benchmark.h>
#include "src/csir.hpp"
#include <random>
#include <algorithm>

namespace {

csir::CSIR_Level create_square_mesh(int size, std::size_t level) {
    csir::CSIR_Level mesh;
    mesh.level = level;
    mesh.intervals_ptr.push_back(0);
    for (int y = 0; y < size; ++y) {
        mesh.y_coords.push_back(y);
        mesh.intervals.push_back({0, size});
        mesh.intervals_ptr.push_back(mesh.intervals.size());
    }
    return mesh;
}

csir::CSIR_Level create_fragmented_mesh(int size, float density, std::size_t level, unsigned seed) {
    csir::CSIR_Level mesh;
    mesh.level = level;
    mesh.intervals_ptr.push_back(0);

    std::mt19937 gen(seed);
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (int y = 0; y < size; ++y) {
        mesh.y_coords.push_back(y);
        int last_x = -1;
        for (int x = 0; x < size; ++x) {
            if (dis(gen) < density) {
                if (last_x == -1) last_x = x;
            } else {
                if (last_x != -1) {
                    mesh.intervals.push_back({last_x, x});
                    last_x = -1;
                }
            }
        }
        if (last_x != -1) mesh.intervals.push_back({last_x, size});
        mesh.intervals_ptr.push_back(mesh.intervals.size());
    }
    return mesh;
}

csir::CSIR_Level create_checkerboard_mesh(int size, std::size_t level) {
    csir::CSIR_Level mesh;
    mesh.level = level;
    mesh.intervals_ptr.push_back(0);
    for (int y = 0; y < size; ++y) {
        mesh.y_coords.push_back(y);
        for (int x = 0; x < size; x += 2) {
            mesh.intervals.push_back({x, x + 1});
        }
        mesh.intervals_ptr.push_back(mesh.intervals.size());
    }
    return mesh;
}

}

static void BM_CSIR2D_Intersection_Solid(benchmark::State& state) {
    int size = state.range(0);
    auto a = create_square_mesh(size, 5);
    auto b = create_square_mesh(size, 5);
    for (auto _ : state) {
        auto res = csir::intersection(a, b);
        benchmark::DoNotOptimize(res);
    }
}
BENCHMARK(BM_CSIR2D_Intersection_Solid)->RangeMultiplier(2)->Range(64, 512);

static void BM_CSIR2D_Intersection_Fragmented(benchmark::State& state) {
    int size = state.range(0);
    auto a = create_fragmented_mesh(size, 0.2, 5, 1337);
    auto b = create_fragmented_mesh(size, 0.2, 5, 4242);
    for (auto _ : state) {
        auto res = csir::intersection(a, b);
        benchmark::DoNotOptimize(res);
    }
}
BENCHMARK(BM_CSIR2D_Intersection_Fragmented)->RangeMultiplier(2)->Range(64, 512);

static void BM_CSIR2D_Intersection_Checkerboard(benchmark::State& state) {
    int size = state.range(0);
    auto a = create_checkerboard_mesh(size, 5);
    auto b = create_checkerboard_mesh(size, 5);
    for (auto _ : state) {
        auto res = csir::intersection(a, b);
        benchmark::DoNotOptimize(res);
    }
}
BENCHMARK(BM_CSIR2D_Intersection_Checkerboard)->RangeMultiplier(2)->Range(64, 512);

static void BM_CSIR2D_ProjectUp(benchmark::State& state) {
    int size = state.range(0);
    auto s = create_checkerboard_mesh(size, 4);
    for (auto _ : state) {
        auto p = csir::project_to_level(s, s.level + 1);
        benchmark::DoNotOptimize(p);
    }
}
BENCHMARK(BM_CSIR2D_ProjectUp)->RangeMultiplier(2)->Range(64, 512);

static void BM_CSIR2D_ProjectDown(benchmark::State& state) {
    int size = state.range(0);
    auto s = create_checkerboard_mesh(size, 5);
    for (auto _ : state) {
        auto p = csir::project_to_level(s, s.level - 1);
        benchmark::DoNotOptimize(p);
    }
}
BENCHMARK(BM_CSIR2D_ProjectDown)->RangeMultiplier(2)->Range(64, 512);
