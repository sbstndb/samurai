#include <benchmark/benchmark.h>
#include "src/csir.hpp"
#include <random>
#include <algorithm>
#include <map>

namespace {

// Helper function to create a simple cube mesh
csir::CSIR_Level_3D create_cube_mesh(std::size_t level, int min_coord, int max_coord) {
    csir::CSIR_Level_3D mesh_3d;
    mesh_3d.level = level;

    csir::CSIR_Level slice_2d;
    slice_2d.level = level;
    slice_2d.intervals_ptr.push_back(0);
    for (int y = min_coord; y < max_coord; ++y) {
        slice_2d.y_coords.push_back(y);
        slice_2d.intervals.push_back({min_coord, max_coord});
        slice_2d.intervals_ptr.push_back(slice_2d.intervals.size());
    }

    for (int z = min_coord; z < max_coord; ++z) {
        mesh_3d.slices[z] = slice_2d;
    }
    return mesh_3d;
}

// Helper function to create a fragmented 3D mesh
csir::CSIR_Level_3D create_fragmented_3d_mesh(std::size_t level, int size, float density, unsigned seed) {
    csir::CSIR_Level_3D mesh_3d;
    mesh_3d.level = level;
    std::mt19937 gen(seed);
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (int z = 0; z < size; ++z) {
        csir::CSIR_Level slice_2d;
        slice_2d.level = level;
        slice_2d.intervals_ptr.push_back(0);

        for (int y = 0; y < size; ++y) {
            if (dis(gen) < density) { // Only create a row if density allows
                slice_2d.y_coords.push_back(y);
                int last_x = -1;
                for (int x = 0; x < size; ++x) {
                    if (dis(gen) < density) {
                        if (last_x == -1) last_x = x;
                    } else {
                        if (last_x != -1) {
                            slice_2d.intervals.push_back({last_x, x});
                            last_x = -1;
                        }
                    }
                }
                if (last_x != -1) slice_2d.intervals.push_back({last_x, size});
                slice_2d.intervals_ptr.push_back(slice_2d.intervals.size());
            }
        }
        if (!slice_2d.empty()) {
            mesh_3d.slices[z] = slice_2d;
        }
    }
    return mesh_3d;
}

}

static void BM_CSIR3D_Intersection_Solid(benchmark::State& state) {
    int size = state.range(0);
    auto a = create_cube_mesh(2, 0, size);
    auto b = create_cube_mesh(2, size/2, size + size/2); // Offset to create partial overlap
    for (auto _ : state) {
        auto res = csir::intersection_3d(a, b);
        benchmark::DoNotOptimize(res);
    }
}
BENCHMARK(BM_CSIR3D_Intersection_Solid)->RangeMultiplier(2)->Range(16, 64);

static void BM_CSIR3D_Intersection_Fragmented(benchmark::State& state) {
    int size = state.range(0);
    auto a = create_fragmented_3d_mesh(2, size, 0.2, 100);
    auto b = create_fragmented_3d_mesh(2, size, 0.2, 200);
    for (auto _ : state) {
        auto res = csir::intersection_3d(a, b);
        benchmark::DoNotOptimize(res);
    }
}
BENCHMARK(BM_CSIR3D_Intersection_Fragmented)->RangeMultiplier(2)->Range(16, 64);

static void BM_CSIR3D_ProjectUp(benchmark::State& state) {
    int size = state.range(0);
    auto s = create_cube_mesh(2, 0, size);
    for (auto _ : state) {
        auto p = csir::project_to_level_3d(s, s.level + 1);
        benchmark::DoNotOptimize(p);
    }
}
BENCHMARK(BM_CSIR3D_ProjectUp)->RangeMultiplier(2)->Range(16, 64);

static void BM_CSIR3D_ProjectDown(benchmark::State& state) {
    int size = state.range(0);
    auto s = create_cube_mesh(3, 0, size * 2);
    for (auto _ : state) {
        auto p = csir::project_to_level_3d(s, s.level - 1);
        benchmark::DoNotOptimize(p);
    }
}
BENCHMARK(BM_CSIR3D_ProjectDown)->RangeMultiplier(2)->Range(16, 64);
