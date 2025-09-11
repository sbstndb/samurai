#include <iostream>
#include <chrono>
#include <vector>
#include <numeric>
#include "csir.hpp"
#include <random>

// --- Mesh Creation Helpers ---

csir::CSIR_Level create_square_mesh(int size) {
    csir::CSIR_Level mesh;
    mesh.level = 5;
    mesh.intervals_ptr.push_back(0);
    for (int y = 0; y < size; ++y) {
        mesh.y_coords.push_back(y);
        mesh.intervals.push_back({0, size});
        mesh.intervals_ptr.push_back(mesh.intervals.size());
    }
    return mesh;
}

csir::CSIR_Level create_fragmented_mesh(int size, float density) {
    csir::CSIR_Level mesh;
    mesh.level = 5;
    mesh.intervals_ptr.push_back(0);

    std::mt19937 gen(1337);
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

csir::CSIR_Level create_checkerboard_mesh(int size) {
    csir::CSIR_Level mesh;
    mesh.level = 5;
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

// --- Benchmark Runner ---
void run_benchmark(const std::string& name, const csir::CSIR_Level& mesh_a, const csir::CSIR_Level& mesh_b) {
    // Warm-up
    auto result = csir::intersection(mesh_a, mesh_b);

    // Timed runs
    int num_runs = 100;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_runs; ++i) {
        auto res = csir::intersection(mesh_a, mesh_b);
        asm volatile("" : : "r,m"(res) : "memory");
    }
    auto end = std::chrono::high_resolution_clock::now();

    auto total_duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    double avg_duration_ns = static_cast<double>(total_duration_ns) / num_runs;
    
    size_t total_intervals = mesh_a.intervals.size() + mesh_b.intervals.size();
    double ns_per_interval = avg_duration_ns / total_intervals;

    std::cout << "--- Benchmark: " << name << " ---" << std::endl;
    std::cout << "  Avg. Time: " << avg_duration_ns << " ns" << std::endl;
    std::cout << "  Total Input Intervals: " << total_intervals << std::endl;
    std::cout << "  Metric (ns per interval): " << ns_per_interval << " ns/interval" << std::endl;
    std::cout << std::endl;
}

void run_projection_benchmark(const std::string& name, const csir::CSIR_Level& source_mesh, std::size_t target_level) {
    // Warm-up
    auto result = csir::project_to_level(source_mesh, target_level);

    // Timed runs
    int num_runs = 100;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_runs; ++i) {
        auto res = csir::project_to_level(source_mesh, target_level);
        asm volatile("" : : "r,m"(res) : "memory");
    }
    auto end = std::chrono::high_resolution_clock::now();

    auto total_duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    double avg_duration_ns = static_cast<double>(total_duration_ns) / num_runs;
    
    size_t total_intervals = source_mesh.intervals.size();
    double ns_per_interval = avg_duration_ns / total_intervals;

    std::cout << "--- Benchmark: " << name << " ---" << std::endl;
    std::cout << "  Avg. Time: " << avg_duration_ns << " ns" << std::endl;
    std::cout << "  Total Input Intervals: " << total_intervals << std::endl;
    std::cout << "  Metric (ns per input interval): " << ns_per_interval << " ns/interval" << std::endl;
    std::cout << std::endl;
}

int main() {
    std::cout << "--- Manual 2D Intersection Benchmark (Optimized Build) ---" << std::endl;

    for (int size : {64, 128, 256, 512}) {
        std::cout << "\n--- Testing Size: " << size << "x" << size << " ---" << std::endl;
        
        auto solid_a = create_square_mesh(size);
        auto solid_b = create_square_mesh(size);
        run_benchmark("Solid Intersection", solid_a, solid_b);

        auto fragmented_a = create_fragmented_mesh(size, 0.2);
        auto fragmented_b = create_fragmented_mesh(size, 0.2);
        run_benchmark("Fragmented Intersection", fragmented_a, fragmented_b);

        auto checker_a = create_checkerboard_mesh(size);
        auto checker_b = create_checkerboard_mesh(size);
        run_benchmark("Checkerboard Intersection", checker_a, checker_b);

        // Scenario 4: Projection of a checkerboard mesh
        run_projection_benchmark("Checkerboard Projection L -> L+1", checker_a, checker_a.level + 1);
    }

    return 0;
}
