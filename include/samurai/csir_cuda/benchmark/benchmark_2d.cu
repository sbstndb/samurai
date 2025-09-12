#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <cstring>

// Original CPU implementation for comparison
#include "../../csir_unified/src/csir.hpp"
// Our new CUDA implementation
#include "../src/csir_cuda.cuh"

// Use a namespace alias to distinguish between CPU and CUDA versions
namespace csir_cpu = csir;

// Data generation function (adapted to produce both CPU and CUDA host types)
namespace {

csir_cpu::CSIR_Level create_fragmented_mesh_cpu(int size, float density, std::size_t level, unsigned seed) {
    csir_cpu::CSIR_Level mesh;
    mesh.level = level;
    mesh.intervals_ptr.push_back(0);

    std::mt19937 gen(seed);
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (int y = 0; y < size; ++y) {
        bool row_has_intervals = false;
        int last_x = -1;
        for (int x = 0; x < size; ++x) {
            if (dis(gen) < density) {
                if (last_x == -1) last_x = x;
            } else {
                if (last_x != -1) {
                    mesh.intervals.push_back({last_x, x});
                    last_x = -1;
                    row_has_intervals = true;
                }
            }
        }
        if (last_x != -1) {
            mesh.intervals.push_back({last_x, size});
            row_has_intervals = true;
        }
        if(row_has_intervals) {
            mesh.y_coords.push_back(y);
            mesh.intervals_ptr.push_back(mesh.intervals.size());
        }
    }
    return mesh;
}

// Helper to convert from CPU struct to CUDA host struct (they are identical in layout)
csir::cuda::CSIR_Level_Host convert_to_cuda_host(const csir_cpu::CSIR_Level& cpu_level) {
    csir::cuda::CSIR_Level_Host host_level;
    host_level.level = cpu_level.level;
    host_level.y_coords = cpu_level.y_coords;
    host_level.intervals_ptr = cpu_level.intervals_ptr;
    // The interval struct is also identical, so we can just copy
    host_level.intervals.resize(cpu_level.intervals.size());
    std::memcpy(host_level.intervals.data(), cpu_level.intervals.data(), cpu_level.intervals.size() * sizeof(csir_cpu::Interval));
    return host_level;
}

}

int main() {
    // --- Benchmark Parameters ---
    const int size = 4096; // Large mesh size
    const float density = 0.1f; // Sparsity of the mesh
    const unsigned seed1 = 1337;
    const unsigned seed2 = 4242;

    std::cout << "--- CSIR 2D CPU vs CUDA Performance Benchmark ---" << std::endl;
    std::cout << "Mesh size: " << size << "x" << size << ", Density: " << density << std::endl << std::endl;

    // --- Data Generation ---
    auto A_cpu = create_fragmented_mesh_cpu(size, density, 5, seed1);
    auto B_cpu = create_fragmented_mesh_cpu(size, density, 5, seed2);

    auto A_cuda_host = convert_to_cuda_host(A_cpu);
    auto B_cuda_host = convert_to_cuda_host(B_cpu);

    // --- CPU Timings ---
    double cpu_union_time, cpu_intersection_time;

    {
        auto start = std::chrono::high_resolution_clock::now();
        auto res_union_cpu = csir_cpu::union_(A_cpu, B_cpu);
        auto end = std::chrono::high_resolution_clock::now();
        cpu_union_time = std::chrono::duration<double, std::milli>(end - start).count();
    }
    {
        auto start = std::chrono::high_resolution_clock::now();
        auto res_intersect_cpu = csir_cpu::intersection(A_cpu, B_cpu);
        auto end = std::chrono::high_resolution_clock::now();
        cpu_intersection_time = std::chrono::duration<double, std::milli>(end - start).count();
    }

    // --- GPU Timings ---
    float gpu_union_time, gpu_intersection_time;
    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);

    {
        cudaEventRecord(start_event);
        auto res_union_gpu = csir::cuda::union_(A_cuda_host, B_cuda_host);
        cudaEventRecord(stop_event);
        cudaEventSynchronize(stop_event);
        cudaEventElapsedTime(&gpu_union_time, start_event, stop_event);
    }
    {
        cudaEventRecord(start_event);
        auto res_intersect_gpu = csir::cuda::intersection(A_cuda_host, B_cuda_host);
        cudaEventRecord(stop_event);
        cudaEventSynchronize(stop_event);
        cudaEventElapsedTime(&gpu_intersection_time, start_event, stop_event);
    }

    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);

    // --- Results ---
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "+--------------+------------------+------------------+" << std::endl;
    std::cout << "| Operation    | CPU Time (ms)    | GPU Time (ms)    |" << std::endl;
    std::cout << "+--------------+------------------+------------------+" << std::endl;
    std::cout << "| Union        | " << std::setw(16) << cpu_union_time << " | " << std::setw(16) << gpu_union_time << " |" << std::endl;
    std::cout << "| Intersection | " << std::setw(16) << cpu_intersection_time << " | " << std::setw(16) << gpu_intersection_time << " |" << std::endl;
    std::cout << "+--------------+------------------+------------------+" << std::endl;

    return 0;
}