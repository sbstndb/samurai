// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier: BSD-3-Clause

#include <iostream>
#include <chrono>

#include <samurai/samurai.hpp>

/**
 * Benchmark to compare standard vs aggregated MPI communications
 * for ghost cell updates in multiresolution context
 */

template <class Field>
void benchmark_ghost_updates(Field& u, const std::string& label, int iterations = 100)
{
    using mesh_t = typename Field::mesh_t;
    auto& mesh = u.mesh();
    
#ifdef SAMURAI_WITH_MPI
    mpi::communicator world;
    
    // Warm-up
    samurai::update_ghost_mr(u);
    world.barrier();
    
    // Benchmark standard version
    auto start_standard = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        samurai::update_ghost_mr(u);
    }
    world.barrier();
    auto end_standard = std::chrono::high_resolution_clock::now();
    
    // Benchmark aggregated version
    auto start_aggregated = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        samurai::update_ghost_mr_aggregated(u);
    }
    world.barrier();
    auto end_aggregated = std::chrono::high_resolution_clock::now();
    
    // Calculate timings
    auto time_standard = std::chrono::duration<double>(end_standard - start_standard).count();
    auto time_aggregated = std::chrono::duration<double>(end_aggregated - start_aggregated).count();
    
    if (world.rank() == 0) {
        std::cout << "\n=== " << label << " ===" << std::endl;
        std::cout << "Mesh info:" << std::endl;
        std::cout << "  - Min level: " << mesh.min_level() << std::endl;
        std::cout << "  - Max level: " << mesh.max_level() << std::endl;
        std::cout << "  - Total cells: " << mesh.nb_cells() << std::endl;
        std::cout << "  - MPI ranks: " << world.size() << std::endl;
        std::cout << "  - Iterations: " << iterations << std::endl;
        
        std::cout << "\nTimings:" << std::endl;
        std::cout << "  - Standard ghost update: " << time_standard << " seconds" << std::endl;
        std::cout << "  - Aggregated ghost update: " << time_aggregated << " seconds" << std::endl;
        std::cout << "  - Speedup: " << time_standard / time_aggregated << "x" << std::endl;
        std::cout << "  - Time saved: " << (time_standard - time_aggregated) << " seconds ("
                  << ((time_standard - time_aggregated) / time_standard * 100) << "%)" << std::endl;
    }
#else
    std::cout << "This benchmark requires MPI support. Please compile with SAMURAI_WITH_MPI." << std::endl;
#endif
}

int main(int argc, char* argv[])
{
    samurai::initialize(argc, argv);
    
    constexpr std::size_t dim = 2;
    using Config = samurai::MRConfig<dim>;
    using Box = samurai::Box<double, dim>;
    using point_t = typename Box::point_t;
    
    std::cout << "=== MPI Communication Aggregation Benchmark ===" << std::endl;
    
    // Test case 1: Uniform mesh
    {
        point_t box_corner1{0., 0.};
        point_t box_corner2{1., 1.};
        Box box(box_corner1, box_corner2);
        
        samurai::MRMesh<Config> mesh{box, 3, 5}; // min_level=3, max_level=5
        
        auto u = samurai::make_field<double, 1>("u", mesh);
        u.array().fill(1.0);
        
        benchmark_ghost_updates(u, "Test 1: Uniform mesh (levels 3-5)", 50);
    }
    
    // Test case 2: Refined center
    {
        point_t box_corner1{0., 0.};
        point_t box_corner2{1., 1.};
        Box box(box_corner1, box_corner2);
        
        samurai::MRMesh<Config> mesh{box, 2, 7}; // min_level=2, max_level=7
        
        // Create refinement in center
        auto tag = samurai::make_field<int, 1>("tag", mesh);
        tag.array().fill(0);
        
        for (std::size_t level = mesh.min_level(); level < mesh.max_level(); ++level) {
            auto leaves = samurai::intersection(mesh[samurai::MeshType::cells][level],
                                               mesh[samurai::MeshType::cells][level]);
            
            leaves([&](const auto& interval, const auto& index) {
                auto x_center = 0.5 * (interval.start + interval.end) * mesh.cell_length(level);
                auto y_center = 0.5 * (index[0] + index[0] + 1) * mesh.cell_length(level);
                
                // Refine cells near center
                if (std::abs(x_center - 0.5) < 0.2 && std::abs(y_center - 0.5) < 0.2) {
                    tag(level, interval, index) = static_cast<int>(samurai::CellFlag::refine);
                } else {
                    tag(level, interval, index) = static_cast<int>(samurai::CellFlag::keep);
                }
            });
        }
        
        mesh = samurai::adapt(tag);
        auto u = samurai::make_field<double, 1>("u", mesh);
        u.array().fill(1.0);
        
        benchmark_ghost_updates(u, "Test 2: Refined center (levels 2-7)", 50);
    }
    
    // Test case 3: Many levels
    {
        point_t box_corner1{0., 0.};
        point_t box_corner2{1., 1.};
        Box box(box_corner1, box_corner2);
        
        samurai::MRMesh<Config> mesh{box, 1, 10}; // min_level=1, max_level=10
        
        auto u = samurai::make_field<double, 1>("u", mesh);
        u.array().fill(1.0);
        
        benchmark_ghost_updates(u, "Test 3: Many levels (1-10)", 20);
    }
    
#ifdef SAMURAI_WITH_MPI
    mpi::communicator world;
    if (world.rank() == 0) {
        std::cout << "\n=== Benchmark completed ===" << std::endl;
        std::cout << "Note: Aggregated communication reduces the number of MPI messages from O(levels × neighbors) to O(neighbors)" << std::endl;
    }
#endif
    
    samurai::finalize();
    return 0;
}
