#include <iostream>
#include "../src/csir_cuda.cuh"

// Helper function to print a host-side CSIR level
void print_level_host(const csir::cuda::CSIR_Level_Host& csir) {
    if (csir.empty()) {
        std::cout << "    <empty slice>" << std::endl;
        return;
    }
    for (size_t i = 0; i < csir.y_coords.size(); ++i)
    {
        std::cout << "    y = " << csir.y_coords[i] << ": ";
        auto start_idx = csir.intervals_ptr[i];
        auto end_idx = csir.intervals_ptr[i+1];
        for (size_t j = start_idx; j < end_idx; ++j)
        {
            std::cout << "[" << csir.intervals[j].start << ", " << csir.intervals[j].end << ") ";
        }
        std::cout << std::endl;
    }
}

// Helper function to create a simple square/rectangular mesh
csir::cuda::CSIR_Level_Host create_square_mesh(std::size_t level, int min_coord_x, int max_coord_x, int min_coord_y, int max_coord_y) {
    csir::cuda::CSIR_Level_Host mesh;
    mesh.level = level;
    mesh.intervals_ptr.push_back(0);

    for (int y = min_coord_y; y < max_coord_y; ++y) {
        mesh.y_coords.push_back(y);
        mesh.intervals.push_back({min_coord_x, max_coord_x});
        mesh.intervals_ptr.push_back(mesh.intervals.size());
    }
    return mesh;
}

int main() {
    std::cout << "--- CSIR CUDA Demo: 2D Union ---" << std::endl;

    // 1. Create two overlapping sets at the same level
    std::cout << "\n1. Creating test sets..." << std::endl;
    auto set_A = create_square_mesh(4, 0, 10, 0, 10);
    auto set_B = create_square_mesh(4, 5, 15, 5, 15);

    std::cout << "--- Set A ---" << std::endl;
    print_level_host(set_A);

    std::cout << "\n--- Set B ---" << std::endl;
    print_level_host(set_B);

    // 2. Compute the union using the CUDA implementation
    std::cout << "\n2. Computing union on GPU..." << std::endl;
    auto cuda_union_result = csir::cuda::union_(set_A, set_B);

    // 3. Print the result
    std::cout << "\n--- CUDA Union Result ---" << std::endl;
    print_level_host(cuda_union_result);

    std::cout << "\nDemo finished." << std::endl;

    return 0;
}