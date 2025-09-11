#include <chrono>
#include <iostream>
#include "csir.hpp"

csir::CSIR_Level create_square_mesh(int size) {
    csir::CSIR_Level mesh;
    mesh.level = 4;
    mesh.intervals_ptr.push_back(0);
    for (int y = 0; y < size; ++y) {
        mesh.y_coords.push_back(y);
        mesh.intervals.push_back({0, size});
        mesh.intervals_ptr.push_back(mesh.intervals.size());
    }
    return mesh;
}

void run_benchmark(const std::string& name, const csir::CSIR_Level& A, const csir::CSIR_Level& B) {
    std::cout << "Benchmark: " << name << std::endl;
    auto t0 = std::chrono::high_resolution_clock::now();
    auto res = csir::intersection(A, B);
    auto t1 = std::chrono::high_resolution_clock::now();
    auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cout << "  intersection took " << dt << " ms\n";
}

int main() {
    int size = 2000;
    auto A = create_square_mesh(size);
    auto B = create_square_mesh(size);
    run_benchmark("dense square 2000x2000", A, B);
    return 0;
}

