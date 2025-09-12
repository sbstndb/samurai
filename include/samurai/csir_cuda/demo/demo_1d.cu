#include <iostream>
#include <vector>
#include <algorithm>
#include "../src/csir_cuda.cuh"

// Helper function to print a host-side 1D CSIR level
void print_level_1d_host(const csir::cuda::CSIR_Level_1D_Host& csir)
{
    if (csir.empty()) { std::cout << "    <empty>" << std::endl; return; }
    std::cout << "    ";
    for (const auto& itv : csir.intervals) { std::cout << "[" << itv.start << ", " << itv.end << ") "; }
    std::cout << std::endl;
}

// Helper to create a simple 1D set
csir::cuda::CSIR_Level_1D_Host create_1d_set(std::size_t level, const std::vector<csir::cuda::Interval>& intervals)
{
    csir::cuda::CSIR_Level_1D_Host set;
    set.level = level;
    set.intervals = intervals;
    return set;
}

int main()
{
    std::cout << "--- CSIR CUDA Demo: 1D Operations ---" << std::endl;

    // Test Union
    std::cout << "\n--- Union Test ---" << std::endl;
    auto set1_1d = create_1d_set(0, {{0, 5}, {10, 15}});
    auto set2_1d = create_1d_set(0, {{3, 8}, {12, 18}});
    std::cout << "Set 1: "; print_level_1d_host(set1_1d);
    std::cout << "Set 2: "; print_level_1d_host(set2_1d);
    auto union_result_1d = csir::cuda::union_1d(set1_1d, set2_1d);
    std::cout << "Union Result: "; print_level_1d_host(union_result_1d);

    // Test Intersection
    std::cout << "\n--- Intersection Test ---" << std::endl;
    auto intersect_result_1d = csir::cuda::intersection_1d(set1_1d, set2_1d);
    std::cout << "Intersection Result: "; print_level_1d_host(intersect_result_1d);

    // Test Difference
    std::cout << "\n--- Difference Test ---" << std::endl;
    auto diff_result_1d = csir::cuda::difference_1d(set1_1d, set2_1d);
    std::cout << "Difference Result (Set1 - Set2): "; print_level_1d_host(diff_result_1d);

    // Test Translate
    std::cout << "\n--- Translate Test ---" << std::endl;
    auto translate_result_1d = csir::cuda::translate_1d(set1_1d, 5);
    std::cout << "Translate Set1 by 5: "; print_level_1d_host(translate_result_1d);

    // Test Contract
    std::cout << "\n--- Contract Test ---" << std::endl;
    auto set3_1d = create_1d_set(0, {{0, 20}}); 
    std::cout << "Set 3: "; print_level_1d_host(set3_1d);
    auto contract_result_1d = csir::cuda::contract_1d(set3_1d, 2);
    std::cout << "Contract Set 3 by 2: "; print_level_1d_host(contract_result_1d);

    // Test Expand
    std::cout << "\n--- Expand Test ---" << std::endl;
    auto set4_1d = create_1d_set(0, {{5, 10}}); 
    std::cout << "Set 4: "; print_level_1d_host(set4_1d);
    auto expand_result_1d = csir::cuda::expand_1d(set4_1d, 2);
    std::cout << "Expand Set 4 by 2: "; print_level_1d_host(expand_result_1d);

    // Test Project to Level
    std::cout << "\n--- Project to Level Test ---" << std::endl;
    auto set5_1d = create_1d_set(1, {{0, 10}, {20, 30}}); 
    std::cout << "Set 5 (Level 1): "; print_level_1d_host(set5_1d);
    auto project_up_result_1d = csir::cuda::project_to_level_1d(set5_1d, 2);
    std::cout << "Project Set 5 to Level 2 (Upscale): "; print_level_1d_host(project_up_result_1d);
    auto project_down_result_1d = csir::cuda::project_to_level_1d(project_up_result_1d, 1);
    std::cout << "Project back to Level 1 (Downscale): "; print_level_1d_host(project_down_result_1d);

    std::cout << "\nDemo finished." << std::endl;

    return 0;
}