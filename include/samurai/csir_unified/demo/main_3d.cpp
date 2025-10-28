#include "src/csir.hpp"
#include <algorithm>
#include <iostream>
#include <random>

// Helper function to create a simple cube mesh
csir::CSIR_Level_3D create_cube_mesh(std::size_t level, int min_coord, int max_coord)
{
    csir::CSIR_Level_3D mesh_3d;
    mesh_3d.level = level;

    csir::CSIR_Level slice_2d;
    slice_2d.level = level;
    slice_2d.intervals_ptr.push_back(0);
    for (int y = min_coord; y < max_coord; ++y)
    {
        slice_2d.y_coords.push_back(y);
        slice_2d.intervals.push_back({min_coord, max_coord});
        slice_2d.intervals_ptr.push_back(slice_2d.intervals.size());
    }

    for (int z = min_coord; z < max_coord; ++z)
    {
        mesh_3d.slices[z] = slice_2d;
    }
    return mesh_3d;
}

int main()
{
    std::cout << "--- CSIR Unified 3D Demo ---" << std::endl;

    // 1. Create a 10x10x10 cube mesh at level 2
    std::cout << "\n1. Creating Mesh A (10x10x10 cube) at Level 2..." << std::endl;
    auto mesh_A_lvl2 = create_cube_mesh(2, 0, 10);
    csir::print_level_3d(mesh_A_lvl2);

    // 2. Create a 5x5x5 cube mesh at level 3, offset to the corner
    std::cout << "\n2. Creating Mesh B (5x5x5 cube, offset) at Level 3..." << std::endl;
    auto mesh_B_lvl3 = create_cube_mesh(3, 15, 20);
    csir::print_level_3d(mesh_B_lvl3);

    // 3. Project A from level 2 to 3
    std::cout << "\n3. Projecting Mesh A from Level 2 to Level 3..." << std::endl;
    auto mesh_A_lvl3 = csir::project_to_level_3d(mesh_A_lvl2, 3);
    csir::print_level_3d(mesh_A_lvl3);

    // 4. Intersection at level 3
    std::cout << "\n4. Intersecting (Projected Mesh A) and (Mesh B)..." << std::endl;
    auto intersection_result = csir::intersection_3d(mesh_A_lvl3, mesh_B_lvl3);
    csir::print_level_3d(intersection_result);

    // 5. Down-project Mesh B from 3 to 2
    std::cout << "\n5. Projecting Mesh B from Level 3 down to Level 2..." << std::endl;
    auto mesh_B_lvl2 = csir::project_to_level_3d(mesh_B_lvl3, 2);
    csir::print_level_3d(mesh_B_lvl2);

    // 6. Random 3D sets (small) demonstration
    auto make_random_2d = [](std::size_t level,
                             int x_min,
                             int x_max,
                             int y_min,
                             int y_max,
                             double row_density,
                             int max_intervals_per_row,
                             int min_len,
                             int max_len,
                             unsigned seed) -> csir::CSIR_Level
    {
        using namespace csir;
        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> prob(0.0, 1.0);
        std::uniform_int_distribution<int> dist_start(x_min, std::max(x_min, x_max - min_len - 1));
        std::uniform_int_distribution<int> dist_len(min_len, max_len);
        std::uniform_int_distribution<int> dist_count(1, std::max(1, max_intervals_per_row));

        CSIR_Level out;
        out.level = level;
        out.intervals_ptr.push_back(0);
        for (int y = y_min; y < y_max; ++y)
        {
            if (prob(rng) > row_density)
            {
                continue;
            }
            std::vector<Interval> tmp;
            int count = dist_count(rng);
            for (int k = 0; k < count; ++k)
            {
                int s = dist_start(rng);
                int e = std::min(x_max, s + dist_len(rng));
                if (s < e)
                {
                    tmp.push_back({s, e});
                }
            }
            std::sort(tmp.begin(),
                      tmp.end(),
                      [](const Interval& a, const Interval& b)
                      {
                          return a.start < b.start;
                      });
            std::vector<Interval> merged;
            for (auto itv : tmp)
            {
                if (merged.empty() || itv.start > merged.back().end)
                {
                    merged.push_back(itv);
                }
                else
                {
                    merged.back().end = std::max(merged.back().end, itv.end);
                }
            }
            if (!merged.empty())
            {
                out.y_coords.push_back(y);
                out.intervals.insert(out.intervals.end(), merged.begin(), merged.end());
                out.intervals_ptr.push_back(out.intervals.size());
            }
        }
        return out;
    };

    auto make_random_3d = [&](std::size_t level,
                              int x_min,
                              int x_max,
                              int y_min,
                              int y_max,
                              int z_min,
                              int z_max,
                              double density,
                              int max_intervals_per_row,
                              int min_len,
                              int max_len,
                              unsigned seed) -> csir::CSIR_Level_3D
    {
        csir::CSIR_Level_3D out;
        out.level = level;
        std::mt19937 base(seed);
        int idx = 0;
        for (int z = z_min; z < z_max; ++z)
        {
            unsigned sd   = base() ^ (idx * 2654435761u);
            out.slices[z] = make_random_2d(level, x_min, x_max, y_min, y_max, density, max_intervals_per_row, min_len, max_len, sd);
            ++idx;
        }
        return out;
    };

    std::cout << "\n6. Random 3D sets (reproducible) ..." << std::endl;
    auto rA = make_random_3d(2, 0, 30, 0, 20, 0, 5, 0.5, 2, 2, 8, 99);
    auto rB = make_random_3d(2, 0, 30, 0, 20, 0, 5, 0.4, 3, 2, 10, 123);
    std::cout << "Random 3D Set A:" << std::endl;
    csir::print_level_3d(rA);
    std::cout << "Random 3D Set B:" << std::endl;
    csir::print_level_3d(rB);
    auto rU = csir::union_3d(rA, rB);
    auto rI = csir::intersection_3d(rA, rB);
    std::cout << "Union(A,B):" << std::endl;
    csir::print_level_3d(rU);
    std::cout << "Intersect(A,B):" << std::endl;
    csir::print_level_3d(rI);

    return 0;
}
