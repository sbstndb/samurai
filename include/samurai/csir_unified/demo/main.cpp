#include "src/csir.hpp"
#include <algorithm>
#include <iostream>
#include <random>

static csir::CSIR_Level make_random_2d(std::size_t level,
                                       int x_min,
                                       int x_max,
                                       int y_min,
                                       int y_max,
                                       double row_density,
                                       int max_intervals_per_row,
                                       int min_len   = 2,
                                       int max_len   = 10,
                                       unsigned seed = 1234)
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
        tmp.reserve(max_intervals_per_row);
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
}

// Helper function to create a simple square/rectangular mesh
csir::CSIR_Level create_square_mesh(std::size_t level, int min_coord_x, int max_coord_x, int min_coord_y, int max_coord_y)
{
    csir::CSIR_Level mesh;
    mesh.level = level;
    mesh.intervals_ptr.push_back(0);

    for (int y = min_coord_y; y < max_coord_y; ++y)
    {
        mesh.y_coords.push_back(y);
        mesh.intervals.push_back({min_coord_x, max_coord_x});
        mesh.intervals_ptr.push_back(mesh.intervals.size());
    }
    return mesh;
}

int main()
{
    std::cout << "--- CSIR Unified Demo: Complex Multi-Level Operations ---" << std::endl;

    // =========== Part 1: Create a frame (ring) at Level 4 ===========
    std::cout << "\n1. Creating a frame at Level 4..." << std::endl;
    auto outer_square_lvl4 = create_square_mesh(4, 0, 20, 0, 20);
    auto hole_lvl4         = create_square_mesh(4, 5, 15, 5, 15);
    auto frame_lvl4        = csir::difference(outer_square_lvl4, hole_lvl4);
    std::cout << "--- Frame at Level 4 (20x20 with a 10x10 hole) ---" << std::endl;
    csir::print_level(frame_lvl4);

    // =========== Part 2: Create a cross shape at Level 5 ===========
    std::cout << "\n2. Creating a cross shape at Level 5..." << std::endl;
    auto vertical_bar_lvl5   = create_square_mesh(5, 18, 22, 10, 30);
    auto horizontal_bar_lvl5 = create_square_mesh(5, 10, 30, 18, 22);
    auto cross_lvl5          = csir::union_(vertical_bar_lvl5, horizontal_bar_lvl5);
    std::cout << "--- Cross at Level 5 ---" << std::endl;
    csir::print_level(cross_lvl5);

    // =========== Part 3: Project the frame to Level 5 and back to 4 ===========
    std::cout << "\n3. Projecting the frame from Level 4 to Level 5..." << std::endl;
    auto frame_lvl5 = csir::project_to_level(frame_lvl4, 5);
    std::cout << "--- Frame at Level 5 (40x40 with a 20x20 hole) ---" << std::endl;
    csir::print_level(frame_lvl5);

    std::cout << "\n4. Projecting the previous frame back to Level 4..." << std::endl;
    auto frame_lvl4_back = csir::project_to_level(frame_lvl5, 4);
    csir::print_level(frame_lvl4_back);

    // =========== Part 4: Final Assembly - Union of Frame and Cross ===========
    std::cout << "\n5. Final Assembly: Union of the projected frame and the cross..." << std::endl;
    auto final_assembly = csir::union_(frame_lvl5, cross_lvl5);
    std::cout << "--- FINAL RESULT ---" << std::endl;
    csir::print_level(final_assembly);

    // =========== Part 5: random 2D sets demonstration ===========
    std::cout << "\n6. Random 2D sets (reproducible) ..." << std::endl;
    auto rA = make_random_2d(4, 0, 50, 0, 25, 0.5, 3, 2, 7, 2025);
    auto rB = make_random_2d(4, 0, 50, 0, 25, 0.4, 2, 2, 9, 7);
    std::cout << "Random Set A:" << std::endl;
    csir::print_level(rA);
    std::cout << "Random Set B:" << std::endl;
    csir::print_level(rB);
    auto rU = csir::union_(rA, rB);
    auto rI = csir::intersection(rA, rB);
    auto rD = csir::difference(rA, rB);
    std::cout << "Union(A,B):" << std::endl;
    csir::print_level(rU);
    std::cout << "Intersect(A,B):" << std::endl;
    csir::print_level(rI);
    std::cout << "Diff(A,B):" << std::endl;
    csir::print_level(rD);

    return 0;
}
