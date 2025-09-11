#include <iostream>
#include "csir.hpp"
#include <random>
#include <algorithm>

static csir::CSIR_Level_1D make_random_1d(std::size_t level, int x_min, int x_max, int n_intervals,
                                           int min_len = 2, int max_len = 15, unsigned seed = 42)
{
    using namespace csir;
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist_start(x_min, std::max(x_min, x_max - min_len - 1));
    std::uniform_int_distribution<int> dist_len(min_len, max_len);

    std::vector<Interval> tmp;
    tmp.reserve(n_intervals);
    for (int k = 0; k < n_intervals; ++k)
    {
        int s = dist_start(rng);
        int e = std::min(x_max, s + dist_len(rng));
        if (s < e) tmp.push_back({s, e});
    }
    // tri + fusion
    std::sort(tmp.begin(), tmp.end(), [](const Interval& a, const Interval& b){ return a.start < b.start; });
    std::vector<Interval> merged;
    for (auto itv : tmp)
    {
        if (merged.empty() || itv.start > merged.back().end)
            merged.push_back(itv);
        else
            merged.back().end = std::max(merged.back().end, itv.end);
    }
    CSIR_Level_1D out; out.level = level; out.intervals = std::move(merged);
    return out;
}

int main()
{
    using namespace csir;
    CSIR_Level_1D s; s.level = 4;
    s.intervals = {{10,20}, {30,35}};

    std::cout << "--- CSIR Unified 1D Demo ---\n";
    std::cout << "Original:" << std::endl; print_level_1d(s);

    auto moved = translate(s, +3);
    std::cout << "\nTranslate +3:" << std::endl; print_level_1d(moved);

    auto eroded = contract(s, 2);
    std::cout << "\nContract width=2:" << std::endl; print_level_1d(eroded);

    auto dilated = expand(s, 2);
    std::cout << "\nExpand width=2:" << std::endl; print_level_1d(dilated);

    auto up = project_to_level(s, 5);
    std::cout << "\nProject up to L5:" << std::endl; print_level_1d(up);

    auto down = project_to_level(up, 4);
    std::cout << "\nProject back to L4:" << std::endl; print_level_1d(down);

    // =========== Part 2: random intervals demonstration ===========
    std::cout << "\n6. Random 1D sets (reproducible) ..." << std::endl;
    auto rA = make_random_1d(4, 0, 120, 20, 3, 12, 1337);
    auto rB = make_random_1d(4, 0, 120, 18, 2, 10, 4242);
    std::cout << "Set A:" << std::endl; print_level_1d(rA);
    std::cout << "Set B:" << std::endl; print_level_1d(rB);

    auto rU = csir::union_(rA, rB);
    auto rI = csir::intersection(rA, rB);
    auto rD = csir::difference(rA, rB);
    std::cout << "Union(A,B):" << std::endl; print_level_1d(rU);
    std::cout << "Intersect(A,B):" << std::endl; print_level_1d(rI);
    std::cout << "Diff(A,B):" << std::endl; print_level_1d(rD);

    return 0;
}
