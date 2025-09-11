#include <chrono>
#include <iostream>
#include "csir.hpp"

int main()
{
    using namespace csir;
    CSIR_Level_1D a; a.level=6; CSIR_Level_1D b; b.level=6;
    // Build ~1e5 intervals
    int n = 100000;
    a.intervals.reserve(n); b.intervals.reserve(n);
    for (int i = 0; i < n; ++i)
    {
        a.intervals.push_back({i*5, i*5+3});
        b.intervals.push_back({i*5+2, i*5+5});
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    auto u  = union_(a,b);
    auto t1 = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count();
    std::cout << "1D union of " << n << " intervals took " << ms << " ms\n";

    t0 = std::chrono::high_resolution_clock::now();
    auto i = intersection(a,b);
    t1 = std::chrono::high_resolution_clock::now();
    ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count();
    std::cout << "1D intersection took " << ms << " ms\n";

    return 0;
}

