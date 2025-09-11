#include <cassert>
#include <iostream>
#include "csir.hpp"

using namespace csir;

static bool equal_1d(const CSIR_Level_1D& a, const CSIR_Level_1D& b)
{
    if (a.level != b.level) return false;
    if (a.intervals.size() != b.intervals.size()) return false;
    for (std::size_t i = 0; i < a.intervals.size(); ++i)
    {
        if (a.intervals[i].start != b.intervals[i].start) return false;
        if (a.intervals[i].end   != b.intervals[i].end)   return false;
    }
    return true;
}

int main()
{
    CSIR_Level_1D s; s.level = 4; s.intervals = {{10, 20}, {30, 35}};

    // translate
    {
        auto moved = translate(s, +5);
        CSIR_Level_1D expected; expected.level=4; expected.intervals={{15,25},{35,40}};
        assert(equal_1d(moved, expected));
    }

    // contract
    {
        auto eroded = contract(s, 2);
        CSIR_Level_1D expected; expected.level=4; expected.intervals={{12,18},{32,33}};
        assert(equal_1d(eroded, expected));
    }

    // expand
    {
        auto dilated = expand(s, 2);
        CSIR_Level_1D expected; expected.level=4; expected.intervals={{8,22},{28,37}};
        assert(equal_1d(dilated, expected));
    }

    // union
    {
        CSIR_Level_1D a; a.level=4; a.intervals={{0,5},{10,15}};
        CSIR_Level_1D b; b.level=4; b.intervals={{3,8},{12,20}};
        auto u = union_(a,b);
        CSIR_Level_1D expected; expected.level=4; expected.intervals={{0,8},{10,20}};
        assert(equal_1d(u, expected));
    }

    // intersection
    {
        CSIR_Level_1D a; a.level=4; a.intervals={{0,5},{10,15}};
        CSIR_Level_1D b; b.level=4; b.intervals={{3,8},{12,20}};
        auto i = intersection(a,b);
        CSIR_Level_1D expected; expected.level=4; expected.intervals={{3,5},{12,15}};
        assert(equal_1d(i, expected));
    }

    // difference
    {
        CSIR_Level_1D a; a.level=4; a.intervals={{0,10}};
        CSIR_Level_1D b; b.level=4; b.intervals={{3,5},{7,12}};
        auto d = difference(a,b);
        CSIR_Level_1D expected; expected.level=4; expected.intervals={{0,3},{5,7}};
        assert(equal_1d(d, expected));
    }

    // projection up/down
    {
        CSIR_Level_1D a; a.level=3; a.intervals={{0,4},{10,12}}; // L3
        auto up = project_to_level(a, 5); // ×4
        CSIR_Level_1D expected_up; expected_up.level=5; expected_up.intervals={{0,16},{40,48}};
        assert(equal_1d(up, expected_up));

        auto down = project_to_level(up, 3);
        // down coarse should merge back to original ranges
        assert(equal_1d(down, a));
    }

    std::cout << "All csir_unified tests (1D) passed." << std::endl;
    return 0;
}

