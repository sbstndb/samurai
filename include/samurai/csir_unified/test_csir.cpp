#include <cassert>
#include <iostream>
#include <algorithm>
#include <array>

#include "csir.hpp"

using namespace csir;

static CSIR_Level make_rectangle(std::size_t level, int x0, int x1, int y0, int y1)
{
    CSIR_Level c;
    c.level = level;
    c.intervals_ptr.push_back(0);
    for (int y = y0; y < y1; ++y)
    {
        c.y_coords.push_back(y);
        c.intervals.push_back({x0, x1});
        c.intervals_ptr.push_back(c.intervals.size());
    }
    return c;
}

static bool equal_rows(const CSIR_Level& a, const CSIR_Level& b)
{
    if (a.y_coords.size() != b.y_coords.size()) return false;
    if (a.intervals_ptr.size() != b.intervals_ptr.size()) return false;
    if (a.intervals.size() != b.intervals.size()) return false;
    for (std::size_t i = 0; i < a.y_coords.size(); ++i)
    {
        if (a.y_coords[i] != b.y_coords[i]) return false;
        auto sa = a.intervals_ptr[i], ea = a.intervals_ptr[i+1];
        auto sb = b.intervals_ptr[i], eb = b.intervals_ptr[i+1];
        if (ea - sa != eb - sb) return false;
        for (std::size_t j = 0; j < ea - sa; ++j)
        {
            if (a.intervals[sa + j].start != b.intervals[sb + j].start) return false;
            if (a.intervals[sa + j].end   != b.intervals[sb + j].end)   return false;
        }
    }
    return true;
}

int main()
{
    // 2D translate
    {
        auto src      = make_rectangle(4, 10, 15, 3, 6);
        auto moved    = translate(src, 2, -1);
        auto expected = make_rectangle(4, 12, 17, 2, 5);
        assert(equal_rows(moved, expected));
    }

    // 2D contract width=1 both axes on a 4x4 block
    {
        auto src      = make_rectangle(2, 0, 4, 0, 4);
        auto eroded   = contract(src, 1);
        auto expected = make_rectangle(2, 1, 3, 1, 3);
        assert(equal_rows(eroded, expected));
    }

    // 2D expand width=1 along X only on a 1x2 strip
    {
        auto src      = make_rectangle(3, 2, 4, 5, 6);
        std::array<bool, 2> mask{true, false};
        auto dilated   = expand(src, 1, mask);
        auto expected = make_rectangle(3, 1, 5, 5, 6);
        assert(equal_rows(dilated, expected));
    }

    // 2D down-projection: level 3 -> level 2
    {
        CSIR_Level fine; fine.level = 3; fine.intervals_ptr.push_back(0);
        for (int y = 0; y < 4; ++y) { fine.y_coords.push_back(y); fine.intervals.push_back({0,4}); fine.intervals_ptr.push_back(fine.intervals.size()); }
        auto coarse = project_to_level(fine, 2);
        CSIR_Level expected; expected.level = 2; expected.intervals_ptr.push_back(0);
        for (int y = 0; y < 2; ++y) { expected.y_coords.push_back(y); expected.intervals.push_back({0,2}); expected.intervals_ptr.push_back(expected.intervals.size()); }
        assert(equal_rows(coarse, expected));
    }

    std::cout << "All csir_unified tests (2D) passed." << std::endl;
    return 0;
}

