#include <cassert>
#include <iostream>
#include <algorithm>
#include <array>
#include <map>

#include "csir.hpp"

using namespace csir;

static CSIR_Level make_rectangle2d(std::size_t level, int x0, int x1, int y0, int y1)
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

static CSIR_Level_3D make_block3d(std::size_t level, int x0, int x1, int y0, int y1, int z0, int z1)
{
    CSIR_Level_3D c3;
    c3.level = level;
    for (int z = z0; z < z1; ++z)
    {
        c3.slices[z] = make_rectangle2d(level, x0, x1, y0, y1);
    }
    return c3;
}

static bool equal_2d(const CSIR_Level& a, const CSIR_Level& b)
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

static bool equal_3d(const CSIR_Level_3D& a, const CSIR_Level_3D& b)
{
    if (a.slices.size() != b.slices.size()) return false;
    for (const auto& [z, sliceA] : a.slices)
    {
        auto it = b.slices.find(z);
        if (it == b.slices.end()) return false;
        if (!equal_2d(sliceA, it->second)) return false;
    }
    return true;
}

int main()
{
    // 3D translate dx=+1, dy=+2, dz=-1
    {
        auto src      = make_block3d(2, 1, 3, 2, 4, 5, 7); // [x:1,3) x [y:2,4) x [z:5,7)
        auto moved    = translate(src, 1, 2, -1);
        auto expected = make_block3d(2, 2, 4, 4, 6, 4, 6); // z becomes [4,6)
        assert(equal_3d(moved, expected));
    }

    // 3D contract width=1 all axes for a 3x3x3 block → 1x1x1 block
    {
        auto src      = make_block3d(1, 0, 3, 0, 3, 0, 3);
        auto eroded   = contract(src, 1);
        auto expected = make_block3d(1, 1, 2, 1, 2, 1, 2);
        assert(equal_3d(eroded, expected));
    }

    // 3D expand width=1 along Z only of a single slab → adds neighbors at z±1
    {
        auto src      = make_block3d(3, 10, 12, 20, 21, 7, 8); // single z at 7
        std::array<bool,3> mask{false, false, true};
        auto dilated   = expand(src, 1, mask);
        auto expected  = make_block3d(3, 10, 12, 20, 21, 6, 9); // z = {6,7,8}
        assert(equal_3d(dilated, expected));
    }

    // 3D down-projection: level 3 -> 2 of a [0,8)^3 cube becomes [0,4)^3
    {
        auto fine = make_block3d(3, 0, 8, 0, 8, 0, 8);
        auto coarse = project_to_level_3d(fine, 2);
        auto expected = make_block3d(2, 0, 4, 0, 4, 0, 4);
        assert(equal_3d(coarse, expected));
    }

    std::cout << "All csir_unified tests (3D) passed." << std::endl;
    return 0;
}

