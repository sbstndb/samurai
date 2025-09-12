#pragma once

#include "src/csir.hpp"
#include <algorithm>
#include <map>
#include <vector>

namespace csir_test_utils
{
    using namespace csir;

    static CSIR_Level_1D make_1d(std::size_t level, std::initializer_list<Interval> iv)
    {
        CSIR_Level_1D s;
        s.level     = level;
        s.intervals = iv;
        return s;
    }

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

    static CSIR_Level_3D make_block3d(std::size_t level, int x0, int x1, int y0, int y1, int z0, int z1)
    {
        CSIR_Level_3D c3;
        c3.level = level;
        for (int z = z0; z < z1; ++z)
        {
            c3.slices[z] = make_rectangle(level, x0, x1, y0, y1);
        }
        return c3;
    }

    // Alias helper used by some tests
    static CSIR_Level_3D make_cube(std::size_t level, int x0, int x1, int y0, int y1, int z0, int z1)
    {
        return make_block3d(level, x0, x1, y0, y1, z0, z1);
    }

    static bool equal_1d(const CSIR_Level_1D& a, const CSIR_Level_1D& b)
    {
        if (a.level != b.level)
        {
            return false;
        }
        if (a.intervals.size() != b.intervals.size())
        {
            return false;
        }
        for (std::size_t i = 0; i < a.intervals.size(); ++i)
        {
            if (a.intervals[i].start != b.intervals[i].start)
            {
                return false;
            }
            if (a.intervals[i].end != b.intervals[i].end)
            {
                return false;
            }
        }
        return true;
    }

    static bool equal_2d(const CSIR_Level& a, const CSIR_Level& b)
    {
        if (a.y_coords.size() != b.y_coords.size())
        {
            return false;
        }
        if (a.intervals_ptr.size() != b.intervals_ptr.size())
        {
            return false;
        }
        if (a.intervals.size() != b.intervals.size())
        {
            return false;
        }
        for (std::size_t i = 0; i < a.y_coords.size(); ++i)
        {
            if (a.y_coords[i] != b.y_coords[i])
            {
                return false;
            }
            auto sa = a.intervals_ptr[i], ea = a.intervals_ptr[i + 1];
            auto sb = b.intervals_ptr[i], eb = b.intervals_ptr[i + 1];
            if (ea - sa != eb - sb)
            {
                return false;
            }
            for (std::size_t j = 0; j < ea - sa; ++j)
            {
                if (a.intervals[sa + j].start != b.intervals[sb + j].start)
                {
                    return false;
                }
                if (a.intervals[sa + j].end != b.intervals[sb + j].end)
                {
                    return false;
                }
            }
        }
        return true;
    }

    static bool equal_3d(const CSIR_Level_3D& a, const CSIR_Level_3D& b)
    {
        if (a.slices.size() != b.slices.size())
        {
            return false;
        }
        for (const auto& [z, sliceA] : a.slices)
        {
            auto it = b.slices.find(z);
            if (it == b.slices.end())
            {
                return false;
            }
            if (!equal_2d(sliceA, it->second))
            {
                return false;
            }
        }
        return true;
    }

} // namespace csir_test_utils
