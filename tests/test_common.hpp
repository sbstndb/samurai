#pragma once

#include <xtensor/xarray.hpp>

#include <samurai/interval.hpp>
#include <samurai/list_of_intervals.hpp>

namespace samurai
{
    template <typename coord_t, typename index_t>
    bool operator==(const ListOfIntervals<coord_t, index_t>& li, const xt::xarray<Interval<coord_t, index_t>>& array)
    {
        auto ix = li.cbegin();
        auto iy = array.cbegin();
        while (ix != li.cend() && iy != array.cend())
        {
            if (*ix != *iy)
            {
                return false;
            }
            ++ix;
            ++iy;
        }
        if (ix == li.cend() && iy == array.cend())
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    // Utility function to create an interval
    template <typename coord_t, typename index_t = default_config::index_t>
    Interval<coord_t, index_t> make_interval(coord_t start, coord_t end, index_t index = 0)
    {
        return Interval<coord_t, index_t>(start, end, index);
    }

    // Utility function to create a list of intervals from initializer list
    template <typename coord_t, typename index_t = default_config::index_t>
    ListOfIntervals<coord_t, index_t> make_list(std::initializer_list<std::pair<coord_t, coord_t>> intervals)
    {
        ListOfIntervals<coord_t, index_t> list;
        for (const auto& interval : intervals)
        {
            list.add_interval(make_interval<coord_t, index_t>(interval.first, interval.second));
        }
        return list;
    }

    // Utility function to create an xarray of intervals from initializer list
    template <typename coord_t, typename index_t = default_config::index_t>
    xt::xarray<Interval<coord_t, index_t>> make_xarray(std::initializer_list<std::pair<coord_t, coord_t>> intervals)
    {
        xt::xarray<Interval<coord_t, index_t>> array;
        array.resize({intervals.size()});
        std::size_t i = 0;
        for (const auto& interval : intervals)
        {
            array[i++] = make_interval<coord_t, index_t>(interval.first, interval.second);
        }
        return array;
    }

    // Utility function to compare two ListOfIntervals
    template <typename coord_t, typename index_t>
    bool operator==(const ListOfIntervals<coord_t, index_t>& li1, const ListOfIntervals<coord_t, index_t>& li2)
    {
        auto ix1 = li1.cbegin();
        auto ix2 = li2.cbegin();
        while (ix1 != li1.cend() && ix2 != li2.cend())
        {
            if (*ix1 != *ix2)
            {
                return false;
            }
            ++ix1;
            ++ix2;
        }
        if (ix1 == li1.cend() && ix2 == li2.cend())
        {
            return true;
        }
        else
        {
            return false;
        }
    }
}
