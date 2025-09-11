#include <gtest/gtest.h>
#include <xtensor/xarray.hpp>

#include <samurai/interval.hpp>
#include <samurai/list_of_intervals.hpp>

#include "test_common.hpp"

namespace samurai
{
    TEST(common_utils, make_interval)
    {
        auto interval = make_interval<int, int>(0, 5, 10);
        EXPECT_EQ(interval.start, 0);
        EXPECT_EQ(interval.end, 5);
        EXPECT_EQ(interval.index, 10);
        EXPECT_EQ(interval.step, 1);
    }

    TEST(common_utils, make_list)
    {
        auto list = make_list<int, int>({{0, 5}, {10, 15}});
        xt::xarray<Interval<int, int>> expected{
            {0, 5},
            {10, 15}
        };
        EXPECT_EQ(list, expected);
    }

    TEST(common_utils, make_xarray)
    {
        auto array = make_xarray<int, int>({{0, 5}, {10, 15}});
        EXPECT_EQ(array.size(), 2);
        EXPECT_EQ(array[0].start, 0);
        EXPECT_EQ(array[0].end, 5);
        EXPECT_EQ(array[1].start, 10);
        EXPECT_EQ(array[1].end, 15);
    }

    TEST(common_utils, list_of_intervals_comparison)
    {
        ListOfIntervals<int, int> list1;
        list1.add_interval({0, 5});
        list1.add_interval({10, 15});

        ListOfIntervals<int, int> list2;
        list2.add_interval({0, 5});
        list2.add_interval({10, 15});

        ListOfIntervals<int, int> list3;
        list3.add_interval({0, 5});
        list3.add_interval({12, 17});

        EXPECT_TRUE(list1 == list2);
        EXPECT_FALSE(list1 == list3);
    }
}