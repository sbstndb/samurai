#include "gtest/gtest.h"
#include "src/csir.hpp"
#include "test_utils.hpp"

using namespace csir;
using namespace csir_test_utils;

TEST(CSIR1DTest, Translate)
{
    CSIR_Level_1D s = make_1d(4, {{10, 20}, {30, 35}});
    auto moved = translate(s, +5);
    CSIR_Level_1D expected = make_1d(4, {{15, 25}, {35, 40}});
    ASSERT_TRUE(equal_1d(moved, expected));
}

TEST(CSIR1DTest, Contract)
{
    CSIR_Level_1D s = make_1d(4, {{10, 20}, {30, 35}});
    auto eroded = contract(s, 2);
    CSIR_Level_1D expected = make_1d(4, {{12, 18}, {32, 33}});
    ASSERT_TRUE(equal_1d(eroded, expected));
}

TEST(CSIR1DTest, Expand)
{
    CSIR_Level_1D s = make_1d(4, {{10, 20}, {30, 35}});
    auto dilated = expand(s, 2);
    CSIR_Level_1D expected = make_1d(4, {{8, 22}, {28, 37}});
    ASSERT_TRUE(equal_1d(dilated, expected));
}

TEST(CSIR1DTest, Union)
{
    CSIR_Level_1D a = make_1d(4, {{0, 5}, {10, 15}});
    CSIR_Level_1D b = make_1d(4, {{3, 8}, {12, 20}});
    auto u = union_(a, b);
    CSIR_Level_1D expected = make_1d(4, {{0, 8}, {10, 20}});
    ASSERT_TRUE(equal_1d(u, expected));
}

TEST(CSIR1DTest, Intersection)
{
    CSIR_Level_1D a = make_1d(4, {{0, 5}, {10, 15}});
    CSIR_Level_1D b = make_1d(4, {{3, 8}, {12, 20}});
    auto i = intersection(a, b);
    CSIR_Level_1D expected = make_1d(4, {{3, 5}, {12, 15}});
    ASSERT_TRUE(equal_1d(i, expected));
}

TEST(CSIR1DTest, Difference)
{
    CSIR_Level_1D a = make_1d(4, {{0, 10}});
    CSIR_Level_1D b = make_1d(4, {{3, 5}, {7, 12}});
    auto d = difference(a, b);
    CSIR_Level_1D expected = make_1d(4, {{0, 3}, {5, 7}});
    ASSERT_TRUE(equal_1d(d, expected));
}

TEST(CSIR1DTest, ProjectionUpDown)
{
    CSIR_Level_1D a = make_1d(3, {{0, 4}, {10, 12}}); // L3
    auto up = project_to_level(a, 5); // ×4
    CSIR_Level_1D expected_up = make_1d(5, {{0, 16}, {40, 48}});
    ASSERT_TRUE(equal_1d(up, expected_up));

    auto down = project_to_level(up, 3);
    // down coarse should merge back to original ranges
    ASSERT_TRUE(equal_1d(down, a));
}