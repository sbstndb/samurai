#include "gtest/gtest.h"
#include "src/csir.hpp"
#include "test_utils.hpp"
#include <array>

using namespace csir;
using namespace csir_test_utils;

TEST(CSIR2DTest, Translate)
{
    auto src = make_rectangle(4, 10, 15, 3, 6);
    auto moved = translate(src, 2, -1);
    auto expected = make_rectangle(4, 12, 17, 2, 5);
    ASSERT_TRUE(equal_2d(moved, expected));
}

TEST(CSIR2DTest, Contract)
{
    auto src = make_rectangle(2, 0, 4, 0, 4);
    auto eroded = contract(src, 1);
    auto expected = make_rectangle(2, 1, 3, 1, 3);
    ASSERT_TRUE(equal_2d(eroded, expected));
}

TEST(CSIR2DTest, Expand)
{
    auto src = make_rectangle(3, 2, 4, 5, 6);
    std::array<bool, 2> mask{true, false};
    auto dilated = expand(src, 1, mask);
    auto expected = make_rectangle(3, 1, 5, 5, 6);
    ASSERT_TRUE(equal_2d(dilated, expected));
}

TEST(CSIR2DTest, DownProjection)
{
    CSIR_Level fine; fine.level = 3; fine.intervals_ptr.push_back(0);
    for (int y = 0; y < 4; ++y) { fine.y_coords.push_back(y); fine.intervals.push_back({0,4}); fine.intervals_ptr.push_back(fine.intervals.size()); }
    auto coarse = project_to_level(fine, 2);
    CSIR_Level expected; expected.level = 2; expected.intervals_ptr.push_back(0);
    for (int y = 0; y < 2; ++y) { expected.y_coords.push_back(y); expected.intervals.push_back({0,2}); expected.intervals_ptr.push_back(expected.intervals.size()); }
    ASSERT_TRUE(equal_2d(coarse, expected));
}