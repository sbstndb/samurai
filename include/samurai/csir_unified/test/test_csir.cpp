#include "src/csir.hpp"
#include "test_utils.hpp"
#include "gtest/gtest.h"
#include <array>

using namespace csir;
using namespace csir_test_utils;

TEST(CSIR2DTest, Translate)
{
    auto src      = make_rectangle(4, 10, 15, 3, 6);
    auto moved    = translate(src, 2, -1);
    auto expected = make_rectangle(4, 12, 17, 2, 5);
    ASSERT_TRUE(equal_2d(moved, expected));
}

TEST(CSIR2DTest, Contract)
{
    auto src      = make_rectangle(2, 0, 4, 0, 4);
    auto eroded   = contract(src, 1);
    auto expected = make_rectangle(2, 1, 3, 1, 3);
    ASSERT_TRUE(equal_2d(eroded, expected));
}

TEST(CSIR2DTest, Expand)
{
    auto src = make_rectangle(3, 2, 4, 5, 6);
    std::array<bool, 2> mask{true, false};
    auto dilated  = expand(src, 1, mask);
    auto expected = make_rectangle(3, 1, 5, 5, 6);
    ASSERT_TRUE(equal_2d(dilated, expected));
}

// Pathological cases for dilation/erosion in 2D
TEST(CSIR2DTest, ExpandX_MergeNearlyTouching)
{
    // Two horizontal segments on same row with gap 2, width=1 along X should merge
    CSIR_Level src;
    src.level = 3;
    src.intervals_ptr.push_back(0);
    src.y_coords.push_back(0);
    src.intervals.push_back({10, 12});
    src.intervals.push_back({14, 16});
    src.intervals_ptr.push_back(src.intervals.size());

    std::array<bool, 2> mask{true, false};
    auto dilated = expand(src, 1, mask);

    CSIR_Level expected;
    expected.level = 3;
    expected.intervals_ptr.push_back(0);
    expected.y_coords.push_back(0);
    expected.intervals.push_back({9, 17});
    expected.intervals_ptr.push_back(expected.intervals.size());

    ASSERT_TRUE(equal_2d(dilated, expected));
}

TEST(CSIR2DTest, ExpandX_NoMergeWhenGapLarge)
{
    // Gap 5 with width=2 along X should not merge
    CSIR_Level src;
    src.level = 3;
    src.intervals_ptr.push_back(0);
    src.y_coords.push_back(0);
    src.intervals.push_back({10, 12});
    src.intervals.push_back({17, 19});
    src.intervals_ptr.push_back(src.intervals.size());

    std::array<bool, 2> mask{true, false};
    auto dilated = expand(src, 2, mask);

    CSIR_Level expected;
    expected.level = 3;
    expected.intervals_ptr.push_back(0);
    expected.y_coords.push_back(0);
    expected.intervals.push_back({8, 14});
    expected.intervals.push_back({15, 21});
    expected.intervals_ptr.push_back(expected.intervals.size());

    ASSERT_TRUE(equal_2d(dilated, expected));
}

TEST(CSIR2DTest, ExpandY_MergeNearlyTouching)
{
    // Two identical rows with a gap of 2 in Y, width=1 along Y should merge
    CSIR_Level src;
    src.level = 2;
    src.intervals_ptr.push_back(0);
    src.y_coords.push_back(10);
    src.intervals.push_back({20, 22});
    src.intervals_ptr.push_back(src.intervals.size());
    src.y_coords.push_back(12);
    src.intervals.push_back({20, 22});
    src.intervals_ptr.push_back(src.intervals.size());

    std::array<bool, 2> mask{false, true};
    auto dilated = expand(src, 1, mask);

    CSIR_Level expected;
    expected.level = 2;
    expected.intervals_ptr.push_back(0);
    for (int y = 9; y <= 13; ++y)
    {
        expected.y_coords.push_back(y);
        expected.intervals.push_back({20, 22});
        expected.intervals_ptr.push_back(expected.intervals.size());
    }

    ASSERT_TRUE(equal_2d(dilated, expected));
}

TEST(CSIR2DTest, Contract_WidthTwoInXDisappears)
{
    // A 2x5 strip eroded by width=1 in both axes should disappear
    auto src    = make_rectangle(2, 10, 12, 0, 5);
    auto eroded = contract(src, 1);
    ASSERT_TRUE(eroded.intervals.empty());
}

TEST(CSIR2DTest, Contract_WidthOneInXDisappears)
{
    // A 1x5 strip eroded by width=1 in both axes should disappear
    auto src    = make_rectangle(2, 10, 11, 0, 5);
    auto eroded = contract(src, 1);
    ASSERT_TRUE(eroded.intervals.empty());
}

TEST(CSIR2DTest, Contract_ThreeByThreeReducesToCenter)
{
    // A 3x3 square eroded by width=1 reduces to a 1x1 center
    auto src    = make_rectangle(2, 10, 13, 5, 8);
    auto eroded = contract(src, 1);

    CSIR_Level expected;
    expected.level = 2;
    expected.intervals_ptr.push_back(0);
    expected.y_coords.push_back(6);
    expected.intervals.push_back({11, 12});
    expected.intervals_ptr.push_back(expected.intervals.size());
    ASSERT_TRUE(equal_2d(eroded, expected));
}

TEST(CSIR2DTest, DownProjection)
{
    CSIR_Level fine;
    fine.level = 3;
    fine.intervals_ptr.push_back(0);
    for (int y = 0; y < 4; ++y)
    {
        fine.y_coords.push_back(y);
        fine.intervals.push_back({0, 4});
        fine.intervals_ptr.push_back(fine.intervals.size());
    }
    auto coarse = project_to_level(fine, 2);
    CSIR_Level expected;
    expected.level = 2;
    expected.intervals_ptr.push_back(0);
    for (int y = 0; y < 2; ++y)
    {
        expected.y_coords.push_back(y);
        expected.intervals.push_back({0, 2});
        expected.intervals_ptr.push_back(expected.intervals.size());
    }
    ASSERT_TRUE(equal_2d(coarse, expected));
}

TEST(CSIR2DTest, CrossLevelUnionDifferenceReturnEmpty)
{
    CSIR_Level a = make_rectangle(2, 0, 4, 0, 3);
    CSIR_Level b = make_rectangle(3, 0, 8, 0, 6);
    ASSERT_TRUE(union_(a, b).intervals.empty());
    ASSERT_TRUE(difference(a, b).intervals.empty());
}

TEST(CSIR2DTest, CrossLevelIntersectionMayBeNonEmpty)
{
    // Current implementation allows intersection across levels
    CSIR_Level a = make_rectangle(2, 0, 4, 0, 3);
    CSIR_Level b = make_rectangle(3, 0, 8, 0, 6);
    ASSERT_FALSE(intersection(a, b).intervals.empty());
}

TEST(CSIR2DTest, Expand_ComposesWithUpscaleXOnly)
{
    auto s             = make_rectangle(1, 2, 6, 3, 5); // width=4, height=2
    std::size_t target = 3;
    int scale          = 1 << (target - s.level);
    std::array<bool, 2> maskX{true, false};

    auto coarse_exp = expand(s, 1, maskX);
    auto up_of_exp  = project_to_level(coarse_exp, target);
    auto up         = project_to_level(s, target);
    auto exp_up     = expand(up, 1 * scale, maskX);
    ASSERT_TRUE(equal_2d(up_of_exp, exp_up));
}

TEST(CSIR2DTest, Contract_ComposesWithUpscaleBothAxes)
{
    auto s             = make_rectangle(3, -4, 4, -3, 3);
    std::size_t target = 5;
    int scale          = 1 << (target - s.level);
    auto coarse_con    = contract(s, 1);
    auto up_of_con     = project_to_level(coarse_con, target);
    auto up            = project_to_level(s, target);
    auto con_up        = contract(up, 1 * scale);
    ASSERT_TRUE(equal_2d(up_of_con, con_up));
}

TEST(CSIR2DTest, UnionAndDifference_DistributeOverUpscale)
{
    auto a             = make_rectangle(2, 0, 6, 0, 4);
    auto b             = make_rectangle(2, 3, 9, 2, 6);
    std::size_t target = 4;
    auto up_u          = project_to_level(union_(a, b), target);
    auto u_up          = union_(project_to_level(a, target), project_to_level(b, target));
    ASSERT_TRUE(equal_2d(up_u, u_up));

    auto up_d = project_to_level(difference(a, b), target);
    auto d_up = difference(project_to_level(a, target), project_to_level(b, target));
    ASSERT_TRUE(equal_2d(up_d, d_up));
}

TEST(CSIR2DTest, ProjectionRoundTripAligned)
{
    CSIR_Level s;
    s.level = 5;
    s.intervals_ptr.push_back(0);
    for (int y = -16; y < -12; ++y)
    {
        s.y_coords.push_back(y);
        s.intervals.push_back({8, 16});
        s.intervals_ptr.push_back(s.intervals.size());
    }
    auto down = project_to_level(s, 3);
    auto up   = project_to_level(down, 5);
    ASSERT_TRUE(equal_2d(s, up));
}
