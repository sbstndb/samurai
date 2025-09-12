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

// Pathological cases for dilation/erosion in 1D
TEST(CSIR1DTest, Expand_MergeNearlyTouching)
{
    // Two intervals with gap 3, width=2 should merge (gap <= 2w)
    CSIR_Level_1D s = make_1d(4, {{10, 12}, {15, 17}});
    auto dilated = expand(s, 2);
    CSIR_Level_1D expected = make_1d(4, {{8, 19}});
    ASSERT_TRUE(equal_1d(dilated, expected));
}

TEST(CSIR1DTest, Expand_MergeAtBoundaryGapEq2w)
{
    // Two intervals with gap 2, width=1 should merge exactly at boundary
    CSIR_Level_1D s = make_1d(4, {{10, 12}, {14, 16}});
    auto dilated = expand(s, 1);
    CSIR_Level_1D expected = make_1d(4, {{9, 17}});
    ASSERT_TRUE(equal_1d(dilated, expected));
}

TEST(CSIR1DTest, Expand_NoMergeWhenGapLarge)
{
    // Gap 5 with width=2 should not merge (gap > 2w)
    CSIR_Level_1D s = make_1d(4, {{10, 12}, {17, 19}});
    auto dilated = expand(s, 2);
    CSIR_Level_1D expected = make_1d(4, {{8, 14}, {15, 21}});
    ASSERT_TRUE(equal_1d(dilated, expected));
}

TEST(CSIR1DTest, Contract_TwoCellsBecomesEmpty)
{
    // Length-2 interval eroded by width=1 disappears
    CSIR_Level_1D s = make_1d(4, {{10, 12}});
    auto eroded = contract(s, 1);
    CSIR_Level_1D expected = make_1d(4, {});
    ASSERT_TRUE(equal_1d(eroded, expected));
}

TEST(CSIR1DTest, Contract_OneCellBecomesEmpty)
{
    // Length-1 interval eroded by width=1 disappears
    CSIR_Level_1D s = make_1d(4, {{10, 11}});
    auto eroded = contract(s, 1);
    CSIR_Level_1D expected = make_1d(4, {});
    ASSERT_TRUE(equal_1d(eroded, expected));
}

TEST(CSIR1DTest, Contract_ThreeCellsReducesToOne)
{
    // Length-3 interval eroded by width=1 reduces to central single cell
    CSIR_Level_1D s = make_1d(4, {{10, 13}});
    auto eroded = contract(s, 1);
    CSIR_Level_1D expected = make_1d(4, {{11, 12}});
    ASSERT_TRUE(equal_1d(eroded, expected));
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

TEST(CSIR1DTest, CrossLevelBinaryOpsReturnEmpty)
{
    CSIR_Level_1D a = make_1d(2, {{0, 10}});
    CSIR_Level_1D b = make_1d(3, {{0, 10}});
    ASSERT_TRUE(union_(a, b).intervals.empty());
    ASSERT_TRUE(difference(a, b).intervals.empty());
    ASSERT_TRUE(intersection(a, b).intervals.empty());
}

TEST(CSIR1DTest, Expand_ComposesWithUpscale)
{
    // Expand at coarse then upscale == upscale then expand with scaled width
    CSIR_Level_1D s = make_1d(2, {{5, 9}, {15, 18}});
    std::size_t target = 4; int scale = 1 << (target - s.level);
    auto coarse_exp = expand(s, 2);
    auto up_of_exp = project_to_level(coarse_exp, target);
    auto up = project_to_level(s, target);
    auto exp_up = expand(up, 2 * scale);
    ASSERT_TRUE(equal_1d(up_of_exp, exp_up));
}

TEST(CSIR1DTest, Contract_ComposesWithUpscale)
{
    CSIR_Level_1D s = make_1d(3, {{-5, 5}, {12, 20}});
    std::size_t target = 5; int scale = 1 << (target - s.level);
    auto coarse_con = contract(s, 1);
    auto up_of_con = project_to_level(coarse_con, target);
    auto up = project_to_level(s, target);
    auto con_up = contract(up, 1 * scale);
    ASSERT_TRUE(equal_1d(up_of_con, con_up));
}

TEST(CSIR1DTest, UnionAndDifference_DistributeOverUpscale)
{
    CSIR_Level_1D a = make_1d(1, {{0, 4}});
    CSIR_Level_1D b = make_1d(1, {{2, 6}});
    std::size_t target = 4;
    auto up_u = project_to_level(union_(a, b), target);
    auto u_up = union_(project_to_level(a, target), project_to_level(b, target));
    ASSERT_TRUE(equal_1d(up_u, u_up));

    auto up_d = project_to_level(difference(a, b), target);
    auto d_up = difference(project_to_level(a, target), project_to_level(b, target));
    ASSERT_TRUE(equal_1d(up_d, d_up));
}

TEST(CSIR1DTest, ProjectionRoundTripAlignedIntervals)
{
    // Coordinates aligned with scale retain exact shape on down-then-up
    CSIR_Level_1D s = make_1d(5, {{-16, -8}, {0, 8}, {24, 32}});
    auto down = project_to_level(s, 3);
    auto up = project_to_level(down, 5);
    ASSERT_TRUE(equal_1d(s, up));
}