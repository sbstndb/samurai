#include "src/csir.hpp"
#include "test_utils.hpp"
#include "gtest/gtest.h"
#include <array>

using namespace csir;
using namespace csir_test_utils;

TEST(CSIR3DTest, Translate)
{
    auto src      = make_block3d(2, 1, 3, 2, 4, 5, 7); // [x:1,3) x [y:2,4) x [z:5,7)
    auto moved    = translate(src, 1, 2, -1);
    auto expected = make_block3d(2, 2, 4, 4, 6, 4, 6); // z becomes [4,6)
    ASSERT_TRUE(equal_3d(moved, expected));
}

TEST(CSIR3DTest, Contract)
{
    auto src      = make_block3d(1, 0, 3, 0, 3, 0, 3);
    auto eroded   = contract(src, 1);
    auto expected = make_block3d(1, 1, 2, 1, 2, 1, 2);
    ASSERT_TRUE(equal_3d(eroded, expected));
}

TEST(CSIR3DTest, Expand)
{
    auto src = make_block3d(3, 10, 12, 20, 21, 7, 8); // single z at 7
    std::array<bool, 3> mask{false, false, true};
    auto dilated  = expand(src, 1, mask);
    auto expected = make_block3d(3, 10, 12, 20, 21, 6, 9); // z = {6,7,8}
    ASSERT_TRUE(equal_3d(dilated, expected));
}

TEST(CSIR3DTest, DownProjection)
{
    auto fine     = make_block3d(3, 0, 8, 0, 8, 0, 8);
    auto coarse   = project_to_level_3d(fine, 2);
    auto expected = make_block3d(2, 0, 4, 0, 4, 0, 4);
    ASSERT_TRUE(equal_3d(coarse, expected));
}
