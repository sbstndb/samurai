#include <gtest/gtest.h>
#include <xtensor/xarray.hpp>
#include <xtensor/xfixed.hpp>

#include <samurai/cell.hpp>
#include <samurai/interval.hpp>

namespace samurai
{
    TEST(cell, length)
    {
        auto indices          = xt::xtensor_fixed<int, xt::xshape<2>>({1, 1});
        double scaling_factor = 1;
        Cell<2, Interval<int>> c{
            {0, 0},
            scaling_factor,
            1,
            indices,
            0
        };
        EXPECT_EQ(c.length, 0.5);
    }

    TEST(cell, center)
    {
        auto indices          = xt::xtensor_fixed<int, xt::xshape<2>>({1, 1});
        double scaling_factor = 1;
        Cell<2, Interval<int>> c{
            {0, 0},
            scaling_factor,
            1,
            indices,
            0
        };
        xt::xarray<double> expected{.75, .75};
        EXPECT_EQ(c.center(), expected);
    }

    TEST(cell, first_corner)
    {
        auto indices          = xt::xtensor_fixed<int, xt::xshape<2>>({1, 1});
        double scaling_factor = 1;
        Cell<2, Interval<int>> c{
            {0, 0},
            scaling_factor,
            1,
            indices,
            0
        };
        xt::xarray<double> expected{.5, .5};
        EXPECT_EQ(c.corner(), expected);
    }

    TEST(cell, corner_coord)
    {
        auto indices          = xt::xtensor_fixed<int, xt::xshape<2>>({1, 1});
        double scaling_factor = 1;
        Cell<2, Interval<int>> c{
            {0, 0},
            scaling_factor,
            1,
            indices,
            0
        };
        EXPECT_EQ(c.corner(0), 0.5);
        EXPECT_EQ(c.corner(1), 0.5);
    }

    TEST(cell, center_coord)
    {
        auto indices          = xt::xtensor_fixed<int, xt::xshape<2>>({1, 1});
        double scaling_factor = 1;
        Cell<2, Interval<int>> c{
            {0, 0},
            scaling_factor,
            1,
            indices,
            0
        };
        EXPECT_EQ(c.center(0), 0.75);
        EXPECT_EQ(c.center(1), 0.75);
    }

    TEST(cell, face_center)
    {
        auto indices          = xt::xtensor_fixed<int, xt::xshape<2>>({1, 1});
        double scaling_factor = 1;
        Cell<2, Interval<int>> c{
            {0, 0},
            scaling_factor,
            1,
            indices,
            0
        };

        // Test face center in x-direction (right face)
        xt::xtensor_fixed<int, xt::xshape<2>> right_direction{1, 0};
        xt::xarray<double> expected_right{1.0, 0.75};
        EXPECT_EQ(c.face_center(right_direction), expected_right);

        // Test face center in y-direction (top face)
        xt::xtensor_fixed<int, xt::xshape<2>> top_direction{0, 1};
        xt::xarray<double> expected_top{0.75, 1.0};
        EXPECT_EQ(c.face_center(top_direction), expected_top);

        // Test face center in negative x-direction (left face)
        xt::xtensor_fixed<int, xt::xshape<2>> left_direction{-1, 0};
        xt::xarray<double> expected_left{0.5, 0.75};
        EXPECT_EQ(c.face_center(left_direction), expected_left);

        // Test face center in negative y-direction (bottom face)
        xt::xtensor_fixed<int, xt::xshape<2>> bottom_direction{0, -1};
        xt::xarray<double> expected_bottom{0.75, 0.5};
        EXPECT_EQ(c.face_center(bottom_direction), expected_bottom);
    }

    TEST(cell, equality_operators)
    {
        auto indices1         = xt::xtensor_fixed<int, xt::xshape<2>>({1, 1});
        auto indices2         = xt::xtensor_fixed<int, xt::xshape<2>>({1, 1});
        auto indices3         = xt::xtensor_fixed<int, xt::xshape<2>>({2, 1});
        double scaling_factor = 1;

        Cell<2, Interval<int>> c1{
            {0, 0},
            scaling_factor,
            1,
            indices1,
            0
        };

        Cell<2, Interval<int>> c2{
            {0, 0},
            scaling_factor,
            1,
            indices2,
            0
        };

        Cell<2, Interval<int>> c3{
            {0, 0},
            scaling_factor,
            1,
            indices3,
            0
        };

        Cell<2, Interval<int>> c4{
            {0, 0},
            scaling_factor,
            2,
            indices1,
            0
        };

        // Test equality
        EXPECT_TRUE(c1 == c2);
        EXPECT_FALSE(c1 == c3);
        EXPECT_FALSE(c1 == c4);

        // Test inequality
        EXPECT_FALSE(c1 != c2);
        EXPECT_TRUE(c1 != c3);
        EXPECT_TRUE(c1 != c4);
    }

    TEST(cell, second_constructor)
    {
        double scaling_factor = 1;
        int i                 = 1;
        xt::xtensor_fixed<int, xt::xshape<1>> others{1};

        Cell<2, Interval<int>> c{
            {0, 0},
            scaling_factor,
            1,
            i,
            others,
            0
        };

        // Check that the indices were set correctly
        EXPECT_EQ(c.indices[0], 1);
        EXPECT_EQ(c.indices[1], 1);

        // Check other properties
        EXPECT_EQ(c.level, 1);
        EXPECT_EQ(c.length, 0.5);
        EXPECT_EQ(c.index, 0);

        xt::xarray<double> expected_center{0.75, 0.75};
        EXPECT_EQ(c.center(), expected_center);
    }

    TEST(cell, cell_length_function)
    {
        double scaling_factor = 2.0;

        // Test level 0 (no refinement)
        EXPECT_EQ(cell_length(scaling_factor, 0), 2.0);

        // Test level 1 (2x refinement)
        EXPECT_EQ(cell_length(scaling_factor, 1), 1.0);

        // Test level 2 (4x refinement)
        EXPECT_EQ(cell_length(scaling_factor, 2), 0.5);

        // Test level 3 (8x refinement)
        EXPECT_EQ(cell_length(scaling_factor, 3), 0.25);
    }

    TEST(cell, one_dimensional)
    {
        auto indices          = xt::xtensor_fixed<int, xt::xshape<1>>({2});
        double scaling_factor = 1;
        Cell<1, Interval<int>> c{{0}, scaling_factor, 1, indices, 0};

        EXPECT_EQ(c.length, 0.5);

        xt::xarray<double> expected_center{1.25};
        EXPECT_EQ(c.center(), expected_center);

        xt::xarray<double> expected_corner{1.0};
        EXPECT_EQ(c.corner(), expected_corner);

        EXPECT_EQ(c.center(0), 1.25);
        EXPECT_EQ(c.corner(0), 1.0);
    }

    TEST(cell, three_dimensional)
    {
        auto indices          = xt::xtensor_fixed<int, xt::xshape<3>>({1, 2, 3});
        double scaling_factor = 1;
        Cell<3, Interval<int>> c{
            {0, 0, 0},
            scaling_factor,
            1,
            indices,
            0
        };

        EXPECT_EQ(c.length, 0.5);

        xt::xarray<double> expected_center{0.75, 1.25, 1.75};
        EXPECT_EQ(c.center(), expected_center);

        xt::xarray<double> expected_corner{0.5, 1.0, 1.5};
        EXPECT_EQ(c.corner(), expected_corner);

        EXPECT_EQ(c.center(0), 0.75);
        EXPECT_EQ(c.center(1), 1.25);
        EXPECT_EQ(c.center(2), 1.75);

        EXPECT_EQ(c.corner(0), 0.5);
        EXPECT_EQ(c.corner(1), 1.0);
        EXPECT_EQ(c.corner(2), 1.5);
    }
}
