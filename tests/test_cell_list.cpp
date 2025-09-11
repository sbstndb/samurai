#include <gtest/gtest.h>
#include <xtensor/xfixed.hpp>

#include <samurai/cell.hpp>
#include <samurai/cell_list.hpp>

namespace samurai
{
    TEST(cell_list, constructor)
    {
        constexpr size_t dim = 2;

        CellList<dim> cell_list;
    }

    TEST(cell_list, constructor_with_parameters)
    {
        constexpr size_t dim = 2;
        xt::xtensor_fixed<double, xt::xshape<dim>> origin_point{0.5, 0.5};
        double scaling_factor = 2.0;

        CellList<dim> cell_list(origin_point, scaling_factor);

        // Check that the origin point and scaling factor are correctly set
        EXPECT_EQ(cell_list.origin_point()[0], origin_point[0]);
        EXPECT_EQ(cell_list.origin_point()[1], origin_point[1]);
        EXPECT_EQ(cell_list.scaling_factor(), scaling_factor);
    }

    TEST(cell_list, access_operator)
    {
        constexpr size_t dim = 2;
        CellList<dim> cell_list;

        // Test accessing different levels
        for (std::size_t level = 0; level <= cell_list.max_size; ++level)
        {
            auto& level_cell_list = cell_list[level];
            EXPECT_EQ(level_cell_list.level(), level);
        }
    }

    TEST(cell_list, origin_point_and_scaling_factor)
    {
        constexpr size_t dim = 2;
        xt::xtensor_fixed<double, xt::xshape<dim>> origin_point{1.0, 2.0};
        double scaling_factor = 0.5;

        CellList<dim> cell_list(origin_point, scaling_factor);

        // Check that origin point and scaling factor are consistent across all levels
        for (std::size_t level = 0; level <= cell_list.max_size; ++level)
        {
            auto& level_cell_list = cell_list[level];
            EXPECT_EQ(level_cell_list.origin_point()[0], origin_point[0]);
            EXPECT_EQ(level_cell_list.origin_point()[1], origin_point[1]);
            EXPECT_EQ(level_cell_list.scaling_factor(), scaling_factor);
        }

        // Check through the CellList methods
        EXPECT_EQ(cell_list.origin_point()[0], origin_point[0]);
        EXPECT_EQ(cell_list.origin_point()[1], origin_point[1]);
        EXPECT_EQ(cell_list.scaling_factor(), scaling_factor);
    }

    TEST(cell_list, clear)
    {
        constexpr size_t dim = 2;
        CellList<dim> cell_list;

        // Add some data to level 0
        cell_list[0][{0}].add_interval({-2, 2});

        // Verify data was added
        EXPECT_FALSE(cell_list[0].empty());

        // Clear the cell list
        cell_list.clear();

        // Verify all levels are empty
        for (std::size_t level = 0; level <= cell_list.max_size; ++level)
        {
            EXPECT_TRUE(cell_list[level].empty());
        }
    }

    TEST(cell_list, empty_initialization)
    {
        constexpr size_t dim = 2;
        CellList<dim> cell_list;

        // All levels should be empty initially
        for (std::size_t level = 0; level <= cell_list.max_size; ++level)
        {
            EXPECT_TRUE(cell_list[level].empty());
        }
    }

    TEST(cell_list, add_cell)
    {
        constexpr size_t dim = 2;
        CellList<dim> cell_list;

        // Create a cell
        xt::xtensor_fixed<double, xt::xshape<dim>> origin_point{0.0, 0.0};
        double scaling_factor = 1.0;
        std::size_t level     = 2;
        xt::xtensor_fixed<int, xt::xshape<dim>> indices{1, 2};
        default_config::index_t index = 5;

        Cell<dim> cell(origin_point, scaling_factor, level, indices, index);

        // Add cell to the list
        cell_list[level].add_cell(cell);

        // Verify the cell was added
        xt::xtensor_fixed<int, xt::xshape<dim - 1>> yz_indices{2}; // indices for y,z dimensions
        auto& interval_list = cell_list[level][yz_indices];
        EXPECT_EQ(interval_list.size(), 1);
    }

    TEST(cell_list, max_size)
    {
        constexpr size_t dim      = 2;
        constexpr size_t max_size = default_config::max_level;
        CellList<dim> cell_list;

        // Verify max_size is correctly set
        EXPECT_EQ(cell_list.max_size, max_size);

        // Verify we can access all levels up to max_size
        for (std::size_t level = 0; level <= max_size; ++level)
        {
            // This should not throw
            auto& level_cell_list = cell_list[level];
            EXPECT_EQ(level_cell_list.level(), level);
        }
    }

    TEST(cell_list, ostream_output)
    {
        constexpr size_t dim = 2;
        CellList<dim> cell_list;

        // Add some data to make the output more interesting
        cell_list[0][{0}].add_interval({-1, 1});
        cell_list[1][{0}].add_interval({-2, 2});

        // Test that we can stream the cell list without throwing
        std::ostringstream oss;
        oss << cell_list;

        // Basic check that output is not empty
        EXPECT_FALSE(oss.str().empty());
        EXPECT_NE(oss.str().find("Level 0"), std::string::npos);
        EXPECT_NE(oss.str().find("Level 1"), std::string::npos);
    }
}
