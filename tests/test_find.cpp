#include <gtest/gtest.h>

#include <samurai/box.hpp>
#include <samurai/field.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>

namespace samurai
{
    TEST(find, find_cell_mr_mesh)
    {
        static constexpr std::size_t dim = 2;
        using Config                     = samurai::MRConfig<dim>;
        using Box                        = samurai::Box<double, dim>;
        using Mesh                       = samurai::MRMesh<Config>;
        using coords_t                   = xt::xtensor_fixed<double, xt::xshape<dim>>;

        Box box({-1., -1.}, {1., 1.});

        std::size_t min_level = 2;
        std::size_t max_level = 6;

        Mesh mesh{box, min_level, max_level};

        auto u = samurai::make_scalar_field<double>("u",
                                                    mesh,
                                                    [](const auto& coords)
                                                    {
                                                        const auto& x = coords(0);
                                                        const auto& y = coords(1);
                                                        return (x >= -0.8 && x <= -0.3 && y >= 0.3 && y <= 0.8) ? 1. : 0.;
                                                    });

        auto MRadaptation = samurai::make_MRAdapt(u);
        auto mra_config   = samurai::mra_config().epsilon(1e-3);
        MRadaptation(mra_config);

        coords_t coords = {0.4, 0.8};
        auto cell       = samurai::find_cell(mesh, coords);

        EXPECT_TRUE(cell.length > 0);                                                             // cell found
        EXPECT_TRUE(static_cast<std::size_t>(cell.index) < mesh.nb_cells());                      // cell index makes sense
        EXPECT_TRUE(xt::all(cell.corner() <= coords && coords <= (cell.corner() + cell.length))); // coords in cell
    }

    TEST(find, find_cell_different_coordinates)
    {
        static constexpr std::size_t dim = 2;
        using Config                     = samurai::MRConfig<dim>;
        using Box                        = samurai::Box<double, dim>;
        using Mesh                       = samurai::MRMesh<Config>;
        using coords_t                   = xt::xtensor_fixed<double, xt::xshape<dim>>;

        Box box({-1., -1.}, {1., 1.});

        std::size_t min_level = 2;
        std::size_t max_level = 6;

        Mesh mesh{box, min_level, max_level};

        auto u = samurai::make_scalar_field<double>("u",
                                                    mesh,
                                                    [](const auto& coords)
                                                    {
                                                        const auto& x = coords(0);
                                                        const auto& y = coords(1);
                                                        return (x >= -0.8 && x <= -0.3 && y >= 0.3 && y <= 0.8) ? 1. : 0.;
                                                    });

        auto MRadaptation = samurai::make_MRAdapt(u);
        auto mra_config   = samurai::mra_config().epsilon(1e-3);
        MRadaptation(mra_config);

        // Test at corner of domain
        coords_t corner_coords = {-1.0, -1.0};
        auto corner_cell       = samurai::find_cell(mesh, corner_coords);

        EXPECT_TRUE(corner_cell.length > 0); // cell found
        EXPECT_TRUE(xt::all(corner_cell.corner() <= corner_coords && corner_coords <= (corner_cell.corner() + corner_cell.length))); // coords
                                                                                                                                     // in
                                                                                                                                     // cell

        // Test at center of domain
        coords_t center_coords = {0.0, 0.0};
        auto center_cell       = samurai::find_cell(mesh, center_coords);

        EXPECT_TRUE(center_cell.length > 0); // cell found
        EXPECT_TRUE(xt::all(center_cell.corner() <= center_coords && center_coords <= (center_cell.corner() + center_cell.length))); // coords
                                                                                                                                     // in
                                                                                                                                     // cell
    }

    TEST(find, find_cell_not_found)
    {
        static constexpr std::size_t dim = 2;
        using Config                     = samurai::MRConfig<dim>;
        using Box                        = samurai::Box<double, dim>;
        using Mesh                       = samurai::MRMesh<Config>;
        using coords_t                   = xt::xtensor_fixed<double, xt::xshape<dim>>;

        Box box({-1., -1.}, {1., 1.});

        std::size_t min_level = 2;
        std::size_t max_level = 6;

        Mesh mesh{box, min_level, max_level};

        auto u = samurai::make_scalar_field<double>("u",
                                                    mesh,
                                                    [](const auto& coords)
                                                    {
                                                        const auto& x = coords(0);
                                                        const auto& y = coords(1);
                                                        return (x >= -0.8 && x <= -0.3 && y >= 0.3 && y <= 0.8) ? 1. : 0.;
                                                    });

        auto MRadaptation = samurai::make_MRAdapt(u);
        auto mra_config   = samurai::mra_config().epsilon(1e-3);
        MRadaptation(mra_config);

        // Test coordinates that are outside the domain
        coords_t coords = {2.0, 2.0}; // Outside the [-1, 1] domain
        auto cell       = samurai::find_cell(mesh, coords);

        EXPECT_TRUE(cell.length == 0); // cell not found
    }

    TEST(find, find_cell_3d)
    {
        static constexpr std::size_t dim = 3;
        using Config                     = samurai::MRConfig<dim>;
        using Box                        = samurai::Box<double, dim>;
        using Mesh                       = samurai::MRMesh<Config>;
        using coords_t                   = xt::xtensor_fixed<double, xt::xshape<dim>>;

        Box box({-1., -1., -1.}, {1., 1., 1.});

        std::size_t min_level = 1;
        std::size_t max_level = 3;

        Mesh mesh{box, min_level, max_level};

        auto u = samurai::make_scalar_field<double>(
            "u",
            mesh,
            [](const auto& coords)
            {
                const auto& x = coords(0);
                const auto& y = coords(1);
                const auto& z = coords(2);
                return (x >= -0.5 && x <= 0.0 && y >= -0.5 && y <= 0.0 && z >= -0.5 && z <= 0.0) ? 1. : 0.;
            });

        auto MRadaptation = samurai::make_MRAdapt(u);
        auto mra_config   = samurai::mra_config().epsilon(1e-3);
        MRadaptation(mra_config);

        // Test finding a cell in 3D
        coords_t coords = {0.25, 0.75, -0.5};
        auto cell       = samurai::find_cell(mesh, coords);

        EXPECT_TRUE(cell.length > 0);                                                             // cell found
        EXPECT_TRUE(static_cast<std::size_t>(cell.index) < mesh.nb_cells());                      // cell index makes sense
        EXPECT_TRUE(xt::all(cell.corner() <= coords && coords <= (cell.corner() + cell.length))); // coords in cell
    }
}
