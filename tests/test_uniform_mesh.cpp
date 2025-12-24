#include <gtest/gtest.h>

#include <samurai/algorithm/update.hpp>
#include <samurai/bc.hpp>
#include <samurai/box.hpp>
#include <samurai/field.hpp>
#include <samurai/uniform_mesh.hpp>

namespace samurai
{
    //==========================================================================
    //                    Basic Mesh Creation Tests
    //==========================================================================

    TEST(UniformMesh, CreateFromBox2D)
    {
        constexpr std::size_t dim = 2;
        Box<double, dim> box({0., 0.}, {1., 1.});

        auto config = mesh_config<dim>().level(5);
        auto mesh   = uniform::make_mesh(box, config);

        EXPECT_EQ(mesh.min_level(), 5);
        EXPECT_EQ(mesh.max_level(), 5);
        EXPECT_GT(mesh.nb_cells(), 0);
        EXPECT_EQ(mesh.nb_cells(), 32 * 32);  // 2^5 x 2^5 = 1024
    }

    TEST(UniformMesh, CreateFromBox1D)
    {
        constexpr std::size_t dim = 1;
        Box<double, dim> box({0.}, {1.});

        auto config = mesh_config<dim>().level(4);
        auto mesh   = uniform::make_mesh(box, config);

        EXPECT_EQ(mesh.min_level(), 4);
        EXPECT_EQ(mesh.max_level(), 4);
        EXPECT_EQ(mesh.nb_cells(), 16);  // 2^4 = 16
    }

    TEST(UniformMesh, CreateFromBox3D)
    {
        constexpr std::size_t dim = 3;
        Box<double, dim> box({0., 0., 0.}, {1., 1., 1.});

        auto config = mesh_config<dim>().level(3);
        auto mesh   = uniform::make_mesh(box, config);

        EXPECT_EQ(mesh.min_level(), 3);
        EXPECT_EQ(mesh.max_level(), 3);
        EXPECT_EQ(mesh.nb_cells(), 8 * 8 * 8);  // 2^3 x 2^3 x 2^3 = 512
    }

    //==========================================================================
    //                    mesh_config.level() Tests
    //==========================================================================

    TEST(MeshConfig, LevelMethod)
    {
        auto config = mesh_config<2>().level(7);

        EXPECT_EQ(config.min_level(), 7);
        EXPECT_EQ(config.max_level(), 7);
        EXPECT_EQ(config.start_level(), 7);
    }

    TEST(MeshConfig, LevelChaining)
    {
        auto config = mesh_config<2>()
                          .level(5)
                          .max_stencil_size(4)
                          .periodic(true);

        EXPECT_EQ(config.min_level(), 5);
        EXPECT_EQ(config.max_level(), 5);
        EXPECT_EQ(config.start_level(), 5);
        EXPECT_EQ(config.max_stencil_size(), 4);
        EXPECT_TRUE(config.periodic()[0]);
        EXPECT_TRUE(config.periodic()[1]);
    }

    //==========================================================================
    //                    Field Creation Tests
    //==========================================================================

    TEST(UniformMesh, ScalarField)
    {
        constexpr std::size_t dim = 2;
        Box<double, dim> box({0., 0.}, {1., 1.});

        auto config = mesh_config<dim>().level(4);
        auto mesh   = uniform::make_mesh(box, config);
        auto u      = make_scalar_field<double>("u", mesh);

        u.fill(1.0);

        for_each_cell(mesh,
                      [&](auto& cell)
                      {
                          EXPECT_DOUBLE_EQ(u[cell], 1.0);
                      });
    }

    TEST(UniformMesh, VectorField)
    {
        constexpr std::size_t dim = 2;
        Box<double, dim> box({0., 0.}, {1., 1.});

        auto config = mesh_config<dim>().level(4);
        auto mesh   = uniform::make_mesh(box, config);
        auto u      = make_vector_field<double, dim>("u", mesh);

        for_each_cell(mesh,
                      [&](auto& cell)
                      {
                          u[cell] = cell.center();
                      });

        // Verify values
        for_each_cell(mesh,
                      [&](auto& cell)
                      {
                          auto center = cell.center();
                          EXPECT_DOUBLE_EQ(u[cell][0], center[0]);
                          EXPECT_DOUBLE_EQ(u[cell][1], center[1]);
                      });
    }

    //==========================================================================
    //                    Ghost Update Tests
    //==========================================================================

    TEST(UniformMesh, GhostUpdate)
    {
        constexpr std::size_t dim = 2;
        Box<double, dim> box({0., 0.}, {1., 1.});

        auto config = mesh_config<dim>().level(4).max_stencil_size(2);
        auto mesh   = uniform::make_mesh(box, config);

        auto u = make_scalar_field<double>("u", mesh);
        make_bc<Dirichlet<1>>(u, 1.);

        u.fill(0.);

        update_ghost_uniform(u);

        EXPECT_TRUE(u.ghosts_updated());
    }

    //==========================================================================
    //                    Periodicity Tests
    //==========================================================================

    TEST(UniformMesh, Periodic2D)
    {
        constexpr std::size_t dim = 2;
        Box<double, dim> box({0., 0.}, {1., 1.});

        auto config = mesh_config<dim>()
                          .level(4)
                          .periodic(true);

        auto mesh = uniform::make_mesh(box, config);

        EXPECT_TRUE(mesh.is_periodic(0));
        EXPECT_TRUE(mesh.is_periodic(1));
        EXPECT_TRUE(mesh.is_periodic());
    }

    TEST(UniformMesh, PartialPeriodic)
    {
        constexpr std::size_t dim = 2;
        Box<double, dim> box({0., 0.}, {1., 1.});

        std::array<bool, dim> periodicity = {true, false};
        auto config                       = mesh_config<dim>()
                          .level(4)
                          .periodic(periodicity);

        auto mesh = uniform::make_mesh(box, config);

        EXPECT_TRUE(mesh.is_periodic(0));
        EXPECT_FALSE(mesh.is_periodic(1));
        EXPECT_TRUE(mesh.is_periodic());  // At least one direction is periodic
    }

    //==========================================================================
    //                    Empty Mesh Tests
    //==========================================================================

    TEST(UniformMesh, EmptyMesh)
    {
        constexpr std::size_t dim = 2;

        auto config = mesh_config<dim>().level(5);
        auto mesh   = uniform::make_empty_mesh(config);

        EXPECT_EQ(mesh.nb_cells(), 0);
    }

    //==========================================================================
    //                    Geometry Tests
    //==========================================================================

    TEST(UniformMesh, CellLength)
    {
        constexpr std::size_t dim = 2;
        Box<double, dim> box({0., 0.}, {1., 1.});

        auto config = mesh_config<dim>().level(5);
        auto mesh   = uniform::make_mesh(box, config);

        // Cell length at level 5 for domain [0,1] should be 1/32
        double expected_length = 1.0 / 32.0;
        EXPECT_NEAR(mesh.cell_length(5), expected_length, 1e-10);
    }

    TEST(UniformMesh, OriginPoint)
    {
        constexpr std::size_t dim = 2;
        Box<double, dim> box({0.5, 0.5}, {1.5, 1.5});

        auto config = mesh_config<dim>().level(4);
        auto mesh   = uniform::make_mesh(box, config);

        auto origin = mesh.origin_point();
        EXPECT_NEAR(origin[0], 0.5, 1e-10);
        EXPECT_NEAR(origin[1], 0.5, 1e-10);
    }

}  // namespace samurai
