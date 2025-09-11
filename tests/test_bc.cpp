#include <gtest/gtest.h>

#include <samurai/bc.hpp>
#include <samurai/field.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/uniform_mesh.hpp>

#include <xtensor/xtensor.hpp>

namespace samurai
{
    TEST(bc, scalar_homogeneous)
    {
        static constexpr std::size_t dim = 1;
        using config                     = UniformConfig<dim>;
        auto mesh                        = UniformMesh<config>({{0}, {1}}, 4);
        auto u                           = make_scalar_field<double>("u", mesh);

        make_bc<Dirichlet<1>>(u);
        EXPECT_EQ(u.get_bc()[0]->constant_value(), 0.);
    }

    TEST(bc, vec_homogeneous)
    {
        static constexpr std::size_t dim = 1;
        using config                     = UniformConfig<dim>;
        auto mesh                        = UniformMesh<config>({{0}, {1}}, 4);
        auto u                           = make_vector_field<double, 4>("u", mesh);

        make_bc<Dirichlet<1>>(u);
        EXPECT_TRUE(compare(u.get_bc()[0]->constant_value(), zeros<double>(4)));
    }

    TEST(bc, scalar_constant_value)
    {
        static constexpr std::size_t dim = 1;
        using config                     = UniformConfig<dim>;
        auto mesh                        = UniformMesh<config>({{0}, {1}}, 4);
        auto u                           = make_scalar_field<double>("u", mesh);

        make_bc<Dirichlet<1>>(u, 2);
        EXPECT_EQ(u.get_bc()[0]->constant_value(), 2);
    }

    TEST(bc, vec_constant_value)
    {
        static constexpr std::size_t dim = 1;
        using config                     = UniformConfig<dim>;
        auto mesh                        = UniformMesh<config>({{0}, {1}}, 4);
        auto u                           = make_vector_field<double, 4>("u", mesh);

        make_bc<Dirichlet<1>>(u, 1., 2., 3., 4.);
        samurai::Array<double, 4, false> expected({1, 2, 3, 4});
        EXPECT_TRUE(compare(u.get_bc()[0]->constant_value(), expected));
    }

    TEST(bc, scalar_function)
    {
        static constexpr std::size_t dim = 1;
        using config                     = MRConfig<dim>;

        Box<double, dim> box = {{0}, {1}};
        auto mesh            = MRMesh<config>(box, 2, 4);
        auto u               = make_scalar_field<double>("u", mesh);

        make_bc<Dirichlet<1>>(u,
                              [](const auto&, const auto&, const auto&)
                              {
                                  return 0;
                              });

        using cell_t   = typename decltype(u)::cell_t;
        using coords_t = typename cell_t::coords_t;
        cell_t cell;
        coords_t coords = {0.};
        EXPECT_EQ(u.get_bc()[0]->value({1}, cell, coords), 0);
    }

    TEST(bc, neumann_scalar_homogeneous)
    {
        static constexpr std::size_t dim = 1;
        using config                     = UniformConfig<dim>;
        auto mesh                        = UniformMesh<config>({{0}, {1}}, 4);
        auto u                           = make_scalar_field<double>("u", mesh);

        make_bc<Neumann<1>>(u);
        EXPECT_EQ(u.get_bc()[0]->constant_value(), 0.);
    }

    TEST(bc, neumann_scalar_constant_value)
    {
        static constexpr std::size_t dim = 1;
        using config                     = UniformConfig<dim>;
        auto mesh                        = UniformMesh<config>({{0}, {1}}, 4);
        auto u                           = make_scalar_field<double>("u", mesh);

        make_bc<Neumann<1>>(u, 2.5);
        EXPECT_EQ(u.get_bc()[0]->constant_value(), 2.5);
    }

    TEST(bc, dirichlet_order2_scalar)
    {
        static constexpr std::size_t dim = 1;
        // For order 2, we need ghost_width >= 2
        using config = UniformConfig<dim, 2>;
        auto mesh    = UniformMesh<config>({{0}, {1}}, 4);
        auto u       = make_scalar_field<double>("u", mesh);

        make_bc<Dirichlet<2>>(u, 3.0);
        EXPECT_EQ(u.get_bc()[0]->constant_value(), 3.0);
    }

    TEST(bc, dirichlet_order2_vector)
    {
        static constexpr std::size_t dim = 1;
        // For order 2, we need ghost_width >= 2
        using config = UniformConfig<dim, 2>;
        auto mesh    = UniformMesh<config>({{0}, {1}}, 4);
        auto u       = make_vector_field<double, 3>("u", mesh);

        make_bc<Dirichlet<2>>(u, 1., 2., 3.);
        samurai::Array<double, 3, false> expected({1, 2, 3});
        EXPECT_TRUE(compare(u.get_bc()[0]->constant_value(), expected));
    }

    TEST(bc, dirichlet_order3_scalar)
    {
        static constexpr std::size_t dim = 1;
        // For order 3, we need ghost_width >= 3
        using config = UniformConfig<dim, 3>;
        auto mesh    = UniformMesh<config>({{0}, {1}}, 4);
        auto u       = make_scalar_field<double>("u", mesh);

        make_bc<Dirichlet<3>>(u, 4.0);
        EXPECT_EQ(u.get_bc()[0]->constant_value(), 4.0);
    }

    TEST(bc, bc_2d_mesh)
    {
        static constexpr std::size_t dim = 2;
        using config                     = UniformConfig<dim>;
        auto mesh                        = UniformMesh<config>(
            {
                {0, 0},
                {1, 1}
        },
            3);
        auto u = make_scalar_field<double>("u", mesh);

        make_bc<Dirichlet<1>>(u, 5.0);
        EXPECT_EQ(u.get_bc()[0]->constant_value(), 5.0);
    }

    TEST(bc, bc_value_function)
    {
        static constexpr std::size_t dim = 1;
        using config                     = UniformConfig<dim>;
        auto mesh                        = UniformMesh<config>({{0}, {1}}, 4);
        auto u                           = make_scalar_field<double>("u", mesh);

        make_bc<Dirichlet<1>>(u,
                              [](const auto&, const auto&, const auto&)
                              {
                                  return 42;
                              });

        using cell_t   = typename decltype(u)::cell_t;
        using coords_t = typename cell_t::coords_t;
        cell_t cell;
        coords_t coords = {0.5};
        EXPECT_EQ(u.get_bc()[0]->value({1}, cell, coords), 42);
    }
}
