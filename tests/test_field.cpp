#include <algorithm>
#include <cmath>

#include <gtest/gtest.h>

#include <samurai/box.hpp>
#include <samurai/field.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/uniform_mesh.hpp>

namespace samurai
{
    TEST(field, from_expr)
    {
        Box<double, 1> box{{0}, {1}};
        // using Config = MRConfig<1>;
        // auto mesh    = MRMesh<Config>(box, 3, 3);

        using Config = UniformConfig<1>;
        auto mesh    = UniformMesh<Config>(box, 3);

        auto u = make_scalar_field<double>("u", mesh);
        u.fill(1.);
        using field_t = decltype(u);
        field_t ue    = 5 + u;

        for_each_cell(mesh,
                      [&](auto cell)
                      {
                          EXPECT_EQ(ue[cell], 6);
                      });
    }

    TEST(field, copy_from_const)
    {
        Box<double, 1> box{{0}, {1}};
        using Config       = UniformConfig<1>;
        auto mesh          = UniformMesh<Config>(box, 3);
        const auto u_const = make_scalar_field<double>("uc", mesh, 1.);

        auto u = u_const;
        EXPECT_EQ(u.name(), u_const.name());
        EXPECT_TRUE(compare(u.array(), u_const.array()));
        EXPECT_EQ(u.mesh(), u_const.mesh());
        EXPECT_EQ(&(u.mesh()), &(u_const.mesh()));

        auto m              = holder(mesh);
        const auto u_const1 = make_scalar_field<double>("uc", m, 1.);
        auto u1             = u_const1;
        EXPECT_EQ(u1.name(), u_const1.name());
        EXPECT_TRUE(compare(u1.array(), u_const1.array()));
        EXPECT_EQ(u1.mesh(), u_const1.mesh());
    }

    TEST(field, copy_assignment)
    {
        Box<double, 1> box{{0}, {1}};
        using Config       = UniformConfig<1>;
        auto mesh1         = UniformMesh<Config>(box, 5);
        auto mesh2         = UniformMesh<Config>(box, 3);
        const auto u_const = make_scalar_field<double>("uc",
                                                       mesh1,
                                                       [](const auto& coords)
                                                       {
                                                           return coords[0];
                                                       });
        auto u             = make_scalar_field<double>("u",
                                           mesh2,
                                           [](const auto& coords)
                                           {
                                               return coords[0];
                                           });

        u = u_const;
        EXPECT_EQ(u.name(), u_const.name());
        EXPECT_TRUE(compare(u.array(), u_const.array()));
        EXPECT_EQ(u.mesh(), u_const.mesh());
        EXPECT_EQ(&(u.mesh()), &(u_const.mesh()));

        auto m1             = holder(mesh1);
        auto m2             = holder(mesh2);
        const auto u_const1 = make_scalar_field<double>("uc",
                                                        m1,
                                                        [](const auto& coords)
                                                        {
                                                            return coords[0];
                                                        });
        auto u1             = make_scalar_field<double>("u",
                                            m2,
                                            [](const auto& coords)
                                            {
                                                return coords[0];
                                            });
        u1                  = u_const1;
        EXPECT_EQ(u1.name(), u_const1.name());
        EXPECT_TRUE(compare(u1.array(), u_const1.array()));
        EXPECT_EQ(u1.mesh(), u_const1.mesh());
    }

    TEST(field, noalias_sum_view)
    {
        Box<double, 2> box{{0, 0}, {1, 1}};
        using Config    = UniformConfig<2>;
        auto mesh       = UniformMesh<Config>(box, 3);
        using mesh_id_t = typename decltype(mesh)::mesh_id_t;

        auto u = make_scalar_field<double>(
            "u", mesh, [](const auto& coords) { return coords[0] + 2.0 * coords[1]; });
        auto v = make_scalar_field<double>(
            "v", mesh, [](const auto& coords) { return 1.0 - coords[0] + coords[1]; });
        auto sum = make_scalar_field<double>("sum", mesh);

        for_each_interval(mesh[mesh_id_t::cells],
                          [&](std::size_t level, const auto& interval, const auto& index)
                          {
                              samurai::noalias(sum(level, interval, index)) =
                                  u(level, interval, index) + v(level, interval, index);
                          });

        for_each_cell(mesh,
                      [&](const auto& cell)
                      {
                          EXPECT_NEAR(sum[cell], u[cell] + v[cell], 1e-12);
                      });
    }

    TEST(field, strided_interval_assignment)
    {
        Box<double, 1> box{{0}, {1}};
        using Config    = UniformConfig<1>;
        auto mesh       = UniformMesh<Config>(box, 4);
        using mesh_id_t = typename decltype(mesh)::mesh_id_t;

        auto base = make_scalar_field<double>(
            "base", mesh, [](const auto& coords) { return coords[0]; });
        auto ref = base;

        std::vector<bool> touched(base.array().size(), false);

        for_each_interval(mesh[mesh_id_t::cells],
                          [&](std::size_t level, auto interval, auto&)
                          {
                              auto even = interval.even_elements();
                              if (even.is_empty())
                              {
                                  return;
                              }
                              auto even_view = base(level, even);
                              auto* base_ptr = base.array().data();
                              for (std::size_t idx = 0; idx < even_view.size(); ++idx)
                              {
                                  auto* cell_ptr        = std::addressof(even_view[idx]);
                                  const std::size_t pos = static_cast<std::size_t>(cell_ptr - base_ptr);
                                  touched[pos]          = true;
                                  even_view[idx] += 1.0;
                              }
                          });

        const auto& ref_storage  = ref.array();
        auto&       base_storage = base.array();
        for (std::size_t idx = 0; idx < base_storage.size(); ++idx)
        {
            const double expected = ref_storage[idx] + (touched[idx] ? 1.0 : 0.0);
            EXPECT_NEAR(base_storage[idx], expected, 1e-12);
        }
    }

    TEST(field, vector_noalias_assignment)
    {
        static constexpr std::size_t dim = 2;
        Box<double, dim> box{xt::zeros<double>({dim}), xt::ones<double>({dim})};
        using Config    = UniformConfig<dim>;
        auto mesh       = UniformMesh<Config>(box, 2);
        using mesh_id_t = typename decltype(mesh)::mesh_id_t;

        auto vec    = make_vector_field<double, 2>("vec", mesh);
        auto scaled = make_vector_field<double, 2>("scaled", mesh);

        for_each_cell(mesh,
                      [&](auto cell)
                      {
                          auto coords = cell.center();
                          vec[cell][0] = coords[0];
                          if constexpr (dim > 1)
                          {
                              vec[cell][1] = coords[1];
                          }
                          else
                          {
                              vec[cell][1] = coords[0];
                          }
                      });

        for_each_interval(mesh[mesh_id_t::cells],
                          [&](std::size_t level, const auto& interval, const auto& index)
                          {
                              samurai::noalias(scaled(level, interval, index)) = 3.0 * vec(level, interval, index);
                          });

        for_each_cell(mesh,
                      [&](const auto& cell)
                      {
                          EXPECT_NEAR(scaled[cell][0], 3.0 * vec[cell][0], 1e-12);
                          EXPECT_NEAR(scaled[cell][1], 3.0 * vec[cell][1], 1e-12);
                      });
    }

    TEST(field, swap_preserves_ghost_flags)
    {
        Box<double, 1> box{{0}, {1}};
        using Config = UniformConfig<1>;
        auto mesh    = UniformMesh<Config>(box, 3);

        auto u = make_scalar_field<double>("u", mesh);
        auto v = make_scalar_field<double>("v", mesh);

        u.ghosts_updated() = true;
        v.ghosts_updated() = false;

        samurai::swap(u, v);

        EXPECT_FALSE(u.ghosts_updated());
        EXPECT_TRUE(v.ghosts_updated());
    }

    TEST(field, iterator)
    {
        using config = MRConfig<2>;
        CellList<2> cl;
        cl[1][{0}].add_interval({0, 2});
        cl[1][{0}].add_interval({4, 6});
        cl[2][{0}].add_interval({4, 8});

        auto mesh  = MRMesh<config>(cl, 1, 2);
        auto field = make_scalar_field<std::size_t>("u", mesh);

        std::size_t index = 0;
        for_each_cell(mesh,
                      [&](auto& cell)
                      {
                          field[cell] = index++;
                      });

        auto it = field.begin();
        EXPECT_TRUE(compare(*it, samurai::Array<std::size_t, 2, true>{0, 1}));
        it += 2;
        EXPECT_TRUE(compare(*it, samurai::Array<std::size_t, 4, true>{4, 5, 6, 7}));
        ++it;
        EXPECT_EQ(it, field.end());

        auto itr = field.rbegin();
        EXPECT_TRUE(compare(*itr, samurai::Array<std::size_t, 4, true>{4, 5, 6, 7}));
        itr += 2;
        EXPECT_TRUE(compare(*itr, samurai::Array<std::size_t, 2, true>{0, 1}));
        ++itr;
        EXPECT_EQ(itr, field.rend());
    }

    TEST(field, name)
    {
        Box<double, 1> box{{0}, {1}};
        using Config = UniformConfig<1>;
        auto mesh    = UniformMesh<Config>(box, 5);
        auto u       = make_scalar_field<double>("u", mesh);

        EXPECT_EQ(u.name(), "u");
        u.name() = "new_name";
        EXPECT_EQ(u.name(), "new_name");
    }
}
