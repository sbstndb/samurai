#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <tuple>
#include <vector>

#include <samurai/algorithm/update.hpp>
#include <samurai/bc.hpp>
#include <samurai/storage/containers_config.hpp>
#include <samurai/algorithm.hpp>
#include <samurai/box.hpp>
#include <samurai/field.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/numeric/projection.hpp>
#include <samurai/subset/node.hpp>

namespace samurai
{
    TEST(cuda_backend, managed_array_basic)
    {
        thrust_container<double, 1, false, true> cont;
        cont.resize(8);
        ASSERT_EQ(cont.data().collapsed(), true);
        cont.data().fill(3.14);
        for (std::size_t i = 0; i < 8; ++i)
        {
            EXPECT_DOUBLE_EQ(cont.data()[i], 3.14);
        }
    }

    TEST(cuda_backend, view_and_mask)
    {
        thrust_container<int, 1, false, true> cont;
        cont.resize(10);
        for (std::size_t i = 0; i < 10; ++i) cont.data()[i] = static_cast<int>(i);

        auto v = view(cont, range_t<long long>{2, 10, 2}); // 2,4,6,8
        ASSERT_EQ(v.size(), 4u);

        auto mask = (v > 4.0);
        apply_on_masked(v, mask, [](int& x) { x *= 2; });

        EXPECT_EQ(v[0], 2);  // 2
        EXPECT_EQ(v[1], 4);  // 4
        EXPECT_EQ(v[2], 12); // 6*2
        EXPECT_EQ(v[3], 16); // 8*2
    }

    TEST(cuda_backend, soa_vector_views)
    {
        thrust_container<double, 3, true, false> cont;
        cont.resize(4);
        for (std::size_t comp = 0; comp < 3; ++comp)
        {
            for (std::size_t cell = 0; cell < 4; ++cell)
            {
                cont.data()(comp, cell) = 10.0 * static_cast<double>(comp) + static_cast<double>(cell);
            }
        }

        auto full = view(cont, range_t<long long>{0, 4, 1});
        EXPECT_EQ(full.items_count(), 3u);
        EXPECT_EQ(full.length_count(), 4u);

        auto second_cell = view(full, placeholders::all(), 2);
        ASSERT_EQ(second_cell.size(), 3u);
        EXPECT_DOUBLE_EQ(second_cell[0], 2.0);
        EXPECT_DOUBLE_EQ(second_cell[1], 12.0);
        EXPECT_DOUBLE_EQ(second_cell[2], 22.0);

        second_cell *= 2.0;
        EXPECT_DOUBLE_EQ(cont.data()(0, 2), 4.0);
        EXPECT_DOUBLE_EQ(cont.data()(1, 2), 24.0);
        EXPECT_DOUBLE_EQ(cont.data()(2, 2), 44.0);

        auto comp1 = view(cont, 1, range_t<long long>{1, 4, 1});
        ASSERT_EQ(comp1.size(), 3u);
        EXPECT_DOUBLE_EQ(comp1[0], 11.0);
        EXPECT_DOUBLE_EQ(comp1[1], 24.0);
        EXPECT_DOUBLE_EQ(comp1[2], 13.0);

        const auto& ccont = cont;
        auto sparse = view(ccont, range_t<long long>{0, 4, 2});
        EXPECT_EQ(sparse.length_count(), 2u);
        auto sparse_last = view(sparse, placeholders::all(), 1);
        EXPECT_DOUBLE_EQ(sparse_last[0], 4.0);
        EXPECT_DOUBLE_EQ(sparse_last[1], 24.0);
        EXPECT_DOUBLE_EQ(sparse_last[2], 44.0);

        auto dims = shape(full);
        EXPECT_EQ(dims[0], 3u);
        EXPECT_EQ(dims[1], 4u);
        EXPECT_EQ(shape(full, 0), 3u);
        EXPECT_EQ(shape(full, 1), 4u);

        auto row_view = view(full, 1);
        ASSERT_EQ(row_view.size(), 4u);
        EXPECT_DOUBLE_EQ(row_view[2], 24.0);
    }

    TEST(cuda_backend, aos_vector_views)
    {
        thrust_container<double, 3, false, false> cont;
        cont.resize(4);
        for (std::size_t cell = 0; cell < 4; ++cell)
        {
            for (std::size_t comp = 0; comp < 3; ++comp)
            {
                cont.data()(cell, comp) = 100.0 * static_cast<double>(cell) + static_cast<double>(comp);
            }
        }

        auto range_view = view(cont, range_t<long long>{1, 4, 1});
        EXPECT_EQ(range_view.items_count(), 3u);
        EXPECT_EQ(range_view.length_count(), 3u);

        auto mid_cell = view(range_view, placeholders::all(), 1);
        ASSERT_EQ(mid_cell.size(), 3u);
        EXPECT_DOUBLE_EQ(mid_cell[0], 200.0);
        EXPECT_DOUBLE_EQ(mid_cell[1], 201.0);
        EXPECT_DOUBLE_EQ(mid_cell[2], 202.0);

        mid_cell *= 3.0;
        EXPECT_DOUBLE_EQ(cont.data()(2, 0), 600.0);
        EXPECT_DOUBLE_EQ(cont.data()(2, 1), 603.0);
        EXPECT_DOUBLE_EQ(cont.data()(2, 2), 606.0);

        auto selected_items = view(cont, range_t<std::size_t>{0, 2, 1}, range_t<long long>{0, 4, 1});
        EXPECT_EQ(selected_items.items_count(), 2u);
        EXPECT_EQ(selected_items.length_count(), 4u);

        auto item_row = view(selected_items, 1);
        ASSERT_EQ(item_row.size(), 4u);
        EXPECT_DOUBLE_EQ(item_row[2], 603.0);

        const auto& ccont = cont;
        auto single_cell = view(ccont, 3);
        ASSERT_EQ(single_cell.size(), 3u);
        EXPECT_DOUBLE_EQ(single_cell[0], 300.0);
        EXPECT_DOUBLE_EQ(single_cell[1], 301.0);
        EXPECT_DOUBLE_EQ(single_cell[2], 302.0);
    }

    TEST(cuda_backend, range_step_write)
    {
        thrust_container<double, 1, false, true> cont;
        cont.resize(10);
        for (std::size_t i = 0; i < 10; ++i)
        {
            cont.data()[i] = static_cast<double>(i);
        }

        auto even_view = view(cont, range_t<long long>{0, 10, 2});
        ASSERT_EQ(even_view.size(), 5u);
        for (std::size_t i = 0; i < even_view.size(); ++i)
        {
            even_view[i] += 5.0;
        }

        for (std::size_t i = 0; i < 10; ++i)
        {
            const double expected = (i % 2 == 0) ? static_cast<double>(i) + 5.0 : static_cast<double>(i);
            EXPECT_DOUBLE_EQ(cont.data()[i], expected);
        }
    }

    TEST(cuda_backend, container_copy_is_deep)
    {
        thrust_container<double, 1, false, true> original;
        original.resize(6);
        for (std::size_t i = 0; i < 6; ++i)
        {
            original.data()[i] = static_cast<double>(i);
        }

        auto copied = original;

        thrust_container<double, 1, false, true> assigned;
        assigned = original;

        original.data()[0] = 42.0;
        original.data()[3] = -7.0;

        ASSERT_EQ(copied.data().size(), original.data().size());
        ASSERT_EQ(assigned.data().size(), original.data().size());

        EXPECT_DOUBLE_EQ(copied.data()[0], 0.0);
        EXPECT_DOUBLE_EQ(copied.data()[3], 3.0);
        EXPECT_DOUBLE_EQ(assigned.data()[0], 0.0);
        EXPECT_DOUBLE_EQ(assigned.data()[3], 3.0);
    }

    TEST(cuda_backend, math_helpers)
    {
        thrust_container<double, 1, false, true> cont;
        cont.resize(4);
        cont.data()[0] = -2.0;
        cont.data()[1] = -1.0;
        cont.data()[2] = 0.0;
        cont.data()[3] = 1.0;

        auto v = view(cont, range_t<long long>{0, 4, 1});
        EXPECT_DOUBLE_EQ(math::sum(v), -2.0);

        auto abs_v = math::abs(v);
        ASSERT_EQ(abs_v.size(), 4u);
        EXPECT_DOUBLE_EQ(abs_v[0], 2.0);
        EXPECT_DOUBLE_EQ(abs_v[1], 1.0);
        EXPECT_DOUBLE_EQ(abs_v[2], 0.0);
        EXPECT_DOUBLE_EQ(abs_v[3], 1.0);

        auto zeros_vec = zeros<double>(v.size());
        auto mins      = math::minimum(v, abs_v);
        auto maxs      = math::maximum(v, zeros_vec);

        EXPECT_DOUBLE_EQ(mins[0], -2.0);
        EXPECT_DOUBLE_EQ(mins[1], -1.0);
        EXPECT_DOUBLE_EQ(mins[2], 0.0);
        EXPECT_DOUBLE_EQ(mins[3], 1.0);

        EXPECT_DOUBLE_EQ(maxs[0], 0.0);
        EXPECT_DOUBLE_EQ(maxs[1], 0.0);
        EXPECT_DOUBLE_EQ(maxs[2], 0.0);
        EXPECT_DOUBLE_EQ(maxs[3], 1.0);
    }

    TEST(cuda_backend, noalias_assign)
    {
        thrust_container<double, 1, false, true> src1;
        thrust_container<double, 1, false, true> dst1;
        src1.resize(5);
        dst1.resize(5);
        for (std::size_t i = 0; i < 5; ++i)
        {
            src1.data()[i] = static_cast<double>(i);
            dst1.data()[i] = -1.0;
        }

        auto src_view = view(src1, range_t<long long>{0, 5, 1});
        auto dst_view = view(dst1, range_t<long long>{0, 5, 1});
        noalias(dst_view) = src_view;
        for (std::size_t i = 0; i < 5; ++i)
        {
            EXPECT_DOUBLE_EQ(dst1.data()[i], static_cast<double>(i));
        }

        thrust_container<double, 3, true, false> src2;
        thrust_container<double, 3, true, false> dst2;
        src2.resize(3);
        dst2.resize(3);
        for (std::size_t comp = 0; comp < 3; ++comp)
        {
            for (std::size_t cell = 0; cell < 3; ++cell)
            {
                src2.data()(comp, cell) = 10.0 * static_cast<double>(comp) + static_cast<double>(cell);
                dst2.data()(comp, cell) = 0.0;
            }
        }

        auto range_src = view(src2, range_t<long long>{0, 3, 1});
        auto range_dst = view(dst2, range_t<long long>{0, 3, 1});
        noalias(range_dst) = range_src;

        for (std::size_t comp = 0; comp < 3; ++comp)
        {
            for (std::size_t cell = 0; cell < 3; ++cell)
            {
                EXPECT_DOUBLE_EQ(dst2.data()(comp, cell), src2.data()(comp, cell));
            }
        }

        auto dst_col = view(range_dst, placeholders::all(), 1);
        auto src_col = view(range_src, placeholders::all(), 1);
        noalias(dst_col) += src_col;
        for (std::size_t comp = 0; comp < 3; ++comp)
        {
            EXPECT_DOUBLE_EQ(dst2.data()(comp, 1), 2.0 * src2.data()(comp, 1));
        }
    }

    TEST(cuda_backend, apply_on_masked_scalar)
    {
        thrust_container<double, 1, false, true> cont;
        cont.resize(6);
        for (std::size_t i = 0; i < 6; ++i)
        {
            cont.data()[i] = static_cast<double>(i);
        }

        auto values = view(cont, range_t<long long>{0, 6, 1});
        auto mask_mid = (values > 1.5) && (values < 3.5);

        apply_on_masked(values, mask_mid, [](double& v) { v += 100.0; });

        std::vector<double> expected{0.0, 1.0, 102.0, 103.0, 4.0, 5.0};
        ASSERT_EQ(values.size(), expected.size());
        for (std::size_t i = 0; i < values.size(); ++i)
        {
            EXPECT_DOUBLE_EQ(values[i], expected[i]);
        }

        auto mask_mid_updated = (values > 90.0) && (values < 110.0);
        std::vector<std::size_t> touched;
        apply_on_masked(mask_mid_updated, [&](std::size_t idx) { touched.push_back(idx); });
        ASSERT_EQ(touched.size(), 2u);
        EXPECT_EQ(touched[0], 2u);
        EXPECT_EQ(touched[1], 3u);

        auto mask_edges = !(values > 3.5);
        apply_on_masked(values, mask_edges, [](double& v) { v -= 1.0; });

        std::vector<double> expected_after{-1.0, 0.0, 102.0, 103.0, 4.0, 5.0};
        for (std::size_t i = 0; i < values.size(); ++i)
        {
            EXPECT_DOUBLE_EQ(values[i], expected_after[i]);
        }
    }

    TEST(cuda_backend, apply_on_masked_vector)
    {
        thrust_container<double, 2, true, false> cont;
        cont.resize(4);
        for (std::size_t comp = 0; comp < 2; ++comp)
        {
            for (std::size_t cell = 0; cell < 4; ++cell)
            {
                cont.data()(comp, cell) = 10.0 * static_cast<double>(comp) + static_cast<double>(cell);
            }
        }

        auto strided = view(cont, range_t<long long>{0, 4, 2});
        auto mask_mid = (strided > 5.0) && !(strided > 12.5);
        apply_on_masked(strided, mask_mid, [](double& v) { v += 200.0; });

        auto full = view(cont, range_t<long long>{0, 4, 1});
        auto mask_hi   = (full > 11.0);
        auto mask_low  = (full < 1.0);
        auto combined  = mask_hi || mask_low;

        std::vector<std::size_t> combined_indices;
        apply_on_masked(combined, [&](std::size_t idx) { combined_indices.push_back(idx); });
        ASSERT_EQ(combined_indices.size(), 4u);
        EXPECT_EQ(combined_indices[0], 0u);
        EXPECT_EQ(combined_indices[1], 1u);
        EXPECT_EQ(combined_indices[2], 5u);
        EXPECT_EQ(combined_indices[3], 7u);

        apply_on_masked(full, combined, [](double& v) { v += 1000.0; });

        std::array<double, 4> expected_comp0{1000.0, 1.0, 2.0, 3.0};
        std::array<double, 4> expected_comp1{1210.0, 11.0, 1212.0, 1013.0};

        for (std::size_t cell = 0; cell < 4; ++cell)
        {
            EXPECT_DOUBLE_EQ(cont.data()(0, cell), expected_comp0[cell]);
            EXPECT_DOUBLE_EQ(cont.data()(1, cell), expected_comp1[cell]);
        }
    }

    TEST(cuda_backend, apply_on_masked_broadcast_scalar_mask)
    {
        thrust_container<double, 1, false, true> cont;
        cont.resize(8);
        const auto total = cont.data().size();
        for (std::size_t i = 0; i < total; ++i)
        {
            cont.data()[i] = static_cast<double>(i);
        }

        auto full  = view(cont, range_t<long long>{0, 8, 1});
        auto first = view(cont, range_t<long long>{0, 1, 1});

        auto mask_full   = (full > 2.5);
        auto mask_single = (first > -1.0);

        EXPECT_EQ(mask_single.size(), 1u);

        auto combined = mask_full && mask_single;
        EXPECT_EQ(combined.size(), full.size());

        apply_on_masked(full, combined, [](double& v) { v += 10.0; });

        std::array<double, 8> expected{0.0, 1.0, 2.0, 13.0, 14.0, 15.0, 16.0, 17.0};
        for (std::size_t idx = 0; idx < full.size(); ++idx)
        {
            EXPECT_DOUBLE_EQ(full[idx], expected[idx]);
        }
    }

    TEST(cuda_backend, temporary_expression_mask_lifetime)
    {
        thrust_container<double, 1, false, true> cont;
        cont.resize(5);
        for (std::size_t i = 0; i < cont.data().size(); ++i)
        {
            cont.data()[i] = static_cast<double>(i);
        }

        auto vals = view(cont, range_t<long long>{0, 5, 1});
        apply_on_masked(vals, (math::abs(vals) < 2.5), [](double& v) { v += 50.0; });

        std::array<double, 5> expected{50.0, 51.0, 52.0, 3.0, 4.0};
        for (std::size_t i = 0; i < expected.size(); ++i)
        {
            EXPECT_DOUBLE_EQ(vals[i], expected[i]);
        }
    }

    TEST(cuda_backend, sum_axis_over_mask_expression)
    {
        thrust_container<double, 2, true, false> cont;
        cont.resize(3);
        // comp0: [0.1, 5.0, 0.4], comp1: [0.05, 0.6, 0.2]
        cont.data()(0, 0) = 0.1;
        cont.data()(0, 1) = 5.0;
        cont.data()(0, 2) = 0.4;
        cont.data()(1, 0) = 0.05;
        cont.data()(1, 1) = 0.6;
        cont.data()(1, 2) = 0.2;

        auto view2d = view(cont, range_t<long long>{0, 3, 1});
        auto mask_expr = (math::abs(view2d) < 0.7);

        auto sum_axis0 = math::sum<0>(mask_expr);
        ASSERT_EQ(sum_axis0.size(), 3u);
        EXPECT_EQ(sum_axis0[0], 2u);
        EXPECT_EQ(sum_axis0[1], 1u);
        EXPECT_EQ(sum_axis0[2], 2u);

        auto sum_axis1 = math::sum<1>(mask_expr);
        ASSERT_EQ(sum_axis1.size(), 2u);
        EXPECT_EQ(sum_axis1[0], 2u);
        EXPECT_EQ(sum_axis1[1], 3u);
    }

    TEST(cuda_backend, all_true_mask_chain)
    {
        thrust_container<double, 2, true, false> cont;
        cont.resize(2);
        cont.data()(0, 0) = 0.1;
        cont.data()(0, 1) = 0.9;
        cont.data()(1, 0) = 0.15;
        cont.data()(1, 1) = 1.5;

        auto v = view(cont, range_t<long long>{0, 2, 1});
        auto base_abs = math::abs(v);
        auto cond = (base_abs < 0.5) && !(base_abs > 1.0) && (v + 0.1 < 1.2) && (math::abs(v - 0.05) < 0.3);

        auto mask = math::all_true<0, 2>(cond);
        ASSERT_EQ(mask.size(), 2u);
        EXPECT_TRUE(mask[0]);
        EXPECT_FALSE(mask[1]);
    }

    TEST(cuda_backend, update_ghost_mr_scalar)
    {
        constexpr std::size_t dim = 2;
        using Config               = MRConfig<dim>;
        using Mesh                 = MRMesh<Config>;
        using mesh_id_t            = typename Mesh::mesh_id_t;

        Box<double, dim> box({0.0, 0.0}, {1.0, 1.0});
        Mesh mesh{box, 0, 2};

        auto field = make_scalar_field<double>("field", mesh);
        field.resize();

        make_bc<Dirichlet<1>>(field, 0.0);

        for_each_cell(mesh[mesh_id_t::cells],
                      [&](auto& cell)
                      {
                          field[cell] = static_cast<double>(cell.level + cell.indices[0] + 10 * cell.indices[1]);
                      });

        update_ghost_mr(field);

        // Probe a cell at level 1 and its parent at level 0 to ensure values are finite.
        bool checked = false;
        for_each_cell(mesh[mesh_id_t::cells],
                      [&](auto& cell)
                      {
                          auto val = field[cell];
                          EXPECT_TRUE(std::isfinite(val));
                          checked = true;
                      });
        EXPECT_TRUE(checked);
    }

    TEST(cuda_backend, projection_scalar_field)
    {
        constexpr std::size_t dim = 2;
        using Config               = MRConfig<dim>;
        using Mesh                 = MRMesh<Config>;
        using mesh_id_t            = typename Mesh::mesh_id_t;

        Box<double, dim> box({0.0, 0.0}, {1.0, 1.0});
        Mesh mesh{box, 0, 1};

        auto field = make_scalar_field<double>("field", mesh);
        field.resize();

        auto value_for = [](long long ix, long long iy)
        {
            return static_cast<double>(ix + 10 * iy);
        };

        for_each_cell(mesh[mesh_id_t::cells],
                      [&](const auto& cell)
                      {
                          if (cell.level == 1)
                          {
                              field[cell] = value_for(cell.indices[0], cell.indices[1]);
                          }
                          else
                          {
                              field[cell] = -1.0;
                          }
                      });

        auto subset = intersection(mesh[mesh_id_t::reference][1], mesh[mesh_id_t::proj_cells][0]).on(0);

        std::vector<std::tuple<long long, long long, long long>> intervals;
        bool has_negative = false;
        subset(
            [&](const auto& interval, const auto& index)
            {
                intervals.emplace_back(interval.start, interval.end, index[0]);
                if (interval.start < 0)
                {
                    has_negative = true;
                }
            });
        EXPECT_FALSE(intervals.empty());
        EXPECT_FALSE(has_negative) << "projection subset includes negative start interval";

        subset.apply_op(projection(field));

        for_each_cell(mesh[mesh_id_t::cells],
                      [&](const auto& cell)
                      {
                          if (cell.level == 0)
                          {
                              const long long ix = cell.indices[0];
                              const long long iy = cell.indices[1];
                              const double expected = 0.25 * (value_for(2 * ix, 2 * iy) + value_for(2 * ix, 2 * iy + 1)
                                                              + value_for(2 * ix + 1, 2 * iy) + value_for(2 * ix + 1, 2 * iy + 1));
                              EXPECT_DOUBLE_EQ(field[cell], expected);
                          }
                      });
    }

    TEST(cuda_backend, variadic_projection_two_fields)
    {
        constexpr std::size_t dim = 2;
        using Config               = MRConfig<dim>;
        using Mesh                 = MRMesh<Config>;
        using mesh_id_t            = typename Mesh::mesh_id_t;

        Box<double, dim> box({0.0, 0.0}, {1.0, 1.0});
        Mesh mesh{box, 0, 1};

        auto field_a = make_scalar_field<double>("field_a", mesh);
        auto field_b = make_scalar_field<double>("field_b", mesh);
        field_a.resize();
        field_b.resize();

        auto value_a = [](long long ix, long long iy)
        {
            return static_cast<double>(ix + 100 * iy);
        };
        auto value_b = [](long long ix, long long iy)
        {
            return static_cast<double>(2 * ix - 3 * iy);
        };

        for_each_cell(mesh[mesh_id_t::cells],
                      [&](const auto& cell)
                      {
                          if (cell.level == 1)
                          {
                              field_a[cell] = value_a(cell.indices[0], cell.indices[1]);
                              field_b[cell] = value_b(cell.indices[0], cell.indices[1]);
                          }
                          else
                          {
                              field_a[cell] = 0.0;
                              field_b[cell] = 0.0;
                          }
                      });

        auto subset = intersection(mesh[mesh_id_t::reference][1], mesh[mesh_id_t::proj_cells][0]).on(0);
        std::vector<std::tuple<long long, long long, long long>> intervals;
        bool has_negative = false;
        subset(
            [&](const auto& interval, const auto& index)
            {
                intervals.emplace_back(interval.start, interval.end, index[0]);
                if (interval.start < 0)
                {
                    has_negative = true;
                }
            });
        EXPECT_FALSE(intervals.empty());
        EXPECT_FALSE(has_negative) << "variadic projection subset includes negative start interval";

        subset.apply_op(variadic_projection(field_a, field_b));

        for_each_cell(mesh[mesh_id_t::cells],
                      [&](const auto& cell)
                      {
                          if (cell.level == 0)
                          {
                              const long long ix = cell.indices[0];
                              const long long iy = cell.indices[1];
                              const double expected_a = 0.25 * (value_a(2 * ix, 2 * iy) + value_a(2 * ix, 2 * iy + 1)
                                                                + value_a(2 * ix + 1, 2 * iy) + value_a(2 * ix + 1, 2 * iy + 1));
                              const double expected_b = 0.25 * (value_b(2 * ix, 2 * iy) + value_b(2 * ix, 2 * iy + 1)
                                                                + value_b(2 * ix + 1, 2 * iy) + value_b(2 * ix + 1, 2 * iy + 1));
                              EXPECT_DOUBLE_EQ(field_a[cell], expected_a);
                              EXPECT_DOUBLE_EQ(field_b[cell], expected_b);
                          }
                      });
    }
}
