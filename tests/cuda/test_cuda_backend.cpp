#include <gtest/gtest.h>

#include <array>
#include <vector>

#include <samurai/storage/containers_config.hpp>

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
}
