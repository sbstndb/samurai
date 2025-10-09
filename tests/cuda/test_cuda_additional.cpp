#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include <samurai/storage/containers_config.hpp>
#include <samurai/algorithm.hpp>

namespace samurai
{
    TEST(cuda_backend, managed_array_self_broadcast)
    {
        thrust_container<double, 2, true, false> cont;
        cont.resize(3);

        // Initialize deterministic values per component / entry
        for (std::size_t comp = 0; comp < 2; ++comp)
        {
            for (std::size_t cell = 0; cell < 3; ++cell)
            {
                cont.data()(comp, cell) = static_cast<double>(comp + 1) * (cell + 1);
            }
        }

        // Multiply element-wise by itself
        cont.data() *= cont.data();
        for (std::size_t comp = 0; comp < 2; ++comp)
        {
            for (std::size_t cell = 0; cell < 3; ++cell)
            {
                auto expected = std::pow(static_cast<double>(comp + 1) * (cell + 1), 2.0);
                EXPECT_DOUBLE_EQ(cont.data()(comp, cell), expected);
            }
        }

        // Divide element-wise by itself → should yield 1.
        cont.data() /= cont.data();
        for (std::size_t comp = 0; comp < 2; ++comp)
        {
            for (std::size_t cell = 0; cell < 3; ++cell)
            {
                EXPECT_DOUBLE_EQ(cont.data()(comp, cell), 1.0);
            }
        }
    }

    TEST(cuda_backend, zeros_like_view)
    {
        thrust_container<int, 1, false, true> cont;
        cont.resize(5);
        for (std::size_t i = 0; i < 5; ++i)
        {
            cont.data()[i] = static_cast<int>(i + 1);
        }

        auto v      = view(cont, range_t<long long>{0, 5, 1});
        auto zeros  = zeros_like(v);
        ASSERT_EQ(zeros.size(), v.size());
        for (std::size_t i = 0; i < zeros.size(); ++i)
        {
            EXPECT_EQ(zeros[i], 0);
        }
    }
}
