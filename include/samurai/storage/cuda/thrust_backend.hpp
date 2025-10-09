// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <concepts>
#include <utility>
#include <vector>
#include <array>
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <cassert>
#include <limits>

#include <cuda_runtime.h>

#include "../utils.hpp"
#include "../xtensor/xtensor_static.hpp"

namespace samurai
{
    namespace placeholders
    {
        struct all_t
        {
        };

        inline constexpr all_t all_v{};
        inline constexpr all_t all()
        {
            return all_v;
        }

        // underscore placeholder not used for CUDA path for now
        struct underscore_t
        {
        };
        inline constexpr underscore_t _{};
    }

    namespace detail
    {
        template <class T>
        struct dependent_false;

        template <class T>
        concept element_indexable = requires(const T& v, std::size_t idx)
        {
            { v.size() } -> std::convertible_to<std::size_t>;
            v[idx];
        };

        template <class T>
        concept mask_indexable = element_indexable<T> && requires(const T& v, std::size_t idx)
        {
            { static_cast<bool>(v[idx]) } -> std::convertible_to<bool>;
        };

        template <class T>
        concept scalar_like = std::is_arithmetic_v<std::remove_cvref_t<T>>;

        template <class T>
        auto decay_copy(T&& v)
        {
            return std::decay_t<T>(std::forward<T>(v));
        }

        template <class DST, class SRC>
        void assign_elements(DST& dst, const SRC& src);

        template <class DST, class SRC>
        void add_elements(DST& dst, const SRC& src);

        template <class DST, class SRC>
        void sub_elements(DST& dst, const SRC& src);
    }

    // Small RAII for cudaMallocManaged
    template <class T>
    struct managed_ptr
    {
        T* ptr = nullptr;

        managed_ptr() = default;
        explicit managed_ptr(std::size_t n)
        {
            allocate(n);
        }
        ~managed_ptr()
        {
            reset();
        }

        void allocate(std::size_t n)
        {
            reset();
            if (n == 0)
            {
                return;
            }
            void* p = nullptr;
            auto  st = ::cudaMallocManaged(&p, n * sizeof(T), cudaMemAttachGlobal);
            if (st != cudaSuccess)
            {
                throw std::runtime_error("cudaMallocManaged failed");
            }
            ptr = static_cast<T*>(p);
        }

        void reset()
        {
            if (ptr)
            {
                ::cudaFree(ptr);
                ptr = nullptr;
            }
        }

        T* get() const { return ptr; }
        T& operator[](std::size_t i) const { return ptr[i]; }
    };

    // Managed 2D array wrapper with optional 1D collapsed mode
    template <class T>
    struct managed_array
    {
        using value_type = T;
        using size_type  = std::size_t;

        managed_ptr<T> m_buf{};
        size_type      m_rows{0};
        size_type      m_cols{0};
        bool           m_collapsed{true};

        managed_array() = default;

        void resize_1d(size_type n)
        {
            m_buf.allocate(n);
            m_rows     = 1;
            m_cols     = n;
            m_collapsed = true;
        }

        void resize_2d(size_type rows, size_type cols)
        {
            m_buf.allocate(rows * cols);
            m_rows      = rows;
            m_cols      = cols;
            m_collapsed = false;
        }

        size_type rows() const { return m_rows; }
        size_type cols() const { return m_cols; }
        size_type size() const { return m_rows * m_cols; }
        bool      collapsed() const { return m_collapsed; }

        // Scalar collapsed access
        T& operator[](size_type i) { return m_buf[i]; }
        const T& operator[](size_type i) const { return m_buf[i]; }

        // 2D access
        T& operator()(size_type r, size_type c) { return m_buf[r * m_cols + c]; }
        const T& operator()(size_type r, size_type c) const { return m_buf[r * m_cols + c]; }

        // Bulk ops
        void fill(const T& v)
        {
            for (size_type i = 0; i < size(); ++i)
            {
                m_buf[i] = v;
            }
        }

        T* data() { return m_buf.get(); }
        const T* data() const { return m_buf.get(); }
    };

    // Lightweight 1D view (contiguous with step)
    template <class T>
    struct thrust_view_1d
    {
        using value_type   = T;
        using scalar_type  = std::remove_const_t<T>;
        using pointer_type = value_type*;

        pointer_type base{nullptr};
        std::size_t  start{0};
        std::size_t  end{0};
        std::size_t  step{1};
        std::size_t  stride{1}; // element stride in underlying storage

        std::size_t size() const
        {
            if (end <= start)
            {
                return 0;
            }
            return (end - start + (step - 1)) / step;
        }

        value_type& operator[](std::size_t i)
            requires(!std::is_const_v<value_type>)
        {
            return base[(start + i * step) * stride];
        }

        const value_type& operator[](std::size_t i) const
        {
            return base[(start + i * step) * stride];
        }

        template <class V>
        thrust_view_1d& operator=(const V& rhs)
            requires(!std::is_const_v<value_type> && detail::element_indexable<V>)
        {
            auto n = size();
            for (std::size_t i = 0; i < n; ++i)
            {
                (*this)[i] = rhs[i];
            }
            return *this;
        }

        thrust_view_1d& operator=(scalar_type value)
            requires(!std::is_const_v<value_type>)
        {
            auto n = size();
            for (std::size_t i = 0; i < n; ++i)
            {
                (*this)[i] = value;
            }
            return *this;
        }

        struct scaled
        {
            const thrust_view_1d& v;
            scalar_type           s;
            scalar_type operator[](std::size_t i) const
            {
                return static_cast<scalar_type>(v[i]) / s;
            }
        };

        scaled operator/(scalar_type s) const { return scaled{*this, s}; }

        template <class V>
        thrust_view_1d& operator+=(const V& rhs)
            requires(!std::is_const_v<value_type> && detail::element_indexable<V>)
        {
            auto n = size();
            for (std::size_t i = 0; i < n; ++i)
            {
                (*this)[i] += rhs[i];
            }
            return *this;
        }

        template <class V>
        thrust_view_1d& operator-=(const V& rhs)
            requires(!std::is_const_v<value_type> && detail::element_indexable<V>)
        {
            auto n = size();
            for (std::size_t i = 0; i < n; ++i)
            {
                (*this)[i] -= rhs[i];
            }
            return *this;
        }

        thrust_view_1d& operator*=(scalar_type s)
            requires(!std::is_const_v<value_type>)
        {
            auto n = size();
            for (std::size_t i = 0; i < n; ++i)
            {
                (*this)[i] *= s;
            }
            return *this;
        }

        thrust_view_1d& operator/=(scalar_type s)
            requires(!std::is_const_v<value_type>)
        {
            auto n = size();
            for (std::size_t i = 0; i < n; ++i)
            {
                (*this)[i] /= s;
            }
            return *this;
        }
    };

    // 2D view: items x range length (row-major in memory for items-first layouts)
    template <class T>
    struct thrust_view_2d
    {
        using value_type   = T;
        using scalar_type  = std::remove_const_t<std::decay_t<T>>;
        using pointer_type = value_type*;

        pointer_type base{nullptr};
        std::size_t  items{0};   // number of components/items
        std::size_t  length{0};  // number of positions along the range dimension
        std::size_t  item_stride{1};
        std::size_t  length_stride{1};

        std::size_t items_count() const { return items; }
        std::size_t length_count() const { return length; }
        std::size_t size() const { return items * length; }

        value_type& operator[](std::size_t idx)
            requires(!std::is_const_v<value_type>)
        {
            if (items == 0 || length == 0)
            {
                throw std::out_of_range("thrust_view_2d::operator[] on empty view");
            }
            const std::size_t item_idx = idx % items;
            const std::size_t pos_idx  = idx / items;
            return base[item_idx * item_stride + pos_idx * length_stride];
        }

        const value_type& operator[](std::size_t idx) const
        {
            if (items == 0 || length == 0)
            {
                throw std::out_of_range("thrust_view_2d::operator[] on empty view");
            }
            const std::size_t item_idx = idx % items;
            const std::size_t pos_idx  = idx / items;
            return base[item_idx * item_stride + pos_idx * length_stride];
        }

        thrust_view_1d<value_type> column(std::size_t j) const
        {
            thrust_view_1d<value_type> v;
            v.base   = base + j * length_stride;
            v.start  = 0;
            v.end    = items;
            v.step   = 1;
            v.stride = item_stride;
            return v;
        }

        thrust_view_1d<value_type> row(std::size_t item) const
        {
            thrust_view_1d<value_type> v;
            v.base   = base + item * item_stride;
            v.start  = 0;
            v.end    = length;
            v.step   = 1;
            v.stride = length_stride;
            return v;
        }

        value_type& operator()(std::size_t item_idx, std::size_t pos_idx)
            requires(!std::is_const_v<value_type>)
        {
            return base[item_idx * item_stride + pos_idx * length_stride];
        }

        const value_type& operator()(std::size_t item_idx, std::size_t pos_idx) const
        {
            return base[item_idx * item_stride + pos_idx * length_stride];
        }

        template <class V>
        thrust_view_2d& operator=(const V& rhs)
            requires(!std::is_const_v<value_type>)
        {
            detail::assign_elements(*this, rhs);
            return *this;
        }

        template <class V>
        thrust_view_2d& operator+=(const V& rhs)
            requires(!std::is_const_v<value_type>)
        {
            detail::add_elements(*this, rhs);
            return *this;
        }

        template <class V>
        thrust_view_2d& operator-=(const V& rhs)
            requires(!std::is_const_v<value_type>)
        {
            detail::sub_elements(*this, rhs);
            return *this;
        }

        thrust_view_2d& operator*=(scalar_type s)
            requires(!std::is_const_v<value_type>)
        {
            for (std::size_t j = 0; j < length; ++j)
            {
                auto col = column(j);
                col *= s;
            }
            return *this;
        }

        thrust_view_2d& operator/=(scalar_type s)
            requires(!std::is_const_v<value_type>)
        {
            for (std::size_t j = 0; j < length; ++j)
            {
                auto col = column(j);
                col /= s;
            }
            return *this;
        }
    };

    namespace detail
    {
        template <class value_t, std::size_t size, bool SOA, bool can_collapse, layout_type L>
        struct thrust_shape_helper
        {
            static void resize(managed_array<value_t>& a, std::size_t dynamic_size)
            {
                if constexpr ((size == 1) && can_collapse)
                {
                    a.resize_1d(dynamic_size);
                }
                else
                {
                    if constexpr (static_size_first_v<size, SOA, can_collapse, L>)
                    {
                        a.resize_2d(size, dynamic_size);
                    }
                    else
                    {
                        a.resize_2d(dynamic_size, size);
                    }
                }
            }
        };

        inline std::size_t compute_length(long long start, long long end, long long step)
        {
            if (step <= 0)
            {
                throw std::invalid_argument("range.step must be positive");
            }
            if (end <= start)
            {
                return 0;
            }
            return static_cast<std::size_t>((end - start + step - 1) / step);
        }

        template <class T>
        struct dependent_false : std::false_type
        {
        };

        template <class T>
        struct is_thrust_view : std::false_type
        {
        };

        template <class T>
        struct is_thrust_view<thrust_view_1d<T>> : std::true_type
        {
        };

        template <class T>
        struct is_thrust_view<thrust_view_2d<T>> : std::true_type
        {
        };

        template <class T>
        struct is_std_vector : std::false_type
        {
        };

        template <class T, class Alloc>
        struct is_std_vector<std::vector<T, Alloc>> : std::true_type
        {
        };

        template <class Op, class Expr>
        struct thrust_expr_unary;

        template <class Op, class L, class R>
        struct thrust_expr_binary;

        template <class Scalar, class Expr, class Op, bool left>
        struct thrust_expr_scalar;

        template <class Expr>
        struct thrust_expr_slice;

        template <class T>
        struct is_thrust_expr : std::false_type
        {
        };

        template <class Op, class Expr>
        struct is_thrust_expr<thrust_expr_unary<Op, Expr>> : std::true_type
        {
        };

        template <class Op, class L, class R>
        struct is_thrust_expr<thrust_expr_binary<Op, L, R>> : std::true_type
        {
        };

        template <class Scalar, class Expr, class Op, bool left>
        struct is_thrust_expr<thrust_expr_scalar<Scalar, Expr, Op, left>> : std::true_type
        {
        };

        template <class Expr>
        struct is_thrust_expr<thrust_expr_slice<Expr>> : std::true_type
        {
        };

        template <class T>
        concept thrust_expression = is_thrust_expr<std::decay_t<T>>::value;

        template <class T>
        concept thrust_sequence = is_thrust_view<std::decay_t<T>>::value || thrust_expression<T> || is_std_vector<std::decay_t<T>>::value;

        template <class V>
        using element_value_t = std::remove_cvref_t<decltype(std::declval<const V&>()[std::declval<std::size_t>()])>;

        template <class Seq>
        std::size_t sequence_size(const Seq& seq)
        {
            if constexpr (requires { seq.size(); })
            {
                return static_cast<std::size_t>(seq.size());
            }
            else if constexpr (requires { seq.length_count(); seq.items_count(); })
            {
                return seq.length_count() * seq.items_count();
            }
            else
            {
                static_assert(dependent_false<Seq>::value, "Sequence type without size()");
            }
        }

        template <class Expr>
        auto make_expr_slice(Expr expr, const range_t<long long>& range);

        struct plus_op
        {
            template <class A, class B>
            static auto apply(A&& a, B&& b)
            {
                return std::forward<A>(a) + std::forward<B>(b);
            }
        };

        struct minus_op
        {
            template <class A, class B>
            static auto apply(A&& a, B&& b)
            {
                return std::forward<A>(a) - std::forward<B>(b);
            }
        };

        struct mult_op
        {
            template <class A, class B>
            static auto apply(A&& a, B&& b)
            {
                return std::forward<A>(a) * std::forward<B>(b);
            }
        };

        struct div_op
        {
            template <class A, class B>
            static auto apply(A&& a, B&& b)
            {
                return std::forward<A>(a) / std::forward<B>(b);
            }
        };

        struct neg_op
        {
            template <class A>
            static auto apply(A&& a)
            {
                return -std::forward<A>(a);
            }
        };

        struct abs_op
        {
            template <class A>
            static auto apply(A&& a)
            {
                using std::abs;
                return abs(std::forward<A>(a));
            }
        };

        struct min_op
        {
            template <class A, class B>
            static auto apply(A&& a, B&& b)
            {
                using std::min;
                return min(std::forward<A>(a), std::forward<B>(b));
            }
        };

        struct max_op
        {
            template <class A, class B>
            static auto apply(A&& a, B&& b)
            {
                using std::max;
                return max(std::forward<A>(a), std::forward<B>(b));
            }
        };

        template <class View>
        struct noalias_proxy
        {
            View view;

            template <class RHS>
            noalias_proxy& operator=(const RHS& rhs)
            {
                assign_elements(view, rhs);
                return *this;
            }

            template <class RHS>
            noalias_proxy& operator+=(const RHS& rhs)
            {
                add_elements(view, rhs);
                return *this;
            }

            template <class RHS>
            noalias_proxy& operator-=(const RHS& rhs)
            {
                sub_elements(view, rhs);
                return *this;
            }

            template <class Scalar>
            noalias_proxy& operator*=(Scalar&& s)
            {
                view *= std::forward<Scalar>(s);
                return *this;
            }

            template <class Scalar>
            noalias_proxy& operator/=(Scalar&& s)
            {
                view /= std::forward<Scalar>(s);
                return *this;
            }
        };

        template <class value_t, std::size_t size, bool SOA, bool can_collapse, class Container>
        auto make_range_view(Container& container, const range_t<long long>& range)
        {
            using container_type = std::remove_reference_t<Container>;
            constexpr bool is_const = std::is_const_v<container_type>;
            using val_type          = std::conditional_t<is_const, const value_t, value_t>;

            auto&& data                = container.data();
            auto   base_ptr            = data.data();
            const auto start_ll        = range.start;
            const auto step_ll         = range.step;
            const std::size_t length   = compute_length(start_ll, range.end, step_ll);
            const std::size_t step_sz  = static_cast<std::size_t>(step_ll);
            const std::size_t start_sz = static_cast<std::size_t>(start_ll);

            if constexpr ((size == 1) && can_collapse)
            {
                thrust_view_1d<val_type> v{};
                v.base   = base_ptr + start_sz;
                v.start  = 0;
                v.end    = length;
                v.step   = 1;
                v.stride = step_sz;
                return v;
            }
            else
            {
                thrust_view_2d<val_type> v{};
                v.length = length;
                if constexpr (static_size_first_v<size, SOA, can_collapse, SAMURAI_DEFAULT_LAYOUT>)
                {
                    const std::size_t cols = data.cols();
                    v.base          = base_ptr + start_sz;
                    v.items         = size;
                    v.item_stride   = cols;
                    v.length_stride = step_sz;
                }
                else
                {
                    const std::size_t cols = data.cols();
                    v.base          = base_ptr + start_sz * cols;
                    v.items         = cols;
                    v.item_stride   = 1;
                    v.length_stride = step_sz * cols;
                }
                return v;
            }
        }

        template <class value_t, std::size_t size, bool SOA, bool can_collapse, class Container>
        auto make_item_range_view(Container& container,
                                  const range_t<std::size_t>& range_item,
                                  const range_t<long long>&   range)
        {
            using container_type = std::remove_reference_t<Container>;
            constexpr bool is_const = std::is_const_v<container_type>;
            using val_type          = std::conditional_t<is_const, const value_t, value_t>;

            auto&& data                  = container.data();
            auto   base_ptr              = data.data();
            const std::size_t item_start = range_item.start;
            const std::size_t item_step  = range_item.step;
            const std::size_t items_len  = compute_length(static_cast<long long>(range_item.start),
                                                          static_cast<long long>(range_item.end),
                                                          static_cast<long long>(range_item.step));
            const std::size_t length     = compute_length(range.start, range.end, range.step);
            const std::size_t range_step = static_cast<std::size_t>(range.step);
            const std::size_t range_start = static_cast<std::size_t>(range.start);

            thrust_view_2d<val_type> v{};
            v.items  = items_len;
            v.length = length;

            if constexpr (static_size_first_v<size, SOA, can_collapse, SAMURAI_DEFAULT_LAYOUT>)
            {
                const std::size_t cols = data.cols();
                v.base          = base_ptr + item_start * cols + range_start;
                v.item_stride   = cols * item_step;
                v.length_stride = range_step;
            }
            else
            {
                const std::size_t cols = data.cols();
                v.base          = base_ptr + range_start * cols + item_start;
                v.item_stride   = item_step;
                v.length_stride = range_step * cols;
            }
            return v;
        }

        template <class value_t, std::size_t size, bool SOA, bool can_collapse, class Container>
        auto make_index_view(Container& container, std::size_t index)
        {
            using container_type = std::remove_reference_t<Container>;
            constexpr bool is_const = std::is_const_v<container_type>;
            using val_type          = std::conditional_t<is_const, const value_t, value_t>;

            auto&& data    = container.data();
            auto   base_ptr = data.data();

            thrust_view_1d<val_type> v{};
            if constexpr ((size == 1) && can_collapse)
            {
                v.base   = base_ptr + index;
                v.start  = 0;
                v.end    = 1;
                v.step   = 1;
                v.stride = 1;
            }
            else if constexpr (static_size_first_v<size, SOA, can_collapse, SAMURAI_DEFAULT_LAYOUT>)
            {
                const std::size_t cols = data.cols();
                v.base   = base_ptr + index;
                v.start  = 0;
                v.end    = size;
                v.step   = 1;
                v.stride = cols;
            }
            else
            {
                const std::size_t cols = data.cols();
                v.base   = base_ptr + index * cols;
                v.start  = 0;
                v.end    = size;
                v.step   = 1;
                v.stride = 1;
            }
            return v;
        }

        template <class value_t, std::size_t size, bool SOA, bool can_collapse, class Container>
        auto make_item_view(Container& container, std::size_t item, const range_t<long long>& range)
        {
            using container_type = std::remove_reference_t<Container>;
            constexpr bool is_const = std::is_const_v<container_type>;
            using val_type          = std::conditional_t<is_const, const value_t, value_t>;

            auto&& data           = container.data();
            auto   base_ptr        = data.data();
            const std::size_t len = compute_length(range.start, range.end, range.step);
            const std::size_t step_sz  = static_cast<std::size_t>(range.step);
            const std::size_t start_sz = static_cast<std::size_t>(range.start);

            thrust_view_1d<val_type> v{};
            if constexpr (static_size_first_v<size, SOA, can_collapse, SAMURAI_DEFAULT_LAYOUT>)
            {
                const std::size_t cols = data.cols();
                v.base   = base_ptr + item * cols + start_sz;
                v.start  = 0;
                v.end    = len;
                v.step   = 1;
                v.stride = step_sz;
            }
            else
            {
                const std::size_t cols = data.cols();
                v.base   = base_ptr + start_sz * cols + item;
                v.start  = 0;
                v.end    = len;
                v.step   = 1;
                v.stride = step_sz * cols;
            }
            return v;
        }

        template <class DST, class SRC>
        void assign_elements(DST& dst, const SRC& src)
        {
            if constexpr (requires { dst.size(); src.size(); } && !requires { dst.column(std::size_t{}); })
            {
                assert(dst.size() == src.size());
                for (std::size_t i = 0; i < dst.size(); ++i)
                {
                    dst[i] = src[i];
                }
            }
            else if constexpr (requires { dst.column(std::size_t{}); src.column(std::size_t{}); dst.length_count(); src.length_count(); })
            {
                assert(dst.length_count() == src.length_count());
                for (std::size_t j = 0; j < dst.length_count(); ++j)
                {
                    auto dst_col = dst.column(j);
                    auto src_col = src.column(j);
                    assign_elements(dst_col, src_col);
                }
            }
            else
            {
                static_assert(dependent_false<DST>::value, "Unsupported assignment between views");
            }
        }

        template <class DST, class SRC>
        void add_elements(DST& dst, const SRC& src)
        {
            if constexpr (requires { dst.size(); src.size(); } && !requires { dst.column(std::size_t{}); })
            {
                assert(dst.size() == src.size());
                for (std::size_t i = 0; i < dst.size(); ++i)
                {
                    dst[i] += src[i];
                }
            }
            else if constexpr (requires { dst.column(std::size_t{}); src.column(std::size_t{}); dst.length_count(); src.length_count(); })
            {
                assert(dst.length_count() == src.length_count());
                for (std::size_t j = 0; j < dst.length_count(); ++j)
                {
                    auto dst_col = dst.column(j);
                    auto src_col = src.column(j);
                    add_elements(dst_col, src_col);
                }
            }
            else
            {
                static_assert(dependent_false<DST>::value, "Unsupported += between views");
            }
        }

        template <class DST, class SRC>
        void sub_elements(DST& dst, const SRC& src)
        {
            if constexpr (requires { dst.size(); src.size(); } && !requires { dst.column(std::size_t{}); })
            {
                assert(dst.size() == src.size());
                for (std::size_t i = 0; i < dst.size(); ++i)
                {
                    dst[i] -= src[i];
                }
            }
            else if constexpr (requires { dst.column(std::size_t{}); src.column(std::size_t{}); dst.length_count(); src.length_count(); })
            {
                assert(dst.length_count() == src.length_count());
                for (std::size_t j = 0; j < dst.length_count(); ++j)
                {
                    auto dst_col = dst.column(j);
                    auto src_col = src.column(j);
                    sub_elements(dst_col, src_col);
                }
            }
            else
            {
                static_assert(dependent_false<DST>::value, "Unsupported -= between views");
            }
        }
    }

    template <class value_t, std::size_t size, bool SOA = false, bool can_collapse = true>
    struct thrust_container
    {
        static constexpr layout_type static_layout = SAMURAI_DEFAULT_LAYOUT;
        using container_t                          = managed_array<value_t>;
        using size_type                            = std::size_t;
        using default_view_type                    = thrust_view_1d<value_t>; // iterator reference type

        thrust_container() = default;

        explicit thrust_container(std::size_t dynamic_size)
        {
            resize(dynamic_size);
        }

        const container_t& data() const { return m_data; }
        container_t& data() { return m_data; }

        void resize(std::size_t dynamic_size)
        {
            detail::thrust_shape_helper<value_t, size, SOA, can_collapse, static_layout>::resize(m_data, dynamic_size);
        }

      private:
        container_t m_data;
    };

    // Views from containers --------------------------------------------------
    template <class value_t, std::size_t size, bool SOA, bool can_collapse>
    auto view(thrust_container<value_t, size, SOA, can_collapse>& container, const range_t<long long>& range)
    {
        return detail::make_range_view<value_t, size, SOA, can_collapse>(container, range);
    }

    template <class value_t, std::size_t size, bool SOA, bool can_collapse>
    auto view(const thrust_container<value_t, size, SOA, can_collapse>& container, const range_t<long long>& range)
    {
        return detail::make_range_view<value_t, size, SOA, can_collapse>(container, range);
    }

    template <class value_t, std::size_t size, bool SOA, bool can_collapse>
    auto view(thrust_container<value_t, size, SOA, can_collapse>& container,
              const range_t<std::size_t>& range_item,
              const range_t<long long>&   range)
    {
        static_assert(size > 1, "size must be greater than 1");
        return detail::make_item_range_view<value_t, size, SOA, can_collapse>(container, range_item, range);
    }

    template <class value_t, std::size_t size, bool SOA, bool can_collapse>
    auto view(const thrust_container<value_t, size, SOA, can_collapse>& container,
              const range_t<std::size_t>& range_item,
              const range_t<long long>&   range)
    {
        static_assert(size > 1, "size must be greater than 1");
        return detail::make_item_range_view<value_t, size, SOA, can_collapse>(container, range_item, range);
    }

    template <class value_t, std::size_t size, bool SOA, bool can_collapse>
    auto view(thrust_container<value_t, size, SOA, can_collapse>& container, std::size_t index)
    {
        return detail::make_index_view<value_t, size, SOA, can_collapse>(container, index);
    }

    template <class value_t, std::size_t size, bool SOA, bool can_collapse>
    auto view(const thrust_container<value_t, size, SOA, can_collapse>& container, std::size_t index)
    {
        return detail::make_index_view<value_t, size, SOA, can_collapse>(container, index);
    }

    template <class value_t, std::size_t size, bool SOA, bool can_collapse>
    auto view(thrust_container<value_t, size, SOA, can_collapse>& container,
              std::size_t item,
              const range_t<long long>& range)
    {
        static_assert(size > 1, "size must be greater than 1");
        return detail::make_item_view<value_t, size, SOA, can_collapse>(container, item, range);
    }

    template <class value_t, std::size_t size, bool SOA, bool can_collapse>
    auto view(const thrust_container<value_t, size, SOA, can_collapse>& container,
              std::size_t item,
              const range_t<long long>& range)
    {
        static_assert(size > 1, "size must be greater than 1");
        return detail::make_item_view<value_t, size, SOA, can_collapse>(container, item, range);
    }

    template <class Expr>
    auto view(Expr&& expr, const range_t<long long>& range)
        requires detail::thrust_expression<std::decay_t<Expr>>
    {
        return detail::make_expr_slice(detail::decay_copy(std::forward<Expr>(expr)), range);
    }

    template <class T>
    auto view(thrust_view_2d<T>& v2d, placeholders::all_t, std::size_t j)
    {
        return v2d.column(j);
    }

    template <class T>
    auto view(const thrust_view_2d<T>& v2d, placeholders::all_t, std::size_t j)
    {
        return v2d.column(j);
    }

    template <class T>
    auto view(thrust_view_2d<T>& v2d, std::size_t item)
    {
        return v2d.row(item);
    }

    template <class T>
    auto view(const thrust_view_2d<T>& v2d, std::size_t item)
    {
        return v2d.row(item);
    }

    template <class D>
    auto eval(const D& exp)
    {
        return exp; // No lazy path yet; operate eagerly
    }

    template <class D1, class D2>
    bool compare(const D1& a, const D2& b)
    {
        // Generic element-wise compare for 1D-like interfaces with size() and operator[]
        if constexpr (requires { a.size(); b.size(); })
        {
            if (a.size() != b.size()) return false;
            for (std::size_t i = 0; i < a.size(); ++i)
            {
                if (a[i] != b[i]) return false;
            }
            return true;
        }
        else if constexpr (requires { a.items_count(); a.length_count(); b.items_count(); b.length_count(); })
        {
            if (a.items_count() != b.items_count() || a.length_count() != b.length_count())
            {
                return false;
            }
            for (std::size_t j = 0; j < a.length_count(); ++j)
            {
                auto col_a = view(a, placeholders::all(), j);
                auto col_b = view(b, placeholders::all(), j);
                if (!compare(col_a, col_b))
                {
                    return false;
                }
            }
            return true;
        }
        else
        {
            return false;
        }
    }

    template <class T>
    auto shape(const thrust_view_1d<T>& view, std::size_t axis)
    {
        if (axis != 0)
        {
            throw std::out_of_range("axis out of bounds for 1D view");
        }
        return view.size();
    }

    template <class T>
    auto shape(const thrust_view_2d<T>& view, std::size_t axis)
    {
        if (axis == 0)
        {
            return view.items_count();
        }
        else if (axis == 1)
        {
            return view.length_count();
        }
        throw std::out_of_range("axis out of bounds for 2D view");
    }

    template <class T>
    auto shape(const thrust_view_1d<T>& view)
    {
        return std::array<std::size_t, 1>{view.size()};
    }

    template <class T>
    auto shape(const thrust_view_2d<T>& view)
    {
        return std::array<std::size_t, 2>{view.items_count(), view.length_count()};
    }

    template <class Expr>
    auto shape(const Expr& expr, std::size_t axis)
        requires detail::thrust_expression<std::decay_t<Expr>>
    {
        if (axis != 0)
        {
            throw std::out_of_range("axis out of bounds for 1D expression");
        }
        return detail::sequence_size(expr);
    }

    template <class Expr>
    auto shape(const Expr& expr)
        requires detail::thrust_expression<std::decay_t<Expr>>
    {
        return std::array<std::size_t, 1>{detail::sequence_size(expr)};
    }

    template <class D>
    auto noalias(D&& d)
    {
        using view_t = std::decay_t<D>;
        return detail::noalias_proxy<view_t>{std::forward<D>(d)};
    }

    template <class T>
    auto zeros(std::size_t n)
    {
        std::vector<T> v(n);
        std::fill(v.begin(), v.end(), T{});
        return v;
    }

    template <class T1, class T2>
    auto range(const T1& start, const T2& end)
    {
        using value_type = long long;
        return range_t<value_type>{static_cast<value_type>(start), static_cast<value_type>(end), static_cast<value_type>(1)};
    }

    template <class T>
    auto range(const T& start)
    {
        using value_type = long long;
        return range_t<value_type>{static_cast<value_type>(start), std::numeric_limits<value_type>::max(), static_cast<value_type>(1)};
    }

    namespace math
    {
        template <class V>
        auto sum(const V& v)
        {
            using T = std::decay_t<decltype(v[0])>;
            T s{};
            for (std::size_t i = 0; i < v.size(); ++i) s += v[i];
            return s;
        }

        template <class V>
        auto abs(const V& v)
        {
            return detail::thrust_expr_unary<detail::abs_op, std::decay_t<V>>{detail::decay_copy(v)};
        }

        template <class V1, class V2>
        auto minimum(const V1& a, const V2& b)
        {
            return detail::thrust_expr_binary<detail::min_op, std::decay_t<V1>, std::decay_t<V2>>{detail::decay_copy(a),
                                                                                                  detail::decay_copy(b)};
        }

        template <class V1, class V2>
        auto maximum(const V1& a, const V2& b)
        {
            return detail::thrust_expr_binary<detail::max_op, std::decay_t<V1>, std::decay_t<V2>>{detail::decay_copy(a),
                                                                                                  detail::decay_copy(b)};
        }

        template <class T>
        auto transpose(thrust_view_2d<T> v)
        {
            thrust_view_2d<T> res{};
            res.base          = v.base;
            res.items         = v.length_count();
            res.length        = v.items_count();
            res.item_stride   = v.length_stride;
            res.length_stride = v.item_stride;
            return res;
        }
    }


    namespace detail
    {
        template <class Mask>
        struct mask_negation
        {
            Mask mask;

            bool operator[](std::size_t idx) const { return !static_cast<bool>(mask[idx]); }
            std::size_t size() const { return mask.size(); }
        };

        template <class Mask1, class Mask2, class BinaryOp>
        struct mask_binary
        {
            Mask1   lhs;
            Mask2   rhs;
            BinaryOp op;

            bool operator[](std::size_t idx) const
            {
                return op(static_cast<bool>(lhs[idx]), static_cast<bool>(rhs[idx]));
            }

            std::size_t size() const
            {
                const auto s1 = lhs.size();
                const auto s2 = rhs.size();
                if (s1 != s2)
                {
                    throw std::runtime_error("mask size mismatch");
                }
                return s1;
            }
        };

        struct mask_logical_and
        {
            bool operator()(bool a, bool b) const { return a && b; }
        };

        struct mask_logical_or
        {
            bool operator()(bool a, bool b) const { return a || b; }
        };
    }

    template <class D>
        requires detail::mask_indexable<D>
    auto operator>(const D& v, double x)
    {
        struct mask_view
        {
            const D&  ref;
            double    thresh;
            bool operator[](std::size_t i) const { return ref[i] > thresh; }
            std::size_t size() const { return ref.size(); }
        };
        return mask_view{v, x};
    }

    template <class D>
        requires detail::mask_indexable<D>
    auto operator<(const D& v, double x)
    {
        struct mask_view
        {
            const D&  ref;
            double    thresh;
            bool operator[](std::size_t i) const { return ref[i] < thresh; }
            std::size_t size() const { return ref.size(); }
        };
        return mask_view{v, x};
    }

    template <class Mask>
    auto operator!(Mask&& mask)
        requires(requires(const std::decay_t<Mask>& m, std::size_t idx) { m.size(); m[idx]; })
    {
        using mask_type = detail::mask_negation<std::decay_t<Mask>>;
        return mask_type{detail::decay_copy(std::forward<Mask>(mask))};
    }

    template <class LHS, class RHS>
    auto operator&&(LHS&& lhs, RHS&& rhs)
        requires(requires(const std::decay_t<LHS>& l, const std::decay_t<RHS>& r, std::size_t idx) {
            l.size();
            r.size();
            l[idx];
            r[idx];
        })
    {
        using lhs_type = std::decay_t<LHS>;
        using rhs_type = std::decay_t<RHS>;
        using expr_t   = detail::mask_binary<lhs_type, rhs_type, detail::mask_logical_and>;
        return expr_t{detail::decay_copy(std::forward<LHS>(lhs)),
                      detail::decay_copy(std::forward<RHS>(rhs)),
                      detail::mask_logical_and{}};
    }

    template <class LHS, class RHS>
    auto operator||(LHS&& lhs, RHS&& rhs)
        requires(requires(const std::decay_t<LHS>& l, const std::decay_t<RHS>& r, std::size_t idx) {
            l.size();
            r.size();
            l[idx];
            r[idx];
        })
    {
        using lhs_type = std::decay_t<LHS>;
        using rhs_type = std::decay_t<RHS>;
        using expr_t   = detail::mask_binary<lhs_type, rhs_type, detail::mask_logical_or>;
        return expr_t{detail::decay_copy(std::forward<LHS>(lhs)),
                      detail::decay_copy(std::forward<RHS>(rhs)),
                      detail::mask_logical_or{}};
    }

    namespace detail
    {
        template <class Mask>
        bool mask_value(const Mask& mask, std::size_t idx)
        {
            if constexpr (requires { mask[idx]; })
            {
                return static_cast<bool>(mask[idx]);
            }
            else if constexpr (requires { mask(idx); })
            {
                return static_cast<bool>(mask(idx));
            }
            else
            {
                static_assert(dependent_false<Mask>::value, "Unsupported mask type in apply_on_masked");
            }
        }

        template <class View>
        decltype(auto) element_at(View& view, std::size_t idx)
        {
            if constexpr (requires { view[idx]; })
            {
                return (view[idx]);
            }
            else if constexpr (requires { view(idx); })
            {
                return (view(idx));
            }
            else if constexpr (requires { view.items_count(); view.length_count(); })
            {
                const auto items = view.items_count();
                if (items == 0)
                {
                    throw std::out_of_range("element_at on empty 2D view");
                }
                const std::size_t item_idx = idx % items;
                const std::size_t pos_idx  = idx / items;
                return (view(item_idx, pos_idx));
            }
            else
            {
                static_assert(dependent_false<View>::value, "Unsupported view type for apply_on_masked");
            }
        }

        template <class View>
        decltype(auto) element_at(const View& view, std::size_t idx)
        {
            if constexpr (requires { view[idx]; })
            {
                return (view[idx]);
            }
            else if constexpr (requires { view(idx); })
            {
                return (view(idx));
            }
            else if constexpr (requires { view.items_count(); view.length_count(); })
            {
                const auto items = view.items_count();
                if (items == 0)
                {
                    throw std::out_of_range("element_at on empty 2D view");
                }
                const std::size_t item_idx = idx % items;
                const std::size_t pos_idx  = idx / items;
                return (view(item_idx, pos_idx));
            }
            else
            {
                static_assert(dependent_false<View>::value, "Unsupported view type for apply_on_masked");
            }
        }

        template <class Op, class Expr>
        struct thrust_expr_unary
        {
            Expr expr;

            std::size_t size() const
            {
                return sequence_size(expr);
            }

            auto operator[](std::size_t idx) const
            {
                return Op::apply(expr[idx]);
            }
        };

        template <class Op, class L, class R>
        struct thrust_expr_binary
        {
            L lhs;
            R rhs;

            std::size_t size() const
            {
                return std::min(sequence_size(lhs), sequence_size(rhs));
            }

            auto operator[](std::size_t idx) const
            {
                return Op::apply(lhs[idx], rhs[idx]);
            }
        };

        template <class Scalar, class Expr, class Op, bool left>
        struct thrust_expr_scalar
        {
            using scalar_t = std::remove_cvref_t<Scalar>;

            scalar_t scalar;
            Expr      expr;

            std::size_t size() const
            {
                return sequence_size(expr);
            }

            auto operator[](std::size_t idx) const
            {
                if constexpr (left)
                {
                    return Op::apply(scalar, expr[idx]);
                }
                else
                {
                    return Op::apply(expr[idx], scalar);
                }
            }
        };

        template <class Expr>
        struct thrust_expr_slice
        {
            Expr        expr;
            std::size_t start{0};
            std::size_t step{1};
            std::size_t length{0};

            std::size_t size() const
            {
                return length;
            }

            auto operator[](std::size_t idx) const
            {
                return expr[start + idx * step];
            }
        };

        template <class Expr>
        auto make_expr_slice(Expr expr, const range_t<long long>& range)
        {
            if (range.step <= 0)
            {
                throw std::invalid_argument("range.step must be positive");
            }
            const auto total_size = sequence_size(expr);
            if (range.start < 0 || static_cast<std::size_t>(range.start) > total_size)
            {
                throw std::out_of_range("range.start out of bounds for expression slice");
            }
            const long long max_end = static_cast<long long>(total_size);
            const long long clamped_end = (range.end < 0 || range.end > max_end) ? max_end : range.end;
            const auto len = compute_length(range.start, clamped_end, range.step);
            thrust_expr_slice<Expr> slice{std::move(expr),
                                          static_cast<std::size_t>(range.start),
                                          static_cast<std::size_t>(range.step),
                                          len};
            return slice;
        }
    }

    // Masked apply (host/UVM baseline)
    template <class DST, class CRIT, class FUNC>
    void apply_on_masked(DST&& dst, const CRIT& criteria, FUNC&& func)
    {
        for (std::size_t i = 0; i < criteria.size(); ++i)
        {
            if (detail::mask_value(criteria, i))
            {
                auto&& ref = detail::element_at(dst, i);
                func(ref);
            }
        }
    }

    template <class CRIT, class FUNC>
    void apply_on_masked(const CRIT& criteria, FUNC&& func)
    {
        for (std::size_t i = 0; i < criteria.size(); ++i)
        {
            if (detail::mask_value(criteria, i))
            {
                func(i);
            }
        }
    }

    // zeros_like: for a 1D view returns a host vector of zeros of same length
    template <class V>
    auto zeros_like(const V& v)
    {
        using T = std::decay_t<decltype(v[0])>;
        std::vector<T> res(v.size());
        std::fill(res.begin(), res.end(), T{});
        return res;
    }

    template <class L, class R>
        requires(detail::element_indexable<L> && detail::element_indexable<R> && (detail::thrust_sequence<L> || detail::thrust_sequence<R>))
    auto operator+(const L& lhs, const R& rhs)
    {
        return detail::thrust_expr_binary<detail::plus_op, std::decay_t<L>, std::decay_t<R>>{detail::decay_copy(lhs),
                                                                                             detail::decay_copy(rhs)};
    }

    template <class L, class R>
        requires(detail::element_indexable<L> && detail::element_indexable<R> && (detail::thrust_sequence<L> || detail::thrust_sequence<R>))
    auto operator-(const L& lhs, const R& rhs)
    {
        return detail::thrust_expr_binary<detail::minus_op, std::decay_t<L>, std::decay_t<R>>{detail::decay_copy(lhs),
                                                                                               detail::decay_copy(rhs)};
    }

    template <class L, class R>
        requires(detail::element_indexable<L> && detail::element_indexable<R> && (detail::thrust_sequence<L> || detail::thrust_sequence<R>))
    auto operator*(const L& lhs, const R& rhs)
    {
        return detail::thrust_expr_binary<detail::mult_op, std::decay_t<L>, std::decay_t<R>>{detail::decay_copy(lhs),
                                                                                              detail::decay_copy(rhs)};
    }

    template <class L, class R>
        requires(detail::element_indexable<L> && detail::element_indexable<R> && (detail::thrust_sequence<L> || detail::thrust_sequence<R>))
    auto operator/(const L& lhs, const R& rhs)
    {
        return detail::thrust_expr_binary<detail::div_op, std::decay_t<L>, std::decay_t<R>>{detail::decay_copy(lhs),
                                                                                            detail::decay_copy(rhs)};
    }

    template <class Seq>
        requires detail::element_indexable<Seq> && detail::thrust_sequence<Seq>
    auto operator-(const Seq& seq)
    {
        return detail::thrust_expr_unary<detail::neg_op, std::decay_t<Seq>>{detail::decay_copy(seq)};
    }

    template <class Scalar, class Seq>
        requires(detail::scalar_like<Scalar> && detail::element_indexable<Seq> && detail::thrust_sequence<Seq>)
    auto operator*(Scalar s, const Seq& seq)
    {
        return detail::thrust_expr_scalar<Scalar, std::decay_t<Seq>, detail::mult_op, true>{s, detail::decay_copy(seq)};
    }

    template <class Seq, class Scalar>
        requires(detail::element_indexable<Seq> && detail::scalar_like<Scalar> && detail::thrust_sequence<Seq>)
    auto operator*(const Seq& seq, Scalar s)
    {
        return detail::thrust_expr_scalar<Scalar, std::decay_t<Seq>, detail::mult_op, false>{s, detail::decay_copy(seq)};
    }

    template <class Seq, class Scalar>
        requires(detail::element_indexable<Seq> && detail::scalar_like<Scalar> && detail::thrust_sequence<Seq>)
    auto operator/(const Seq& seq, Scalar s)
    {
        return detail::thrust_expr_scalar<Scalar, std::decay_t<Seq>, detail::div_op, false>{s, detail::decay_copy(seq)};
    }

    template <class Seq, class Scalar>
        requires(detail::element_indexable<Seq> && detail::scalar_like<Scalar> && detail::thrust_sequence<Seq>)
    auto operator+(const Seq& seq, Scalar s)
    {
        return detail::thrust_expr_scalar<Scalar, std::decay_t<Seq>, detail::plus_op, false>{s, detail::decay_copy(seq)};
    }

    template <class Scalar, class Seq>
        requires(detail::scalar_like<Scalar> && detail::element_indexable<Seq> && detail::thrust_sequence<Seq>)
    auto operator+(Scalar s, const Seq& seq)
    {
        return detail::thrust_expr_scalar<Scalar, std::decay_t<Seq>, detail::plus_op, true>{s, detail::decay_copy(seq)};
    }

    template <class Seq, class Scalar>
        requires(detail::element_indexable<Seq> && detail::scalar_like<Scalar> && detail::thrust_sequence<Seq>)
    auto operator-(const Seq& seq, Scalar s)
    {
        return detail::thrust_expr_scalar<Scalar, std::decay_t<Seq>, detail::minus_op, false>{s, detail::decay_copy(seq)};
    }

    template <class Scalar, class Seq>
        requires(detail::scalar_like<Scalar> && detail::element_indexable<Seq> && detail::thrust_sequence<Seq>)
    auto operator-(Scalar s, const Seq& seq)
    {
        return detail::thrust_expr_scalar<Scalar, std::decay_t<Seq>, detail::minus_op, true>{s, detail::decay_copy(seq)};
    }

    // Static array aliases (reuse xtensor static forms to avoid reimplementation)
    template <class value_type, std::size_t size, bool /*SOA*/>
    using thrust_static_array = xtensor_static_array<value_type, size>;

    template <class value_type, std::size_t size, bool /*SOA*/, bool can_collapse>
    using thrust_local_collapsable_array = CollapsableArray<thrust_static_array<value_type, size, false>, value_type, size, can_collapse>;
}
