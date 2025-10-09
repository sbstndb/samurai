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
#include <sstream>
#include <cmath>
#include <cassert>
#include <limits>
#include <iterator>

#include <cuda_runtime.h>

#include <xtensor/xlayout.hpp>
#include <xtensor/xexpression.hpp>
#include <xtensor/xiterator.hpp>
#include <xtensor/xstrides.hpp>
#include <xtensor/xslice.hpp>
#include <xtensor/xview.hpp>

#include "../utils.hpp"
#include "../xtensor/xtensor_static.hpp"

namespace samurai
{
    template <class T>
    struct range_t;

    namespace placeholders
    {
        struct all_t;
    }
}

namespace xt
{
    namespace samurai_detail
    {
        template <class T>
        struct is_samurai_range : std::false_type
        {
        };

        template <class T>
        struct is_samurai_range<samurai::range_t<T>> : std::true_type
        {
        };

        template <class T>
        struct is_samurai_all : std::false_type
        {
        };

        template <>
        struct is_samurai_all<samurai::placeholders::all_t> : std::true_type
        {
        };

        template <class T>
        struct is_samurai_slice : std::bool_constant<is_samurai_range<T>::value || is_samurai_all<T>::value>
        {
        };

        template <class Slice>
        inline decltype(auto) convert_slice(Slice&& slice)
        {
            if constexpr (is_samurai_range<std::decay_t<Slice>>::value)
            {
                return xt::range(slice.start, slice.end, slice.step);
            }
            else if constexpr (is_samurai_all<std::decay_t<Slice>>::value)
            {
                return xt::all();
            }
            else
            {
                return std::forward<Slice>(slice);
            }
        }
    } // namespace samurai_detail

    template <class E, class... S>
        requires((samurai_detail::is_samurai_slice<std::decay_t<S>>::value || ...))
    inline auto view(E&& e, S&&... slices)
    {
        return xt::detail::make_view_impl(
            std::forward<E>(e),
            std::make_index_sequence<sizeof...(S)>{},
            samurai_detail::convert_slice(std::forward<S>(slices))...);
    }
}

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
        concept has_bracket_index = requires(const T& v, std::size_t idx)
        {
            v[idx];
        };

        template <class T>
        concept has_call_index = requires(const T& v, std::size_t idx)
        {
            v(idx);
        };

        template <class T>
        concept element_indexable = requires(const T& v)
        {
            { v.size() } -> std::convertible_to<std::size_t>;
        } && (has_bracket_index<T> || has_call_index<T>);

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

        template <class View>
        decltype(auto) element_at(View& view, std::size_t idx);

        template <class View>
        decltype(auto) element_at(const View& view, std::size_t idx);

        template <class Mask>
        std::size_t mask_size(const Mask& mask);

        template <class Mask>
        class mask_const_iterator
        {
          public:
            using iterator_category = std::forward_iterator_tag;
            using difference_type   = std::ptrdiff_t;
            using value_type        = bool;
            using reference         = bool;
            using pointer           = void;

            mask_const_iterator() = default;
            mask_const_iterator(const Mask* mask, std::size_t index)
                : m_mask(mask)
                , m_index(index)
            {
            }

            bool operator*() const
            {
                return (*m_mask)[m_index];
            }

            mask_const_iterator& operator++()
            {
                ++m_index;
                return *this;
            }

            mask_const_iterator operator++(int)
            {
                mask_const_iterator tmp(*this);
                ++(*this);
                return tmp;
            }

            bool operator==(const mask_const_iterator& other) const
            {
                return m_mask == other.m_mask && m_index == other.m_index;
            }

            bool operator!=(const mask_const_iterator& other) const
            {
                return !(*this == other);
            }

          private:
            const Mask* m_mask  = nullptr;
            std::size_t m_index = 0;
        };
    }

    template <class T>
    struct thrust_view_1d;

    // Small RAII for cudaMallocManaged
    template <class T>
    struct managed_ptr
    {
        T* ptr = nullptr;
        std::size_t m_size = 0;

        managed_ptr() = default;

        explicit managed_ptr(std::size_t n)
        {
            allocate(n);
        }

        managed_ptr(const managed_ptr& other)
        {
            copy_from(other);
        }

        managed_ptr& operator=(const managed_ptr& other)
        {
            if (this != &other)
            {
                copy_from(other);
            }
            return *this;
        }

        managed_ptr(managed_ptr&& other) noexcept
            : ptr(other.ptr)
            , m_size(other.m_size)
        {
            other.ptr   = nullptr;
            other.m_size = 0;
        }

        managed_ptr& operator=(managed_ptr&& other) noexcept
        {
            if (this != &other)
            {
                reset();
                ptr        = other.ptr;
                m_size     = other.m_size;
                other.ptr  = nullptr;
                other.m_size = 0;
            }
            return *this;
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
            m_size = n;
        }

        void reset()
        {
            if (ptr)
            {
                ::cudaFree(ptr);
                ptr = nullptr;
            }
            m_size = 0;
        }

        T* get() const { return ptr; }
        std::size_t size() const { return m_size; }
        T& operator[](std::size_t i) const { return ptr[i]; }

      private:
        void copy_from(const managed_ptr& other)
        {
            if (other.m_size == 0)
            {
                reset();
                return;
            }
            if (ptr == nullptr || m_size != other.m_size)
            {
                allocate(other.m_size);
            }
            std::copy(other.ptr, other.ptr + other.m_size, ptr);
            m_size = other.m_size;
        }
    };

    // Managed 2D array wrapper with optional 1D collapsed mode
    template <class T>
    struct managed_array
    {
        using value_type = T;
        using default_view_type = thrust_view_1d<value_type>;
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

        template <class Expr>
        managed_array& operator*=(const Expr& expr)
            requires detail::element_indexable<Expr>
        {
            apply_expression(expr, [](value_type& dst, auto&& factor) { dst *= factor; });
            return *this;
        }

        template <class Expr>
        managed_array& operator/=(const Expr& expr)
            requires detail::element_indexable<Expr>
        {
            apply_expression(expr, [](value_type& dst, auto&& factor) { dst /= factor; });
            return *this;
        }

      private:

        template <class Expr, class Func>
        void apply_expression(const Expr& expr, Func&& func)
        {
            using expr_t = std::decay_t<Expr>;
            const std::size_t expr_size = static_cast<std::size_t>(expr.size());
            const auto fetch = [&](std::size_t idx)
            {
                if constexpr (detail::has_bracket_index<expr_t>)
                {
                    return expr[idx];
                }
                else
                {
                    return expr(idx);
                }
            };

            const std::size_t total = size();
            if (expr_size == 0)
            {
                throw std::runtime_error("Expression has zero size");
            }

            if (expr_size == 1)
            {
                const auto factor = fetch(0);
                for (std::size_t i = 0; i < total; ++i)
                {
                    func(m_buf[i], factor);
                }
                return;
            }

            if (m_collapsed)
            {
                if (expr_size != total)
                {
                    throw std::runtime_error("Incompatible expression size for collapsed managed_array");
                }
                for (std::size_t i = 0; i < total; ++i)
                {
                    func(m_buf[i], fetch(i));
                }
                return;
            }

            const std::size_t rows = m_rows;
            const std::size_t cols = m_cols;
            if (expr_size == total)
            {
                for (std::size_t i = 0; i < total; ++i)
                {
                    func(m_buf[i], fetch(i));
                }
                return;
            }

            if (expr_size == rows && rows <= cols)
            {
                for (std::size_t r = 0; r < rows; ++r)
                {
                    const auto factor = fetch(r);
                    const std::size_t base = r * cols;
                    for (std::size_t c = 0; c < cols; ++c)
                    {
                        func(m_buf[base + c], factor);
                    }
                }
                return;
            }

            if (expr_size == cols)
            {
                for (std::size_t r = 0; r < rows; ++r)
                {
                    const std::size_t base = r * cols;
                    for (std::size_t c = 0; c < cols; ++c)
                    {
                        func(m_buf[base + c], fetch(c));
                    }
                }
                return;
            }

            if (expr_size == rows && rows > cols)
            {
                for (std::size_t r = 0; r < rows; ++r)
                {
                    const auto factor = fetch(r);
                    const std::size_t base = r * cols;
                    for (std::size_t c = 0; c < cols; ++c)
                    {
                        func(m_buf[base + c], factor);
                    }
                }
                return;
            }

            throw std::runtime_error("Expression size incompatible with managed_array dimensions");
        }
    };

    // Lightweight 1D view (contiguous with step)
    template <class T>
    struct thrust_view_1d : xt::xexpression<thrust_view_1d<T>>
    {
        thrust_view_1d() = default;
        thrust_view_1d(const thrust_view_1d&) = default;
        thrust_view_1d(thrust_view_1d&&) = default;
        thrust_view_1d& operator=(const thrust_view_1d&) = default;
        thrust_view_1d& operator=(thrust_view_1d&&) = default;
        ~thrust_view_1d() = default;

        using value_type   = T;
        using scalar_type  = std::remove_const_t<T>;
        using pointer_type = value_type*;
        using pointer       = pointer_type;
        using const_pointer = const scalar_type*;
        using size_type    = std::size_t;
        using shape_type   = std::array<std::size_t, 1>;
        using difference_type = std::ptrdiff_t;
        using reference       = value_type&;
        using const_reference = const value_type&;
        using bool_load_type  = bool;
        using storage_type    = thrust_view_1d;
        using inner_shape_type       = shape_type;
        using inner_strides_type     = shape_type;
        using inner_backstrides_type = shape_type;
        using storage_iterator              = pointer_type;
        using const_storage_iterator        = const_pointer;
        using storage_reverse_iterator      = std::reverse_iterator<storage_iterator>;
        using const_storage_reverse_iterator = std::reverse_iterator<const_storage_iterator>;
        using iterator               = storage_iterator;
        using const_iterator         = const_storage_iterator;
        using reverse_iterator       = storage_reverse_iterator;
        using const_reverse_iterator = const_storage_reverse_iterator;
        using expression_tag         = xt::xtensor_expression_tag;
        using stepper                = xt::xindexed_stepper<thrust_view_1d, false>;
        using const_stepper          = xt::xindexed_stepper<thrust_view_1d, true>;
        inline static constexpr bool contiguous_layout = false;
        inline static constexpr xt::layout_type static_layout =
#ifdef SAMURAI_CONTAINER_LAYOUT_COL_MAJOR
            xt::layout_type::column_major;
#else
            xt::layout_type::row_major;
#endif
        inline static constexpr layout_type samurai_layout =
#ifdef SAMURAI_CONTAINER_LAYOUT_COL_MAJOR
            layout_type::column_major;
#else
            layout_type::row_major;
#endif

        pointer_type base{nullptr};
        std::size_t  start{0};
        std::size_t  stop{0};
        std::size_t  step{1};
        std::size_t  stride{1}; // element stride in underlying storage

        std::size_t size() const
        {
            if (stop <= start)
            {
                return 0;
            }
            return (stop - start + (step - 1)) / step;
        }

        std::size_t dimension() const { return 1; }

        std::size_t shape(std::size_t axis) const
        {
            if (axis == 0)
            {
                return size();
            }
            throw std::out_of_range("axis out of bounds for thrust_view_1d");
        }

        std::array<std::size_t, 1> shape() const
        {
            return {size()};
        }

        std::size_t stride_value() const
        {
            return step * stride;
        }

        inner_strides_type strides() const
        {
            return {stride_value()};
        }

        inner_backstrides_type backstrides() const
        {
            if (size() == 0)
            {
                return {0};
            }
            return {(size() - 1) * stride_value()};
        }

        xt::layout_type layout() const
        {
            return static_layout;
        }

        bool is_contiguous() const
        {
            return step == 1 && stride == 1;
        }

        std::size_t storage_span() const
        {
            if (size() == 0)
            {
                return 0;
            }
            return (size() - 1) * stride_value() + 1;
        }

        pointer data()
            requires(!std::is_const_v<value_type>)
        {
            return base + start * stride;
        }

        const_pointer data() const
        {
            return base + start * stride;
        }

        storage_iterator storage_begin()
            requires(!std::is_const_v<value_type>)
        {
            return data();
        }

        storage_iterator storage_end()
            requires(!std::is_const_v<value_type>)
        {
            return data() + storage_span();
        }

        const_storage_iterator storage_begin() const
        {
            return data();
        }

        const_storage_iterator storage_end() const
        {
            return data() + storage_span();
        }

        storage_reverse_iterator storage_rbegin()
            requires(!std::is_const_v<value_type>)
        {
            return storage_reverse_iterator(storage_end());
        }

        storage_reverse_iterator storage_rend()
            requires(!std::is_const_v<value_type>)
        {
            return storage_reverse_iterator(storage_begin());
        }

        const_storage_reverse_iterator storage_rbegin() const
        {
            return const_storage_reverse_iterator(storage_end());
        }

        const_storage_reverse_iterator storage_rend() const
        {
            return const_storage_reverse_iterator(storage_begin());
        }

        const_storage_iterator storage_cbegin() const
        {
            return storage_begin();
        }

        const_storage_iterator storage_cend() const
        {
            return storage_end();
        }

        const_storage_reverse_iterator storage_crbegin() const
        {
            return const_storage_reverse_iterator(storage_end());
        }

        const_storage_reverse_iterator storage_crend() const
        {
            return const_storage_reverse_iterator(storage_begin());
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

        value_type& operator()(std::size_t i)
            requires(!std::is_const_v<value_type>)
        {
            return (*this)[i];
        }

        const value_type& operator()(std::size_t i) const
        {
            return (*this)[i];
        }

        iterator begin()
            requires(!std::is_const_v<value_type>)
        {
            return storage_begin();
        }

        iterator end()
            requires(!std::is_const_v<value_type>)
        {
            return storage_end();
        }

        const_iterator begin() const
        {
            return storage_begin();
        }

        const_iterator end() const
        {
            return storage_end();
        }

        const_iterator cbegin() const
        {
            return storage_begin();
        }

        const_iterator cend() const
        {
            return storage_end();
        }

        reverse_iterator rbegin()
            requires(!std::is_const_v<value_type>)
        {
            return storage_rbegin();
        }

        reverse_iterator rend()
            requires(!std::is_const_v<value_type>)
        {
            return storage_rend();
        }

        const_reverse_iterator rbegin() const
        {
            return storage_rbegin();
        }

        const_reverse_iterator rend() const
        {
            return storage_rend();
        }

        const_reverse_iterator crbegin() const
        {
            return storage_rbegin();
        }

        const_reverse_iterator crend() const
        {
            return storage_rend();
        }

        storage_type& storage()
            requires(!std::is_const_v<value_type>)
        {
            return *this;
        }

        const storage_type& storage() const
        {
            return *this;
        }

        std::size_t data_offset() const
        {
            return start * stride;
        }

        template <class Strides>
        bool has_linear_assign(const Strides&) const
        {
            return false;
        }

        template <class S>
        bool broadcast_shape(S& shape, bool /*reuse_cache*/ = false) const
        {
            return xt::broadcast_shape(this->shape(), shape);
        }

        template <class S>
        auto stepper_begin(const S&) requires(!std::is_const_v<value_type>)
        {
            return xt::xindexed_stepper<thrust_view_1d, false>(this, data_offset());
        }

        template <class S>
        auto stepper_end(const S&, xt::layout_type) requires(!std::is_const_v<value_type>)
        {
            return xt::xindexed_stepper<thrust_view_1d, false>(this, size(), true);
        }

        template <class S>
        auto stepper_begin(const S&) const
        {
            return xt::xindexed_stepper<thrust_view_1d, true>(this, data_offset());
        }

        template <class S>
        auto stepper_end(const S&, xt::layout_type) const
        {
            return xt::xindexed_stepper<thrust_view_1d, true>(this, size(), true);
        }

        template <class It>
        reference element(It first, It last)
            requires(!std::is_const_v<value_type>)
        {
            assert(std::distance(first, last) == 1);
            (void)last;
            return (*this)(static_cast<std::size_t>(*first));
        }

        template <class It>
        const_reference element(It first, It last) const
        {
            assert(std::distance(first, last) == 1);
            (void)last;
            return (*this)(static_cast<std::size_t>(*first));
        }

        template <class V>
        thrust_view_1d& operator=(const V& rhs)
            requires(!std::is_const_v<value_type> && detail::element_indexable<V>)
        {
            detail::assign_elements(*this, rhs);
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
            detail::add_elements(*this, rhs);
            return *this;
        }

        template <class V>
        thrust_view_1d& operator-=(const V& rhs)
            requires(!std::is_const_v<value_type> && detail::element_indexable<V>)
        {
            detail::sub_elements(*this, rhs);
            return *this;
        }

        template <class V>
        thrust_view_1d& operator&=(const V& rhs)
            requires(!std::is_const_v<value_type> && detail::element_indexable<V>)
        {
            const auto n = size();
            for (std::size_t i = 0; i < n; ++i)
            {
                auto&& ref = detail::element_at(*this, i);
                ref &= static_cast<value_type>(detail::element_at(rhs, i));
            }
            return *this;
        }

        thrust_view_1d& operator&=(value_type value)
            requires(!std::is_const_v<value_type>)
        {
            auto n = size();
            for (std::size_t i = 0; i < n; ++i)
            {
                (*this)[i] &= value;
            }
            return *this;
        }

        template <class V>
        thrust_view_1d& operator|=(const V& rhs)
            requires(!std::is_const_v<value_type> && detail::element_indexable<V>)
        {
            const auto n = size();
            for (std::size_t i = 0; i < n; ++i)
            {
                auto&& ref = detail::element_at(*this, i);
                ref |= static_cast<value_type>(detail::element_at(rhs, i));
            }
            return *this;
        }

        thrust_view_1d& operator|=(value_type value)
            requires(!std::is_const_v<value_type>)
        {
            auto n = size();
            for (std::size_t i = 0; i < n; ++i)
            {
                (*this)[i] |= value;
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
    struct thrust_view_2d : xt::xexpression<thrust_view_2d<T>>
    {
        thrust_view_2d() = default;
        thrust_view_2d(const thrust_view_2d&) = default;
        thrust_view_2d(thrust_view_2d&&) = default;
        thrust_view_2d& operator=(const thrust_view_2d&) = default;
        thrust_view_2d& operator=(thrust_view_2d&&) = default;
        ~thrust_view_2d() = default;

        using value_type   = T;
        using scalar_type  = std::remove_const_t<std::decay_t<T>>;
        using pointer_type = value_type*;
        using pointer       = pointer_type;
        using const_pointer = const scalar_type*;
        using size_type    = std::size_t;
        using shape_type   = std::array<std::size_t, 2>;
        using difference_type = std::ptrdiff_t;
        using reference       = value_type&;
        using const_reference = const value_type&;
        using bool_load_type  = bool;
        using inner_shape_type       = shape_type;
        using inner_strides_type     = shape_type;
        using inner_backstrides_type = shape_type;
        using storage_type           = thrust_view_2d;
        using storage_iterator              = pointer_type;
        using const_storage_iterator        = const_pointer;
        using storage_reverse_iterator      = std::reverse_iterator<storage_iterator>;
        using const_storage_reverse_iterator = std::reverse_iterator<const_storage_iterator>;
        using iterator               = storage_iterator;
        using const_iterator         = const_storage_iterator;
        using reverse_iterator       = storage_reverse_iterator;
        using const_reverse_iterator = const_storage_reverse_iterator;
        using expression_tag         = xt::xtensor_expression_tag;
        using stepper                = xt::xindexed_stepper<thrust_view_2d, false>;
        using const_stepper          = xt::xindexed_stepper<thrust_view_2d, true>;
        inline static constexpr bool contiguous_layout = false;
        inline static constexpr xt::layout_type static_layout =
#ifdef SAMURAI_CONTAINER_LAYOUT_COL_MAJOR
            xt::layout_type::column_major;
#else
            xt::layout_type::row_major;
#endif
        inline static constexpr layout_type samurai_layout =
#ifdef SAMURAI_CONTAINER_LAYOUT_COL_MAJOR
            layout_type::column_major;
#else
            layout_type::row_major;
#endif

        pointer_type base{nullptr};
        std::size_t  items{0};   // number of components/items
        std::size_t  length{0};  // number of positions along the range dimension
        std::size_t  item_stride{1};
        std::size_t  length_stride{1};

        std::size_t items_count() const { return items; }
        std::size_t length_count() const { return length; }
        std::size_t size() const { return items * length; }

        std::size_t dimension() const { return 2; }

        std::size_t shape(std::size_t axis) const
        {
            if (axis == 0)
            {
                return items_count();
            }
            if (axis == 1)
            {
                return length_count();
            }
            throw std::out_of_range("axis out of bounds for thrust_view_2d");
        }

        std::array<std::size_t, 2> shape() const
        {
            return {items_count(), length_count()};
        }

        inner_strides_type strides() const
        {
            return {item_stride, length_stride};
        }

        inner_backstrides_type backstrides() const
        {
            if (items == 0 || length == 0)
            {
                return {0, 0};
            }
            return {(items - 1) * item_stride, (length - 1) * length_stride};
        }

        xt::layout_type layout() const
        {
            return static_layout;
        }

        bool is_contiguous() const
        {
            return false;
        }

        std::size_t storage_span() const
        {
            if (items == 0 || length == 0)
            {
                return 0;
            }
            const std::size_t max_item_offset   = (items - 1) * item_stride;
            const std::size_t max_length_offset = (length - 1) * length_stride;
            return max_item_offset + max_length_offset + 1;
        }

        pointer_type data()
            requires(!std::is_const_v<value_type>)
        {
            return base;
        }

        const_pointer data() const
        {
            return base;
        }

        storage_iterator storage_begin()
            requires(!std::is_const_v<value_type>)
        {
            return data();
        }

        storage_iterator storage_end()
            requires(!std::is_const_v<value_type>)
        {
            return data() + storage_span();
        }

        const_storage_iterator storage_begin() const
        {
            return data();
        }

        const_storage_iterator storage_end() const
        {
            return data() + storage_span();
        }

        storage_reverse_iterator storage_rbegin()
            requires(!std::is_const_v<value_type>)
        {
            return storage_reverse_iterator(storage_end());
        }

        storage_reverse_iterator storage_rend()
            requires(!std::is_const_v<value_type>)
        {
            return storage_reverse_iterator(storage_begin());
        }

        const_storage_reverse_iterator storage_rbegin() const
        {
            return const_storage_reverse_iterator(storage_end());
        }

        const_storage_reverse_iterator storage_rend() const
        {
            return const_storage_reverse_iterator(storage_begin());
        }

        const_storage_iterator storage_cbegin() const
        {
            return storage_begin();
        }

        const_storage_iterator storage_cend() const
        {
            return storage_end();
        }

        const_storage_reverse_iterator storage_crbegin() const
        {
            return const_storage_reverse_iterator(storage_end());
        }

        const_storage_reverse_iterator storage_crend() const
        {
            return const_storage_reverse_iterator(storage_begin());
        }

        iterator begin()
            requires(!std::is_const_v<value_type>)
        {
            return storage_begin();
        }

        iterator end()
            requires(!std::is_const_v<value_type>)
        {
            return storage_end();
        }

        const_iterator begin() const
        {
            return storage_begin();
        }

        const_iterator end() const
        {
            return storage_end();
        }

        const_iterator cbegin() const
        {
            return storage_begin();
        }

        const_iterator cend() const
        {
            return storage_end();
        }

        reverse_iterator rbegin()
            requires(!std::is_const_v<value_type>)
        {
            return storage_rbegin();
        }

        reverse_iterator rend()
            requires(!std::is_const_v<value_type>)
        {
            return storage_rend();
        }

        const_reverse_iterator rbegin() const
        {
            return storage_rbegin();
        }

        const_reverse_iterator rend() const
        {
            return storage_rend();
        }

        const_reverse_iterator crbegin() const
        {
            return storage_rbegin();
        }

        const_reverse_iterator crend() const
        {
            return storage_rend();
        }

        storage_type& storage()
            requires(!std::is_const_v<value_type>)
        {
            return *this;
        }

        const storage_type& storage() const
        {
            return *this;
        }

        std::size_t data_offset() const
        {
            return 0;
        }

        template <class Strides>
        bool has_linear_assign(const Strides&) const
        {
            return false;
        }

        template <class S>
        bool broadcast_shape(S& shape, bool /*reuse_cache*/ = false) const
        {
            return xt::broadcast_shape(this->shape(), shape);
        }

        template <class S>
        auto stepper_begin(const S&) requires(!std::is_const_v<value_type>)
        {
            return xt::xindexed_stepper<thrust_view_2d, false>(this, data_offset());
        }

        template <class S>
        auto stepper_end(const S&, xt::layout_type) requires(!std::is_const_v<value_type>)
        {
            return xt::xindexed_stepper<thrust_view_2d, false>(this, size(), true);
        }

        template <class S>
        auto stepper_begin(const S&) const
        {
            return xt::xindexed_stepper<thrust_view_2d, true>(this, data_offset());
        }

        template <class S>
        auto stepper_end(const S&, xt::layout_type) const
        {
            return xt::xindexed_stepper<thrust_view_2d, true>(this, size(), true);
        }

        template <class It>
        reference element(It first, It last)
            requires(!std::is_const_v<value_type>)
        {
            assert(std::distance(first, last) == 2);
            (void)last;
            auto item_idx = static_cast<std::size_t>(*first);
            auto pos_idx  = static_cast<std::size_t>(*(first + 1));
            return (*this)(item_idx, pos_idx);
        }

        template <class It>
        const_reference element(It first, It last) const
        {
            assert(std::distance(first, last) == 2);
            (void)last;
            auto item_idx = static_cast<std::size_t>(*first);
            auto pos_idx  = static_cast<std::size_t>(*(first + 1));
            return (*this)(item_idx, pos_idx);
        }

        value_type& operator()(std::size_t idx)
            requires(!std::is_const_v<value_type>)
        {
            return (*this)[idx];
        }

        const value_type& operator()(std::size_t idx) const
        {
            return (*this)[idx];
        }

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
            v.stop   = items;
            v.step   = 1;
            v.stride = item_stride;
            return v;
        }

        thrust_view_1d<value_type> row(std::size_t item) const
        {
            thrust_view_1d<value_type> v;
            v.base   = base + item * item_stride;
            v.start  = 0;
            v.stop   = length;
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

    template <class L, class R>
    bool operator==(const thrust_view_1d<L>& lhs, const thrust_view_1d<R>& rhs)
    {
        const std::size_t n = lhs.size();
        if (n != rhs.size())
        {
            return false;
        }
        for (std::size_t i = 0; i < n; ++i)
        {
            if (lhs[i] != rhs[i])
            {
                return false;
            }
        }
        return true;
    }

    template <class L, class R>
    bool operator!=(const thrust_view_1d<L>& lhs, const thrust_view_1d<R>& rhs)
    {
        return !(lhs == rhs);
    }

    template <class L, class R>
    bool operator==(const thrust_view_2d<L>& lhs, const thrust_view_2d<R>& rhs)
    {
        if (lhs.items_count() != rhs.items_count() || lhs.length_count() != rhs.length_count())
        {
            return false;
        }
        const std::size_t total = lhs.size();
        for (std::size_t idx = 0; idx < total; ++idx)
        {
            if (lhs[idx] != rhs[idx])
            {
                return false;
            }
        }
        return true;
    }

    template <class L, class R>
    bool operator!=(const thrust_view_2d<L>& lhs, const thrust_view_2d<R>& rhs)
    {
        return !(lhs == rhs);
    }

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

        template <class T>
        struct vector_wrapper;

        template <class T>
        struct is_vector_wrapper : std::false_type
        {
        };

        template <class T>
        struct is_vector_wrapper<vector_wrapper<T>> : std::true_type
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
        concept thrust_sequence = is_thrust_view<std::decay_t<T>>::value || thrust_expression<T> || is_std_vector<std::decay_t<T>>::value
            || is_vector_wrapper<std::decay_t<T>>::value;

        template <class T>
        concept has_size_method = requires(const T& v) {
            { v.size() } -> std::convertible_to<std::size_t>;
        };

        template <class T>
        concept has_items_length = requires(const T& v) {
            { v.items_count() } -> std::convertible_to<std::size_t>;
            { v.length_count() } -> std::convertible_to<std::size_t>;
        };

        template <class T>
        concept sequence_accessible = has_size_method<T> || has_items_length<T> || thrust_expression<T>;

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

        struct expression_shape
        {
            std::size_t items{0};
            std::size_t length{0};
            bool        valid{false};
        };

        template <class Expr>
        expression_shape infer_shape(const Expr& expr)
        {
            if constexpr (requires { expr.items_count(); expr.length_count(); })
            {
                return expression_shape{expr.items_count(), expr.length_count(), true};
            }
            else if constexpr (requires { expr.lhs; })
            {
                auto lhs_shape = infer_shape(expr.lhs);
                if (lhs_shape.valid)
                {
                    return lhs_shape;
                }
                if constexpr (requires { expr.rhs; })
                {
                    return infer_shape(expr.rhs);
                }
            }
            else if constexpr (requires { expr.expr; })
            {
                return infer_shape(expr.expr);
            }
            else if constexpr (requires { expr.mask; })
            {
                return infer_shape(expr.mask);
            }
            else if constexpr (requires { expr.ref; })
            {
                return infer_shape(expr.ref);
            }

            return expression_shape{0u, detail::sequence_size(expr), false};
        }

        template <class T>
        struct vector_wrapper
        {
            std::vector<T> data;

            std::size_t size() const { return data.size(); }

            void resize(std::size_t n) { data.resize(n); }

            T operator[](std::size_t idx) const
            {
                return data[idx];
            }

            void set(std::size_t idx, T value)
            {
                data[idx] = value;
            }
        };

        template <>
        struct vector_wrapper<bool>
        {
            std::vector<unsigned char> data;

            std::size_t size() const { return data.size(); }

            void resize(std::size_t n) { data.resize(n); }

            bool operator[](std::size_t idx) const
            {
                return data[idx] != 0u;
            }

            void set(std::size_t idx, bool value)
            {
                data[idx] = value ? 1u : 0u;
            }
        };

        template <class Expr>
        auto evaluate_to_vector(const Expr& expr)
        {
            using value_t = element_value_t<Expr>;
            vector_wrapper<value_t> out;
            const auto total = detail::sequence_size(expr);
            out.resize(total);
            for (std::size_t idx = 0; idx < total; ++idx)
            {
                out.set(idx, element_at(expr, idx));
            }
            return out;
        }

        template <class Container, class Expr, class Func>
        void apply_container_inplace(Container& container, const Expr& expr, Func&& func)
        {
            using value_t = typename Container::value_type;
            auto expr_size = detail::sequence_size(expr);
            auto& data     = container.data();
            auto* buffer   = data.data();

            auto apply_value = [&](std::size_t idx, auto&& factor)
            {
                func(buffer[idx], static_cast<value_t>(factor));
            };

            if (data.collapsed())
            {
                const std::size_t total = data.size();
                if (expr_size == 1)
                {
                    const auto factor = element_at(expr, 0);
                    for (std::size_t i = 0; i < total; ++i)
                    {
                        apply_value(i, factor);
                    }
                }
                else if (expr_size == total)
                {
                    for (std::size_t i = 0; i < total; ++i)
                    {
                        apply_value(i, element_at(expr, i));
                    }
                }
                else
                {
                    throw std::runtime_error("Incompatible expression size for collapsed container");
                }
            }
            else
            {
                const std::size_t rows = data.rows();
                const std::size_t cols = data.cols();
                if constexpr (static_size_first_v<Container::static_size, Container::static_soa, Container::static_can_collapse, Container::static_layout>)
                {
                    const std::size_t comps   = rows;
                    const std::size_t entries = rows * cols;
                    if (expr_size == comps)
                    {
                        for (std::size_t comp = 0; comp < comps; ++comp)
                        {
                            const auto factor = element_at(expr, comp);
                            const std::size_t row_offset = comp * cols;
                            for (std::size_t j = 0; j < cols; ++j)
                            {
                                apply_value(row_offset + j, factor);
                            }
                        }
                    }
                    else if (expr_size == entries)
                    {
                        for (std::size_t idx = 0; idx < entries; ++idx)
                        {
                            apply_value(idx, element_at(expr, idx));
                        }
                    }
                    else
                    {
                        throw std::runtime_error("Incompatible expression size for component-major container");
                    }
                }
                else
                {
                    const std::size_t comps   = cols;
                    const std::size_t entries = rows * cols;
                    if (expr_size == comps)
                    {
                        for (std::size_t row = 0; row < rows; ++row)
                        {
                            const std::size_t row_offset = row * cols;
                            for (std::size_t comp = 0; comp < comps; ++comp)
                            {
                                apply_value(row_offset + comp, element_at(expr, comp));
                            }
                        }
                    }
                    else if (expr_size == entries)
                    {
                        for (std::size_t idx = 0; idx < entries; ++idx)
                        {
                            apply_value(idx, element_at(expr, idx));
                        }
                    }
                    else if (expr_size == rows)
                    {
                        for (std::size_t row = 0; row < rows; ++row)
                        {
                            const auto factor = element_at(expr, row);
                            const std::size_t row_offset = row * cols;
                            for (std::size_t comp = 0; comp < cols; ++comp)
                            {
                                apply_value(row_offset + comp, factor);
                            }
                        }
                    }
                    else
                    {
                        throw std::runtime_error("Incompatible expression size for cell-major container");
                    }
                }
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

        struct bit_and_op
        {
            template <class A, class B>
            static auto apply(A&& a, B&& b)
            {
                return std::forward<A>(a) & std::forward<B>(b);
            }
        };

        struct bit_or_op
        {
            template <class A, class B>
            static auto apply(A&& a, B&& b)
            {
                return std::forward<A>(a) | std::forward<B>(b);
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
                v.base   = base_ptr;
                v.start  = start_sz;
                v.stop   = static_cast<std::size_t>(range.end);
                v.step   = step_sz;
                v.stride = 1;
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
                v.stop   = 1;
                v.step   = 1;
                v.stride = 1;
            }
            else if constexpr (static_size_first_v<size, SOA, can_collapse, SAMURAI_DEFAULT_LAYOUT>)
            {
                const std::size_t cols = data.cols();
                v.base   = base_ptr + index;
                v.start  = 0;
                v.stop   = size;
                v.step   = 1;
                v.stride = cols;
            }
            else
            {
                const std::size_t cols = data.cols();
                v.base   = base_ptr + index * cols;
                v.start  = 0;
                v.stop   = size;
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
                v.stop   = len;
                v.step   = 1;
                v.stride = step_sz;
            }
            else
            {
                const std::size_t cols = data.cols();
                v.base   = base_ptr + start_sz * cols + item;
                v.start  = 0;
                v.stop   = len;
                v.step   = 1;
                v.stride = step_sz * cols;
            }
            return v;
        }

        template <class DST, class SRC>
        void assign_elements(DST& dst, const SRC& src)
        {
            using src_decay = std::decay_t<SRC>;
            if constexpr (std::is_arithmetic_v<src_decay>)
            {
                if constexpr (requires { dst.size(); } && !requires { dst.column(std::size_t{}); })
                {
                    const auto n = dst.size();
                    for (std::size_t i = 0; i < n; ++i)
                    {
                        auto&& ref = detail::element_at(dst, i);
                        using ref_t = std::remove_reference_t<decltype(ref)>;
                        ref        = static_cast<ref_t>(src);
                    }
                }
                else if constexpr (requires { dst.column(std::size_t{}); dst.length_count(); })
                {
                    for (std::size_t j = 0; j < dst.length_count(); ++j)
                    {
                        auto dst_col = dst.column(j);
                        assign_elements(dst_col, src);
                    }
                }
                else if constexpr (detail::sequence_accessible<std::decay_t<DST>>)
                {
                    const auto n = detail::sequence_size(dst);
                    for (std::size_t i = 0; i < n; ++i)
                    {
                        auto&& ref = detail::element_at(dst, i);
                        using ref_t = std::remove_reference_t<decltype(ref)>;
                        ref        = static_cast<ref_t>(src);
                    }
                }
                else
                {
                    static_assert(dependent_false<DST>::value, "Unsupported scalar assignment target");
                }
            }
            else if constexpr (requires { dst.size(); src.size(); } && !requires { dst.column(std::size_t{}); })
            {
                assert(dst.size() == src.size());
                for (std::size_t i = 0; i < dst.size(); ++i)
                {
                    auto&& ref = detail::element_at(dst, i);
                    using ref_t = std::remove_reference_t<decltype(ref)>;
                    ref        = static_cast<ref_t>(detail::element_at(src, i));
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
            else if constexpr (detail::sequence_accessible<std::decay_t<DST>> && detail::sequence_accessible<std::decay_t<SRC>>)
            {
                const auto dst_size = detail::sequence_size(dst);
                const auto src_size = detail::sequence_size(src);
                assert(dst_size == src_size);
                (void)src_size;
                for (std::size_t i = 0; i < dst_size; ++i)
                {
                    auto&& ref = detail::element_at(dst, i);
                    ref        = detail::element_at(src, i);
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
            using src_decay = std::decay_t<SRC>;
            if constexpr (std::is_arithmetic_v<src_decay>)
            {
                if constexpr (requires { dst.size(); } && !requires { dst.column(std::size_t{}); })
                {
                    const auto n = dst.size();
                    for (std::size_t i = 0; i < n; ++i)
                    {
                        auto&& ref = detail::element_at(dst, i);
                        using ref_t = std::remove_reference_t<decltype(ref)>;
                        ref += static_cast<ref_t>(src);
                    }
                }
                else if constexpr (requires { dst.column(std::size_t{}); dst.length_count(); })
                {
                    for (std::size_t j = 0; j < dst.length_count(); ++j)
                    {
                        auto dst_col = dst.column(j);
                        add_elements(dst_col, src);
                    }
                }
                else if constexpr (detail::sequence_accessible<std::decay_t<DST>>)
                {
                    const auto n = detail::sequence_size(dst);
                    for (std::size_t i = 0; i < n; ++i)
                    {
                        auto&& ref = detail::element_at(dst, i);
                        using ref_t = std::remove_reference_t<decltype(ref)>;
                        ref += static_cast<ref_t>(src);
                    }
                }
                else
                {
                    static_assert(dependent_false<DST>::value, "Unsupported scalar addition target");
                }
            }
            else if constexpr (requires { dst.size(); src.size(); } && !requires { dst.column(std::size_t{}); })
            {
                assert(dst.size() == src.size());
                for (std::size_t i = 0; i < dst.size(); ++i)
                {
                    auto&& ref = detail::element_at(dst, i);
                    using ref_t = std::remove_reference_t<decltype(ref)>;
                    ref += static_cast<ref_t>(detail::element_at(src, i));
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
            else if constexpr (detail::sequence_accessible<std::decay_t<DST>> && detail::sequence_accessible<std::decay_t<SRC>>)
            {
                const auto dst_size = detail::sequence_size(dst);
                const auto src_size = detail::sequence_size(src);
                assert(dst_size == src_size);
                (void)src_size;
                for (std::size_t i = 0; i < dst_size; ++i)
                {
                    auto&& ref = detail::element_at(dst, i);
                    ref += detail::element_at(src, i);
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
            using src_decay = std::decay_t<SRC>;
            if constexpr (std::is_arithmetic_v<src_decay>)
            {
                if constexpr (requires { dst.size(); } && !requires { dst.column(std::size_t{}); })
                {
                    const auto n = dst.size();
                    for (std::size_t i = 0; i < n; ++i)
                    {
                        auto&& ref = detail::element_at(dst, i);
                        using ref_t = std::remove_reference_t<decltype(ref)>;
                        ref -= static_cast<ref_t>(src);
                    }
                }
                else if constexpr (requires { dst.column(std::size_t{}); dst.length_count(); })
                {
                    for (std::size_t j = 0; j < dst.length_count(); ++j)
                    {
                        auto dst_col = dst.column(j);
                        sub_elements(dst_col, src);
                    }
                }
                else if constexpr (detail::sequence_accessible<std::decay_t<DST>>)
                {
                    const auto n = detail::sequence_size(dst);
                    for (std::size_t i = 0; i < n; ++i)
                    {
                        auto&& ref = detail::element_at(dst, i);
                        using ref_t = std::remove_reference_t<decltype(ref)>;
                        ref -= static_cast<ref_t>(src);
                    }
                }
                else
                {
                    static_assert(dependent_false<DST>::value, "Unsupported scalar subtraction target");
                }
            }
            else if constexpr (requires { dst.size(); src.size(); } && !requires { dst.column(std::size_t{}); })
            {
                assert(dst.size() == src.size());
                for (std::size_t i = 0; i < dst.size(); ++i)
                {
                    auto&& ref = detail::element_at(dst, i);
                    using ref_t = std::remove_reference_t<decltype(ref)>;
                    ref -= static_cast<ref_t>(detail::element_at(src, i));
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
            else if constexpr (detail::sequence_accessible<std::decay_t<DST>> && detail::sequence_accessible<std::decay_t<SRC>>)
            {
                const auto dst_size = detail::sequence_size(dst);
                const auto src_size = detail::sequence_size(src);
                assert(dst_size == src_size);
                (void)src_size;
                for (std::size_t i = 0; i < dst_size; ++i)
                {
                    auto&& ref = detail::element_at(dst, i);
                    ref -= detail::element_at(src, i);
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
        static constexpr std::size_t static_size   = size;
        static constexpr bool        static_soa    = SOA;
        static constexpr bool        static_can_collapse = can_collapse;
        using container_t                          = managed_array<value_t>;
        using size_type                            = std::size_t;
        using value_type                           = value_t;
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

        template <class Expr>
        thrust_container& operator*=(const Expr& expr)
            requires(detail::element_indexable<Expr>)
        {
            detail::apply_container_inplace(*this, expr, [](value_type& dst, auto&& factor) { dst *= factor; });
            return *this;
        }

        template <class Expr>
        thrust_container& operator/=(const Expr& expr)
            requires(detail::element_indexable<Expr>)
        {
            detail::apply_container_inplace(*this, expr, [](value_type& dst, auto&& factor) { dst /= factor; });
            return *this;
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

    template <class E>
    auto shape(const xt::xexpression<E>& expr, std::size_t axis)
    {
        const auto& derived = expr.derived_cast();
        return derived.shape()[static_cast<std::size_t>(axis)];
    }

    template <class E>
    auto shape(const xt::xexpression<E>& expr)
    {
        return expr.derived_cast().shape();
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

        template <std::size_t axis, class Expr>
        auto sum(const Expr& expr)
        {
            static_assert(axis < 2, "sum axis out of range for CUDA backend");
            using value_t = detail::element_value_t<Expr>;
            using sum_value_t = std::conditional_t<std::is_same_v<std::remove_cv_t<value_t>, bool>, std::size_t, value_t>;
            const auto shape = detail::infer_shape(expr);

            if (!shape.valid)
            {
                auto base = detail::evaluate_to_vector(expr);
                detail::vector_wrapper<sum_value_t> out;
                out.resize(base.size());
                for (std::size_t idx = 0; idx < base.size(); ++idx)
                {
                    out.set(idx, static_cast<sum_value_t>(base[idx]));
                }
                return out;
            }

            if constexpr (axis == 0)
            {
                detail::vector_wrapper<sum_value_t> out;
                out.resize(shape.length);
                for (std::size_t pos = 0; pos < shape.length; ++pos)
                {
                    sum_value_t acc{};
                    for (std::size_t item = 0; item < shape.items; ++item)
                    {
                        const std::size_t idx = item + pos * shape.items;
                        acc += static_cast<sum_value_t>(detail::element_at(expr, idx));
                    }
                    out.set(pos, acc);
                }
                return out;
            }
            else
            {
                detail::vector_wrapper<sum_value_t> out;
                out.resize(shape.items);
                for (std::size_t item = 0; item < shape.items; ++item)
                {
                    sum_value_t acc{};
                    for (std::size_t pos = 0; pos < shape.length; ++pos)
                    {
                        const std::size_t idx = item + pos * shape.items;
                        acc += static_cast<sum_value_t>(detail::element_at(expr, idx));
                    }
                    out.set(item, acc);
                }
                return out;
            }
        }

        template <std::size_t axis, std::size_t size, class Expr>
        auto all_true(const Expr& expr)
        {
            auto sums = sum<axis>(expr);
            detail::vector_wrapper<bool> mask;
            mask.resize(sums.size());
            if (sums.size() == 0)
            {
                return mask;
            }
            using entry_t = std::remove_cv_t<decltype(sums[0])>;
            const auto threshold = static_cast<entry_t>(size - 1);
            for (std::size_t idx = 0; idx < sums.size(); ++idx)
            {
                mask.set(idx, static_cast<bool>(sums[idx] > threshold));
            }
            return mask;
        }
    }


    namespace detail
    {
        template <class Mask>
        struct mask_negation
        {
            using value_type     = bool;
            using const_iterator = mask_const_iterator<mask_negation>;

            Mask mask;

            value_type operator[](std::size_t idx) const { return !static_cast<bool>(mask[idx]); }
            std::size_t size() const { return mask.size(); }

            const_iterator begin() const { return const_iterator{this, 0}; }
            const_iterator end() const { return const_iterator{this, size()}; }
            const_iterator cbegin() const { return begin(); }
            const_iterator cend() const { return end(); }
        };

        template <class Mask1, class Mask2, class BinaryOp>
        struct mask_binary
        {
            using value_type     = bool;
            using const_iterator = mask_const_iterator<mask_binary>;

            Mask1   lhs;
            Mask2   rhs;
            BinaryOp op;

            value_type operator[](std::size_t idx) const
            {
                const auto s1 = mask_size(lhs);
                const auto s2 = mask_size(rhs);
                const std::size_t lhs_idx = (s1 == 1) ? 0 : idx;
                const std::size_t rhs_idx = (s2 == 1) ? 0 : idx;

                if (lhs_idx >= s1 || rhs_idx >= s2)
                {
                    std::ostringstream oss;
                    oss << "mask_binary index out of range: idx=" << idx << " lhs_size=" << s1 << " rhs_size=" << s2;
                    throw std::out_of_range(oss.str());
                }

                return op(static_cast<bool>(lhs[lhs_idx]), static_cast<bool>(rhs[rhs_idx]));
            }

            std::size_t size() const
            {
                const auto s1 = mask_size(lhs);
                const auto s2 = mask_size(rhs);
                if (s1 == s2)
                {
                    return s1;
                }
                if (s1 == 1)
                {
                    return s2;
                }
                if (s2 == 1)
                {
                    return s1;
                }

                if (s1 != s2)
                {
                    std::ostringstream oss;
                    oss << "mask size mismatch: lhs=" << s1 << " rhs=" << s2;
                    throw std::runtime_error(oss.str());
                }
                return s1;
            }

            const_iterator begin() const { return const_iterator{this, 0}; }
            const_iterator end() const { return const_iterator{this, size()}; }
            const_iterator cbegin() const { return begin(); }
            const_iterator cend() const { return end(); }
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
        requires(detail::mask_indexable<std::decay_t<D>> && detail::thrust_sequence<std::decay_t<D>>)
    auto operator>(D&& v, double x)
    {
        struct mask_view
        {
            using value_type     = bool;
            using const_iterator = detail::mask_const_iterator<mask_view>;

            std::decay_t<D> ref;
            double          thresh;

            value_type operator[](std::size_t i) const
            {
                return static_cast<value_type>(detail::element_at(ref, i) > thresh);
            }

            std::size_t size() const
            {
                return detail::sequence_size(ref);
            }

            const_iterator begin() const { return const_iterator{this, 0}; }
            const_iterator end() const { return const_iterator{this, size()}; }
            const_iterator cbegin() const { return begin(); }
            const_iterator cend() const { return end(); }
        };
        return mask_view{detail::decay_copy(std::forward<D>(v)), x};
    }

    template <class D>
        requires(detail::mask_indexable<std::decay_t<D>> && detail::thrust_sequence<std::decay_t<D>>)
    auto operator<(D&& v, double x)
    {
        struct mask_view
        {
            using value_type     = bool;
            using const_iterator = detail::mask_const_iterator<mask_view>;

            std::decay_t<D> ref;
            double          thresh;

            value_type operator[](std::size_t i) const
            {
                return static_cast<value_type>(detail::element_at(ref, i) < thresh);
            }

            std::size_t size() const
            {
                return detail::sequence_size(ref);
            }

            const_iterator begin() const { return const_iterator{this, 0}; }
            const_iterator end() const { return const_iterator{this, size()}; }
            const_iterator cbegin() const { return begin(); }
            const_iterator cend() const { return end(); }
        };
        return mask_view{detail::decay_copy(std::forward<D>(v)), x};
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

        template <class Mask>
        std::size_t mask_size(const Mask& mask)
        {
            if constexpr (requires { mask.size(); })
            {
                return static_cast<std::size_t>(mask.size());
            }
            else if constexpr (sequence_accessible<Mask>)
            {
                return sequence_size(mask);
            }
            else
            {
                static_assert(dependent_false<Mask>::value, "Unsupported mask type for size()");
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
                return detail::sequence_size(expr);
            }

            auto operator[](std::size_t idx) const
            {
                return Op::apply(expr[idx]);
            }

            auto operator()(std::size_t idx) const
            {
                return (*this)[idx];
            }
        };

        template <class Op, class L, class R>
        struct thrust_expr_binary
        {
            L lhs;
            R rhs;

            std::size_t size() const
            {
                return std::min(detail::sequence_size(lhs), detail::sequence_size(rhs));
            }

            auto operator[](std::size_t idx) const
            {
                return Op::apply(lhs[idx], rhs[idx]);
            }

            auto operator()(std::size_t idx) const
            {
                return (*this)[idx];
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
                return detail::sequence_size(expr);
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

            auto operator()(std::size_t idx) const
            {
                return (*this)[idx];
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

            auto operator()(std::size_t idx) const
            {
                return (*this)[idx];
            }
        };

        template <class Expr>
        auto make_expr_slice(Expr expr, const range_t<long long>& range)
        {
            if (range.step <= 0)
            {
                throw std::invalid_argument("range.step must be positive");
            }
            const auto total_size = detail::sequence_size(expr);
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
        const auto mask_total = detail::mask_size(criteria);
        if constexpr (detail::sequence_accessible<std::decay_t<DST>>)
        {
            const auto dst_total = detail::sequence_size(dst);
            if (dst_total != mask_total)
            {
                std::ostringstream oss;
                oss << "apply_on_masked size mismatch: mask=" << mask_total << " dst=" << dst_total;
                throw std::runtime_error(oss.str());
            }
        }

        for (std::size_t i = 0; i < mask_total; ++i)
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
        const auto mask_total = detail::mask_size(criteria);
        for (std::size_t i = 0; i < mask_total; ++i)
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

    template <class L, class R>
        requires(detail::element_indexable<L> && detail::element_indexable<R> && (detail::thrust_sequence<L> || detail::thrust_sequence<R>))
    auto operator&(const L& lhs, const R& rhs)
    {
        return detail::thrust_expr_binary<detail::bit_and_op, std::decay_t<L>, std::decay_t<R>>{detail::decay_copy(lhs), detail::decay_copy(rhs)};
    }

    template <class Seq, class Scalar>
        requires(detail::element_indexable<Seq> && detail::scalar_like<Scalar> && detail::thrust_sequence<Seq>)
    auto operator&(const Seq& seq, Scalar s)
    {
        return detail::thrust_expr_scalar<Scalar, std::decay_t<Seq>, detail::bit_and_op, false>{s, detail::decay_copy(seq)};
    }

    template <class Scalar, class Seq>
        requires(detail::scalar_like<Scalar> && detail::element_indexable<Seq> && detail::thrust_sequence<Seq>)
    auto operator&(Scalar s, const Seq& seq)
    {
        return detail::thrust_expr_scalar<Scalar, std::decay_t<Seq>, detail::bit_and_op, true>{s, detail::decay_copy(seq)};
    }

    template <class L, class R>
        requires(detail::element_indexable<L> && detail::element_indexable<R> && (detail::thrust_sequence<L> || detail::thrust_sequence<R>))
    auto operator|(const L& lhs, const R& rhs)
    {
        return detail::thrust_expr_binary<detail::bit_or_op, std::decay_t<L>, std::decay_t<R>>{detail::decay_copy(lhs), detail::decay_copy(rhs)};
    }

    template <class Seq, class Scalar>
        requires(detail::element_indexable<Seq> && detail::scalar_like<Scalar> && detail::thrust_sequence<Seq>)
    auto operator|(const Seq& seq, Scalar s)
    {
        return detail::thrust_expr_scalar<Scalar, std::decay_t<Seq>, detail::bit_or_op, false>{s, detail::decay_copy(seq)};
    }

    template <class Scalar, class Seq>
        requires(detail::scalar_like<Scalar> && detail::element_indexable<Seq> && detail::thrust_sequence<Seq>)
    auto operator|(Scalar s, const Seq& seq)
    {
        return detail::thrust_expr_scalar<Scalar, std::decay_t<Seq>, detail::bit_or_op, true>{s, detail::decay_copy(seq)};
    }

    // Static array aliases (reuse xtensor static forms to avoid reimplementation)
    template <class value_type, std::size_t size, bool /*SOA*/>
    using thrust_static_array = xtensor_static_array<value_type, size>;

    template <class value_type, std::size_t size, bool /*SOA*/, bool can_collapse>
    using thrust_local_collapsable_array = CollapsableArray<thrust_static_array<value_type, size, false>, value_type, size, can_collapse>;
}

namespace xt
{
    template <class T>
    struct xiterable_inner_types<samurai::thrust_view_1d<T>>
    {
        using expression_type  = samurai::thrust_view_1d<T>;
        using inner_shape_type = typename expression_type::shape_type;
        using stepper          = xindexed_stepper<expression_type, false>;
        using const_stepper    = xindexed_stepper<expression_type, true>;
    };

    template <class T>
    struct xiterable_inner_types<samurai::thrust_view_2d<T>>
    {
        using expression_type  = samurai::thrust_view_2d<T>;
        using inner_shape_type = typename expression_type::shape_type;
        using stepper          = xindexed_stepper<expression_type, false>;
        using const_stepper    = xindexed_stepper<expression_type, true>;
    };
}
