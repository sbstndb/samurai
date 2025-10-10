// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <algorithm>
#include <array>
#include <type_traits>

#include <thrust/host_vector.h>

#include <xtensor/xadapt.hpp>

#include "../xtensor/xtensor.hpp"
#include "../xtensor/xtensor_static.hpp"

namespace samurai
{
    namespace detail
    {
        struct thrust_xtensor_tag
        {
        };

        struct thrust_host_vector_tag
        {
        };

        template <class T>
        class host_vector_container : public thrust::host_vector<T>
        {
          public:

            using base_type = thrust::host_vector<T>;
            using base_type::base_type;

            void fill(const T& value)
            {
                std::fill(this->begin(), this->end(), value);
            }
        };

        template <class T>
        struct is_host_vector_container : std::false_type
        {
        };

        template <class T>
        struct is_host_vector_container<host_vector_container<T>> : std::true_type
        {
        };

        template <class T>
        inline constexpr bool is_host_vector_container_v = is_host_vector_container<std::remove_cvref_t<T>>::value;

        template <class Vector>
        inline auto adapt_host_vector(Vector& vec)
        {
            using size_type = typename Vector::size_type;
            auto shape      = std::array<std::size_t, 1>{static_cast<std::size_t>(vec.size())};
            return xt::adapt(vec.data(), static_cast<size_type>(vec.size()), xt::no_ownership(), shape);
        }

        template <class Vector>
        inline auto adapt_host_vector(const Vector& vec)
        {
            using size_type = typename Vector::size_type;
            auto shape      = std::array<std::size_t, 1>{static_cast<std::size_t>(vec.size())};
            return xt::adapt(vec.data(), static_cast<size_type>(vec.size()), xt::no_ownership(), shape);
        }
    }
    //----------------------------------------------------------------------------//
    // Temporary Thrust backend scaffolding.                                      //
    //                                                                            //
    // Progressive migration towards Thrust-backed containers will reuse the     //
    // existing xtensor-based implementation until dedicated Thrust data handles //
    // are introduced.                                                            //
    //----------------------------------------------------------------------------//

    template <class value_type, std::size_t size = 1, bool SOA = false, bool can_collapse = true>
    class thrust_container
    {
      public:

        using storage_tag = detail::thrust_xtensor_tag;
        using backend_t   = xtensor_container<value_type, size, SOA, can_collapse>;
        using size_type   = typename backend_t::size_type;
        using container_t = typename backend_t::container_t;
        static constexpr auto static_layout = backend_t::static_layout;
        static constexpr bool uses_xtensor_backend = true;

        thrust_container() = default;

        explicit thrust_container(std::size_t dynamic_size)
            : m_backend(dynamic_size)
        {
        }

        const backend_t& backend() const
        {
            return m_backend;
        }

        backend_t& backend()
        {
            return m_backend;
        }

        const auto& data() const
        {
            return m_backend.data();
        }

        auto& data()
        {
            return m_backend.data();
        }

        value_type& operator[](size_type index)
        {
            return m_backend[index];
        }

        const value_type& operator[](size_type index) const
        {
            return m_backend[index];
        }

        size_type value_count() const
        {
            return m_backend.value_count();
        }

        void fill(const value_type& value)
        {
            m_backend.fill(value);
        }

        void resize(std::size_t dynamic_size)
        {
            m_backend.resize(dynamic_size);
        }

      private:

        backend_t m_backend;
    };

    template <class value_type, std::size_t size, bool SOA = false, bool can_collapse = true>
    using thrust_collapsable_static_array = xtensor_collapsable_static_array<value_type, size, can_collapse>;

    template <class value_type, std::size_t size, bool SOA, bool can_collapse>
    auto& view_backend(thrust_container<value_type, size, SOA, can_collapse>& container)
    {
        return container.backend();
    }

    template <class value_type, std::size_t size, bool SOA, bool can_collapse>
    const auto& view_backend(const thrust_container<value_type, size, SOA, can_collapse>& container)
    {
        return container.backend();
    }

    template <class value_t, std::size_t size, bool SOA, bool can_collapse>
    auto view(const thrust_container<value_t, size, SOA, can_collapse>& container, const range_t<long long>& range)
    {
        return view(view_backend(container), range);
    }

    template <class value_t, std::size_t size, bool SOA, bool can_collapse>
    auto view(thrust_container<value_t, size, SOA, can_collapse>& container, const range_t<long long>& range)
    {
        return view(view_backend(container), range);
    }

    template <class value_t, std::size_t size, bool SOA, bool can_collapse>
    auto view(const thrust_container<value_t, size, SOA, can_collapse>& container,
              const range_t<std::size_t>& range_item,
              const range_t<long long>& range)
    {
        return view(view_backend(container), range_item, range);
    }

    template <class value_t, std::size_t size, bool SOA, bool can_collapse>
    auto view(thrust_container<value_t, size, SOA, can_collapse>& container,
              const range_t<std::size_t>& range_item,
              const range_t<long long>& range)
    {
        return view(view_backend(container), range_item, range);
    }

    template <class value_t, std::size_t size, bool SOA, bool can_collapse>
    auto view(const thrust_container<value_t, size, SOA, can_collapse>& container, std::size_t item, const range_t<long long>& range)
    {
        return view(view_backend(container), item, range);
    }

    template <class value_t, std::size_t size, bool SOA, bool can_collapse>
    auto view(thrust_container<value_t, size, SOA, can_collapse>& container, std::size_t item, const range_t<long long>& range)
    {
        return view(view_backend(container), item, range);
    }

    template <class value_t, std::size_t size, bool SOA, bool can_collapse>
    auto view(const thrust_container<value_t, size, SOA, can_collapse>& container, std::size_t index)
    {
        return view(view_backend(container), index);
    }

    template <class value_t, std::size_t size, bool SOA, bool can_collapse>
    auto view(thrust_container<value_t, size, SOA, can_collapse>& container, std::size_t index)
    {
        return view(view_backend(container), index);
    }
}
