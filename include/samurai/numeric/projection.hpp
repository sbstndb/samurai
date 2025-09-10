// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "../operators_base.hpp"

namespace samurai
{
    /////////////////////////
    // projection operator //
    /////////////////////////

    template <std::size_t dim, class TInterval>
    class projection_op_ : public field_operator_base<dim, TInterval>
    {
      public:

        INIT_OPERATOR(projection_op_)

        template <class T1, class T2>
        inline void operator()(Dim<1>, T1& dest, const T2& src) const
        {
            times::expert_timers.start("numeric:projection:projection_op_1d");
            dest(level, i) = .5 * (src(level + 1, 2 * i) + src(level + 1, 2 * i + 1));
            times::expert_timers.stop("numeric:projection:projection_op_1d");
        }

        template <class T1, class T2>
        inline void operator()(Dim<2>, T1& dest, const T2& src) const
        {
            times::expert_timers.start("numeric:projection:projection_op_2d");
            dest(level, i, j) = .25
                              * (src(level + 1, 2 * i, 2 * j) + src(level + 1, 2 * i, 2 * j + 1) + src(level + 1, 2 * i + 1, 2 * j)
                                 + src(level + 1, 2 * i + 1, 2 * j + 1));
            times::expert_timers.stop("numeric:projection:projection_op_2d");
        }

        template <class T1, class T2>
        inline void operator()(Dim<3>, T1& dest, const T2& src) const
        {
            times::expert_timers.start("numeric:projection:projection_op_3d");
            dest(level, i, j, k) = .125
                                 * (src(level + 1, 2 * i, 2 * j, 2 * k) + src(level + 1, 2 * i + 1, 2 * j, 2 * k)
                                    + src(level + 1, 2 * i, 2 * j + 1, 2 * k) + src(level + 1, 2 * i + 1, 2 * j + 1, 2 * k)
                                    + src(level + 1, 2 * i, 2 * j, 2 * k + 1) + src(level + 1, 2 * i + 1, 2 * j, 2 * k + 1)
                                    + src(level + 1, 2 * i, 2 * j + 1, 2 * k + 1) + src(level + 1, 2 * i + 1, 2 * j + 1, 2 * k + 1));
            times::expert_timers.stop("numeric:projection:projection_op_3d");
        }
    };

    template <std::size_t dim, class TInterval>
    class variadic_projection_op_ : public field_operator_base<dim, TInterval>
    {
      public:

        INIT_OPERATOR(variadic_projection_op_)

        template <std::size_t d>
        inline void operator()(Dim<d>) const
        {
        }

        template <class Head, class... Tail>
        inline void operator()(Dim<1>, Head& source, Tail&... sources) const
        {
            times::expert_timers.start("numeric:projection:variadic_projection_op_1d");
            projection_op_<dim, interval_t>(level, i)(Dim<1>{}, source, source);
            this->operator()(Dim<1>{}, sources...);
            times::expert_timers.stop("numeric:projection:variadic_projection_op_1d");
        }

        template <class Head, class... Tail>
        inline void operator()(Dim<2>, Head& source, Tail&... sources) const
        {
            times::expert_timers.start("numeric:projection:variadic_projection_op_2d");
            projection_op_<dim, interval_t>(level, i, j)(Dim<2>{}, source, source);
            this->operator()(Dim<2>{}, sources...);
            times::expert_timers.stop("numeric:projection:variadic_projection_op_2d");
        }

        template <class Head, class... Tail>
        inline void operator()(Dim<3>, Head& source, Tail&... sources) const
        {
            times::expert_timers.start("numeric:projection:variadic_projection_op_3d");
            projection_op_<dim, interval_t>(level, i, j, k)(Dim<3>{}, source, source);
            this->operator()(Dim<3>{}, sources...);
            times::expert_timers.stop("numeric:projection:variadic_projection_op_3d");
        }
    };

    template <class T>
    inline auto projection(T&& field)
    {
        times::expert_timers.start("algorithm:projection:projection");
        auto result = make_field_operator_function<projection_op_>(std::forward<T>(field), std::forward<T>(field));
        times::expert_timers.stop("algorithm:projection:projection");
        return result;
    }

    template <class... T>
    inline auto variadic_projection(T&&... fields)
    {
        times::expert_timers.start("algorithm:projection:variadic_projection");
        auto result = make_field_operator_function<variadic_projection_op_>(std::forward<T>(fields)...);
        times::expert_timers.stop("algorithm:projection:variadic_projection");
        return result;
    }

    template <class T1, class T2>
    inline auto projection(T1&& field_dest, T2&& field_src)
    {
        times::expert_timers.start("algorithm:projection:projection_with_source");
        auto result = make_field_operator_function<projection_op_>(std::forward<T1>(field_dest), std::forward<T2>(field_src));
        times::expert_timers.stop("algorithm:projection:projection_with_source");
        return result;
    }
}
