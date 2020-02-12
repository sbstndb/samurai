#pragma once

#include "xtl/xtype_traits.hpp"

#include "xtensor/xexpression.hpp"
#include "xtensor/xmath.hpp"

#include "cell.hpp"
#include "interval.hpp"
#include "utils.hpp"

namespace mure
{
    struct field_expression_tag
    {
    };

    template<class E>
    struct is_field_expression
        : std::is_same<xt::xexpression_tag_t<E>, field_expression_tag>
    {
    };

    template<class... E>
    struct field_comparable : xtl::conjunction<is_field_expression<E>...>
    {
    };

    template<class D>
    class field_expression : public xt::xexpression<D> {
      public:
        using expression_tag = field_expression_tag;
    };

    namespace detail
    {
        template<class E, class enable = void>
        struct xview_type_impl
        {
            using type = E;
        };

        template<class E>
        struct xview_type_impl<E,
                               std::enable_if_t<is_field_expression<E>::value>>
        {
            using type = typename E::view_type;
        };
    }

    template<class E>
    using xview_type = detail::xview_type_impl<E>;

    template<class E>
    using xview_type_t = typename xview_type<E>::type;

    template<class F, class... CT>
    class field_function : public field_expression<field_function<F, CT...>> {
      public:
        using self_type = field_function<F, CT...>;
        using functor_type = std::remove_reference_t<F>;

        static constexpr std::size_t dim = detail::compute_dim<CT...>();

        using interval_t = Interval<int>;

        using expression_tag = field_expression_tag;

        template<
            class Func, class... CTA,
            class U = std::enable_if<!std::is_base_of<Func, self_type>::value>>
        field_function(Func &&f, CTA &&... e) noexcept;

        template<class... T>
        auto operator()(const std::size_t &level, const interval_t &interval,
                        const T &... index) const
        {
            auto expr = evaluate(std::make_index_sequence<sizeof...(CT)>(),
                                 level, interval, index...);
            return expr;
        }

        template<class coord_index_t, std::size_t dim>
        auto operator()(const Cell<coord_index_t, dim> &cell) const
        {
            return evaluate(std::make_index_sequence<sizeof...(CT)>(), cell);
        }

        template<std::size_t... I, class... T>
        auto evaluate(std::index_sequence<I...>, T &&... t) const
        {
            return m_f(
                std::get<I>(m_e).template operator()(std::forward<T>(t)...)...);
        }

      private:
        std::tuple<CT...> m_e;
        functor_type m_f;
    };

    template<class F, class... CT>
    template<class Func, class... CTA, class>
    inline field_function<F, CT...>::field_function(Func &&f,
                                                    CTA &&... e) noexcept
        : m_e(std::forward<CTA>(e)...), m_f(std::forward<Func>(f))
    {}

    template<class F, class... E>
    inline auto make_field_function(E &&... e) noexcept
    {
        using type = field_function<F, E...>;
        return type(F(), std::forward<E>(e)...);
    }
}

namespace xt
{
    namespace detail
    {
        template<class F, class... E>
        struct select_xfunction_expression<mure::field_expression_tag, F, E...>
        {
            using type = mure::field_function<F, E...>;
        };
    }
}

namespace mure
{
    using xt::operator+;
    using xt::operator-;
    using xt::operator*;
    using xt::operator/;
    using xt::operator%;
}