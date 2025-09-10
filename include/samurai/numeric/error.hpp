#pragma once
#include "../timers.hpp"
#include "gauss_legendre.hpp"

namespace samurai
{
    template <class scalar_or_vectorview>
    double __square(scalar_or_vectorview x)
    {
        double norm_square;
        if constexpr (std::is_same_v<scalar_or_vectorview, double>) // scalar
        {
            norm_square = x * x;
        }
        else // vector view
        {
            using namespace samurai::math;
            norm_square = sum(x * x);
        }
        return norm_square;
    }

    /**
     * Computes the L2-error with respect to an exact solution.
     * @tparam relative_error: if true, compute the relative error instead of the absolute one.
     */
    template <bool relative_error, class Field, class Func>
    double L2_error(Field& approximate, Func&& exact)
    {
        times::expert_timers.start("numeric:L2_error");
        times::timers.start("error computation");

        // In FV, we want only 1 quadrature point.
        // This is equivalent to
        //       error += pow(exact(cell.center()) - approximate(cell.index), 2) * cell.length^dim;
        GaussLegendre<0> gl;

        double error_norm    = 0;
        double solution_norm = 0;
        for_each_cell(approximate.mesh(),
                      [&](const auto& cell)
                      {
                          error_norm += gl.quadrature<1>(cell,
                                                         [&](const auto& point)
                                                         {
                                                             auto e = exact(point) - approximate[cell];
                                                             return __square(e);
                                                         });
                          if constexpr (relative_error)
                          {
                              solution_norm += gl.quadrature<1>(cell,
                                                                [&](const auto& point)
                                                                {
                                                                    auto v = exact(point);
                                                                    return __square(v);
                                                                });
                          }
                      });

        error_norm    = sqrt(error_norm);
        solution_norm = sqrt(solution_norm);

        times::timers.stop("error computation");
        times::expert_timers.stop("numeric:L2_error");

        if constexpr (relative_error)
        {
            return error_norm / solution_norm;
        }
        else
        {
            return error_norm;
        }
    }

    template <class Field, class Func>
    double L2_error(Field& approximate, Func&& exact)
    {
        times::expert_timers.start("numeric:L2_error_default");
        auto result = L2_error<false, Field, Func>(approximate, std::forward<Func>(exact));
        times::expert_timers.stop("numeric:L2_error_default");
        return result;
    }

    template <std::size_t order>
    double compute_error_bound_hidden_constant(double h, double error)
    {
        times::expert_timers.start("numeric:compute_error_bound_hidden_constant");
        auto result = error / std::pow(h, order);
        times::expert_timers.stop("numeric:compute_error_bound_hidden_constant");
        return result;
    }

    template <std::size_t order>
    double theoretical_error_bound(double hidden_constant, double h)
    {
        times::expert_timers.start("numeric:theoretical_error_bound");
        auto result = hidden_constant * std::pow(h, order);
        times::expert_timers.stop("numeric:theoretical_error_bound");
        return result;
    }

    inline double convergence_order(double h1, double error1, double h2, double error2)
    {
        times::expert_timers.start("numeric:convergence_order");
        auto result = log(error2 / error1) / log(h2 / h1);
        times::expert_timers.stop("numeric:convergence_order");
        return result;
    }
}
