#pragma once
#include <type_traits>
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
            auto expr       = x * x;
            norm_square     = 0.0;
            const std::size_t n = static_cast<std::size_t>(expr.size());
            for (std::size_t i = 0; i < n; ++i)
            {
                if constexpr (requires { expr[i]; })
                {
                    norm_square += static_cast<double>(expr[i]);
                }
                else
                {
                    norm_square += static_cast<double>(expr(i));
                }
            }
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
        return L2_error<false, Field, Func>(approximate, std::forward<Func>(exact));
    }

    template <std::size_t order>
    double compute_error_bound_hidden_constant(double h, double error)
    {
        return error / std::pow(h, order);
    }

    template <std::size_t order>
    double theoretical_error_bound(double hidden_constant, double h)
    {
        return hidden_constant * std::pow(h, order);
    }

    inline double convergence_order(double h1, double error1, double h2, double error2)
    {
        return log(error2 / error1) / log(h2 / h1);
    }
}
