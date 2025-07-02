// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause
#pragma once

#include <fmt/format.h>
#include <fmt/printf.h>
#include <functional>
#include <string_view>

#ifdef SAMURAI_WITH_MPI
#include <boost/mpi.hpp>
namespace mpi = boost::mpi;
#endif

namespace samurai
{
    namespace output
    {
        // Fonctions utilitaires internes (préfixées d'un '_' pour ne pas
        // polluer l'API publique).

        template<class F>
        inline void _on_rank(int target_rank, F&& f)
        {
#ifdef SAMURAI_WITH_MPI
            mpi::communicator world;
            if (world.rank() == target_rank)
            {
                std::forward<F>(f)();
            }
#else
            std::forward<F>(f)();
#endif
        }

        template<class F>
        inline void _with_rank(F&& f)
        {
#ifdef SAMURAI_WITH_MPI
            mpi::communicator world;
            std::forward<F>(f)(world.rank());
#else
            std::forward<F>(f)(0);
#endif
        }

        template<typename... Args>
        inline void _print_out(std::string_view prefix,
                               const char* format,
                               Args&&... args)
        {
            if (prefix.empty())
            {
                fmt::print(fmt::runtime(format), std::forward<Args>(args)...);
            }
            else
            {
                fmt::print("{}{}", prefix,
                           fmt::format(fmt::runtime(format),
                                       std::forward<Args>(args)...));
            }
        }

        template<typename... Args>
        inline void _print_err(std::string_view prefix,
                               const char* format,
                               Args&&... args)
        {
            if (prefix.empty())
            {
                fmt::print(stderr, fmt::runtime(format), std::forward<Args>(args)...);
            }
            else
            {
                fmt::print(stderr, "{}{}", prefix,
                           fmt::format(fmt::runtime(format),
                                       std::forward<Args>(args)...));
            }
        }

        template<typename... Args>
        void print(int target_rank, const char* format, Args&&... args)
        {
            _on_rank(target_rank, [&] {
                _print_out("", format, std::forward<Args>(args)...);
            });
        }


        template<typename... Args>
        void print(const char* format, Args&&... args)
        {
            print(0, format, std::forward<Args>(args)...);
        }


        template<typename... Args>
        void print_all(const char* format, Args&&... args)
        {
            _with_rank([&](int rank) {
                _print_out(fmt::format("[Rang {}] ", rank),
                           format,
                           std::forward<Args>(args)...);
            });
        }


        template<typename... Args>
        void print_error(const char* format, Args&&... args)
        {
            _with_rank([&](int rank) {
                _print_err(fmt::format("[Rang {}] ERREUR: ", rank),
                           format,
                           std::forward<Args>(args)...);
            });
        }

        // Forward declaration needed for two-phase lookup in templates using print_reduce below.
        template<typename T, typename Op>
        void print_reduce(const T& local_value, Op op, const char* format);

        template<typename T>
        void print_max(const T& local_value, const char* format = "Max value: {}\n")
        {
#ifdef SAMURAI_WITH_MPI
            print_reduce(local_value, mpi::maximum<T>(), format);
#else
            fmt::print(fmt::runtime(format), local_value);
#endif
        }

        template<typename T>
        void print_min(const T& local_value, const char* format = "Min value: {}\n")
        {
#ifdef SAMURAI_WITH_MPI
            print_reduce(local_value, mpi::minimum<T>(), format);
#else
            fmt::print(fmt::runtime(format), local_value);
#endif
        }

        template<typename T>
        void print_sum(const T& local_value, const char* format = "Somme globale: {}\n")
        {
#ifdef SAMURAI_WITH_MPI
            print_reduce(local_value, std::plus<T>(), format);
#else
            fmt::print(fmt::runtime(format), local_value);
#endif
        }

        template<typename T, typename Op>
        void print_reduce(const T& local_value, Op op, const char* format)
        {
#ifdef SAMURAI_WITH_MPI
            mpi::communicator world;
            T global_val;
            mpi::all_reduce(world, local_value, global_val, op);
            if (world.rank() == 0)
            {
                fmt::print(fmt::runtime(format), global_val);
            }
#else
            fmt::print(fmt::runtime(format), local_value);
#endif
        }
    }
} 