// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause
#pragma once

#include <fmt/format.h>
#include <fmt/printf.h>
#include <functional>
// <string_view> n'est plus nécessaire après simplification

#ifdef SAMURAI_WITH_MPI
#include <boost/mpi.hpp>
namespace mpi = boost::mpi;
#endif

namespace samurai
{
    namespace output
    {
        // Fonction utilitaire minimaliste : rang courant du processus.
        inline int rank()
        {
#ifdef SAMURAI_WITH_MPI
            mpi::communicator world;
            return world.rank();
#else
            return 0;
#endif
        }

        // ---------------------- Impression standard ----------------------

        template <typename... Args>
        void print(int target_rank, const char* format, Args&&... args)
        {
#ifdef SAMURAI_WITH_MPI
            if (rank() != target_rank)
                return;
#endif
            fmt::print(fmt::runtime(format), std::forward<Args>(args)...);
        }

        template <typename... Args>
        void print(const char* format, Args&&... args)
        {
            print(0, format, std::forward<Args>(args)...);
        }

        template <typename... Args>
        void print_all(const char* format, Args&&... args)
        {
#ifdef SAMURAI_WITH_MPI
            fmt::print("[Rang {}] ", rank());
#endif
            fmt::print(fmt::runtime(format), std::forward<Args>(args)...);
        }

        template <typename... Args>
        void print_error(const char* format, Args&&... args)
        {
#ifdef SAMURAI_WITH_MPI
            fmt::print(stderr, "[Rang {}] ERREUR: ", rank());
#else
            fmt::print(stderr, "ERREUR: ");
#endif
            fmt::print(stderr, fmt::runtime(format), std::forward<Args>(args)...);
        }

        // Forward declaration needed for two-phase lookup in templates using print_reduce below.
        template <typename T, typename Op>
        void print_reduce(const T& local_value, Op op, const char* format);

        template <typename T>
        void print_max(const T& local_value, const char* format = "Max value: {}\n")
        {
#ifdef SAMURAI_WITH_MPI
            print_reduce(local_value, mpi::maximum<T>(), format);
#else
            fmt::print(fmt::runtime(format), local_value);
#endif
        }

        template <typename T>
        void print_min(const T& local_value, const char* format = "Min value: {}\n")
        {
#ifdef SAMURAI_WITH_MPI
            print_reduce(local_value, mpi::minimum<T>(), format);
#else
            fmt::print(fmt::runtime(format), local_value);
#endif
        }

        template <typename T>
        void print_sum(const T& local_value, const char* format = "Somme globale: {}\n")
        {
#ifdef SAMURAI_WITH_MPI
            print_reduce(local_value, std::plus<T>(), format);
#else
            fmt::print(fmt::runtime(format), local_value);
#endif
        }

        template <typename T, typename Op>
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
