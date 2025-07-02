// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#ifndef SAMURAI_DEBUG_ENABLED
#define SAMURAI_DEBUG_ENABLED 1  // Enabled by default, can be disabled in production
#endif

#if SAMURAI_DEBUG_ENABLED
#include <iostream>
#include <fstream>
#include <string>
#endif

#ifdef SAMURAI_WITH_MPI
#include <boost/mpi.hpp>
namespace mpi = boost::mpi;
#endif

namespace samurai
{
    namespace debug
    {
#if SAMURAI_DEBUG_ENABLED
        inline std::ofstream& get_debug_stream()
        {
            static std::ofstream stream;
            return stream;
        }
        
        inline void initialize_debug(const std::string& base_filename = "samurai_debug")
        {
            auto& stream = get_debug_stream();
            if (stream.is_open())
                return;
            
            std::string filename;
            #ifdef SAMURAI_WITH_MPI
            mpi::communicator world;
            filename = base_filename + "_rank" + std::to_string(world.rank()) + ".log";
            #else
            filename = base_filename + ".log";
            #endif
            
            stream.open(filename);
            if (!stream.is_open())
            {
                std::cerr << "Warning: Could not open debug file " << filename << std::endl;
            }
        }
        
        template<typename... Args>
        inline void debug_log(const std::string& scope, Args&&... args)
        {
            auto& stream = get_debug_stream();
            if (stream.is_open())
            {
                stream << "[" << scope << "] ";
                (stream << ... << std::forward<Args>(args));
                stream << std::endl;
                stream.flush();
            }
        }
#else
        // No-debug version: empty inline functions
        inline void initialize_debug(const std::string& = "samurai_debug") {}

        template<typename... Args>
        inline void debug_log(const std::string&, Args&&...) {}
#endif // SAMURAI_DEBUG_ENABLED
    }
} 