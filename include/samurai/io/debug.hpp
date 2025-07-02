// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#ifndef SAMURAI_DEBUG_ENABLED
#define SAMURAI_DEBUG_ENABLED 1  // Par défaut activé, peut être désactivé en production
#endif

#if SAMURAI_DEBUG_ENABLED
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
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
        class DebugLogger
        {
        private:
            std::ofstream file_stream_;
            std::string filename_;
            bool enabled_;

        public:
            DebugLogger() : enabled_(false) {}
            
            ~DebugLogger()
            {
                if (file_stream_.is_open())
                {
                    file_stream_.close();
                }
            }
            
            void initialize(const std::string& base_filename = "samurai_debug")
            {
                if (enabled_)
                    return;
                
                #ifdef SAMURAI_WITH_MPI
                mpi::communicator world;
                int rank = world.rank();
                filename_ = base_filename + "_rank" + std::to_string(rank) + ".log";
                #else
                filename_ = base_filename + ".log";
                #endif
                
                file_stream_.open(filename_);
                if (file_stream_.is_open())
                {
                    enabled_ = true;
                }
                else
                {
                    std::cerr << "Warning: Could not open debug file " << filename_ << std::endl;
                }
            }
            
            template<typename... Args>
            void log(const std::string& scope, Args&&... args)
            {
                if (!enabled_)
                    return;
                    
                if (file_stream_.is_open())
                {
                    file_stream_ << "[" << scope << "] ";
                    (log_impl(std::forward<Args>(args)), ...);
                    file_stream_ << std::endl;
                    file_stream_.flush();
                }
            }
            
        private:
            template<typename T>
            void log_impl(const T& value)
            {
                file_stream_ << value;
            }
            
            template<typename T>
            void log_impl(const T& value, const std::string& separator)
            {
                file_stream_ << value << separator;
            }
        };
        
        // Singleton instance
        inline DebugLogger& get_logger()
        {
            static DebugLogger logger;
            return logger;
        }
        
        // Convenience functions
        inline void initialize_debug(const std::string& base_filename = "samurai_debug")
        {
            get_logger().initialize(base_filename);
        }
        
        template<typename... Args>
        inline void debug_log(const std::string& scope, Args&&... args)
        {
            get_logger().log(scope, std::forward<Args>(args)...);
        }
#else
        // Version sans debug : fonctions vides inlinées
        inline void initialize_debug(const std::string& = "samurai_debug") {}

        template<typename... Args>
        inline void debug_log(const std::string&, Args&&...) {}
#endif // SAMURAI_DEBUG_ENABLED
    }

} 