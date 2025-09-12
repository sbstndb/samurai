#pragma once

#include <vector>
#include <cstddef>
#include <array>

namespace csir
{
    namespace cuda
    {
        // Base interval type, same as CPU version
        struct Interval
        {
            int start, end;
        };

        // Device-side representation of a 2D CSIR level.
        // This struct contains pointers to GPU memory.
        struct CSIR_Level_Device
        {
            int* y_coords;
            std::size_t* intervals_ptr;
            Interval* intervals;

            std::size_t y_coords_count;
            std::size_t intervals_ptr_count;
            std::size_t intervals_count;
            std::size_t level;

            // Utility to free device memory
            void free();
        };

        // Host-side representation of a 2D CSIR level.
        // This struct holds the data on the host and provides
        // methods to copy data to and from the GPU.
        struct CSIR_Level_Host
        {
            std::vector<int> y_coords;
            std::vector<std::size_t> intervals_ptr;
            std::vector<Interval> intervals;
            std::size_t level = 0;

            bool empty() const { return intervals.empty(); }

            // Methods for Host <-> Device communication
            CSIR_Level_Device to_device() const;
            void from_device(const CSIR_Level_Device& device_level);
        };

        // --- Function declarations for CUDA operations ---
        // These functions will be called from the host code.

        CSIR_Level_Host union_(const CSIR_Level_Host& a, const CSIR_Level_Host& b);
        CSIR_Level_Host intersection(const CSIR_Level_Host& a, const CSIR_Level_Host& b);
        CSIR_Level_Host difference(const CSIR_Level_Host& a, const CSIR_Level_Host& b);

        CSIR_Level_Host translate(const CSIR_Level_Host& src, int dx, int dy);
        CSIR_Level_Host project_to_level(const CSIR_Level_Host& source, std::size_t target_level);
        CSIR_Level_Host contract(const CSIR_Level_Host& set, std::size_t width, const bool dir_mask[2]);
        CSIR_Level_Host contract(const CSIR_Level_Host& set, std::size_t width);
        CSIR_Level_Host expand(const CSIR_Level_Host& set, std::size_t width, const bool dir_mask[2]);
        CSIR_Level_Host expand(const CSIR_Level_Host& set, std::size_t width);

        // --- 1D Structures ---
        struct CSIR_Level_1D_Device
        {
            Interval* intervals;
            std::size_t intervals_count;
            std::size_t level;

            void free();
        };

        struct CSIR_Level_1D_Host
        {
            std::vector<Interval> intervals;
            std::size_t level = 0;

            bool empty() const { return intervals.empty(); }

            CSIR_Level_1D_Device to_device() const;
            void from_device(const CSIR_Level_1D_Device& device_level);
        };

        // --- 1D Operations ---
        CSIR_Level_1D_Host union_1d(const CSIR_Level_1D_Host& a, const CSIR_Level_1D_Host& b);
        CSIR_Level_1D_Host intersection_1d(const CSIR_Level_1D_Host& a, const CSIR_Level_1D_Host& b);
        CSIR_Level_1D_Host difference_1d(const CSIR_Level_1D_Host& a, const CSIR_Level_1D_Host& b);
        CSIR_Level_1D_Host translate_1d(const CSIR_Level_1D_Host& src, int dx);
        CSIR_Level_1D_Host contract_1d(const CSIR_Level_1D_Host& set, std::size_t width);
        CSIR_Level_1D_Host expand_1d(const CSIR_Level_1D_Host& set, std::size_t width);
        CSIR_Level_1D_Host project_to_level_1d(const CSIR_Level_1D_Host& source, std::size_t target_level);

    } // namespace cuda
} // namespace csir