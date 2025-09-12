#pragma once

#include <vector>
#include <cstddef>

// Forward declaration of the CUDA implementation
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
        // CSIR_Level_Host intersection(const CSIR_Level_Host& a, const CSIR_Level_Host& b);
        // ... other operations like difference, translate, expand, etc.

    } // namespace cuda
} // namespace csir