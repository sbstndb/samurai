#include "csir_cuda.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <vector>

namespace csir
{
    namespace cuda
    {
        // --- Error Handling & Memory Management ---
        void check_cuda_error(cudaError_t err, const char* file, int line) {
            if (err != cudaSuccess) {
                std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << file << ":" << line << std::endl;
                exit(EXIT_FAILURE);
            }
        }
        #define CUDA_CHECK(err) check_cuda_error(err, __FILE__, __LINE__)

        void CSIR_Level_Device::free() {
            if (y_coords) CUDA_CHECK(cudaFree(y_coords));
            if (intervals_ptr) CUDA_CHECK(cudaFree(intervals_ptr));
            if (intervals) CUDA_CHECK(cudaFree(intervals));
            y_coords = nullptr;
            intervals_ptr = nullptr;
            intervals = nullptr;
        }

        CSIR_Level_Device CSIR_Level_Host::to_device() const {
            CSIR_Level_Device dev_level = {};
            dev_level.level = level;
            
            if (!y_coords.empty()) {
                dev_level.y_coords_count = y_coords.size();
                CUDA_CHECK(cudaMalloc(&dev_level.y_coords, y_coords.size() * sizeof(int)));
                CUDA_CHECK(cudaMemcpy(dev_level.y_coords, y_coords.data(), y_coords.size() * sizeof(int), cudaMemcpyHostToDevice));
            }
            if (!intervals_ptr.empty()) {
                dev_level.intervals_ptr_count = intervals_ptr.size();
                CUDA_CHECK(cudaMalloc(&dev_level.intervals_ptr, intervals_ptr.size() * sizeof(std::size_t)));
                CUDA_CHECK(cudaMemcpy(dev_level.intervals_ptr, intervals_ptr.data(), intervals_ptr.size() * sizeof(std::size_t), cudaMemcpyHostToDevice));
            }
            if (!intervals.empty()) {
                dev_level.intervals_count = intervals.size();
                CUDA_CHECK(cudaMalloc(&dev_level.intervals, intervals.size() * sizeof(Interval)));
                CUDA_CHECK(cudaMemcpy(dev_level.intervals, intervals.data(), intervals.size() * sizeof(Interval), cudaMemcpyHostToDevice));
            }
            return dev_level;
        }

        void CSIR_Level_Host::from_device(const CSIR_Level_Device& dev_level) {
            level = dev_level.level;
            y_coords.resize(dev_level.y_coords_count);
            intervals_ptr.resize(dev_level.intervals_ptr_count);
            intervals.resize(dev_level.intervals_count);

            if (dev_level.y_coords_count > 0)
                CUDA_CHECK(cudaMemcpy(y_coords.data(), dev_level.y_coords, dev_level.y_coords_count * sizeof(int), cudaMemcpyDeviceToHost));
            if (dev_level.intervals_ptr_count > 0)
                CUDA_CHECK(cudaMemcpy(intervals_ptr.data(), dev_level.intervals_ptr, dev_level.intervals_ptr_count * sizeof(std::size_t), cudaMemcpyDeviceToHost));
            if (dev_level.intervals_count > 0)
                CUDA_CHECK(cudaMemcpy(intervals.data(), dev_level.intervals, dev_level.intervals_count * sizeof(Interval), cudaMemcpyDeviceToHost));
        }

        // --- Device-side Helpers & Kernels ---

        __device__ int find_row_index(const int* y_coords, int count, int y) {
            int low = 0, high = count;
            while (low < high) {
                int mid = low + (high - low) / 2;
                if (y_coords[mid] < y) {
                    low = mid + 1;
                } else {
                    high = mid;
                }
            }
            if (low < count && y_coords[low] == y) {
                return low;
            }
            return -1;
        }

        __device__ void union_1d_device(
            const Interval* a_begin, const Interval* a_end,
            const Interval* b_begin, const Interval* b_end,
            Interval* result, std::size_t& result_count, bool compute)
        {
            result_count = 0;
            if (a_begin == a_end && b_begin == b_end) return;

            const Interval* a_it = a_begin;
            const Interval* b_it = b_begin;

            if (a_begin == a_end) {
                result_count = (b_end - b_begin);
                if (compute) for (int i=0; i<result_count; ++i) result[i] = b_begin[i];
                return;
            }
            if (b_begin == b_end) {
                result_count = (a_end - a_begin);
                if (compute) for (int i=0; i<result_count; ++i) result[i] = a_begin[i];
                return;
            }

            Interval current_interval;
            if (a_it->start < b_it->start) {
                current_interval = *a_it++;
            } else {
                current_interval = *b_it++;
            }

            while (a_it != a_end || b_it != b_end) {
                const Interval* next_interval = nullptr;
                if (a_it != a_end && (b_it == b_end || a_it->start < b_it->start)) {
                    next_interval = a_it++;
                } else if (b_it != b_end) {
                    next_interval = b_it++;
                }

                if (next_interval->start <= current_interval.end) {
                    current_interval.end = max(current_interval.end, next_interval->end);
                } else {
                    if (compute) result[result_count] = current_interval;
                    result_count++;
                    current_interval = *next_interval;
                }
            }
            if (compute) result[result_count] = current_interval;
            result_count++;
        }

        // --- Device-side 1D Intersection ---
        __device__ void intersection_1d_device(
            const Interval* a_begin, const Interval* a_end,
            const Interval* b_begin, const Interval* b_end,
            Interval* result, std::size_t& result_count, bool compute)
        {
            result_count = 0;
            if (a_begin == a_end || b_begin == b_end) return;

            auto a_it = a_begin;
            auto b_it = b_begin;

            while (a_it != a_end && b_it != b_end)
            {
                auto max_start = max(a_it->start, b_it->start);
                auto min_end = min(a_it->end, b_it->end);
                if (max_start < min_end) { 
                    if (compute) result[result_count] = {max_start, min_end};
                    result_count++;
                }
                if (a_it->end < b_it->end) { ++a_it; } else { ++b_it; }
            }
        }

        // --- Generic 2D Kernel (for Union and Intersection) ---
        __global__ void set_op_2d_kernel(
            CSIR_Level_Device a, CSIR_Level_Device b,
            const int* unified_y, int unified_y_count,
            CSIR_Level_Device result, 
            const std::size_t* row_output_offsets,
            std::size_t* row_output_counts,
            bool compute,
            bool is_union) // true for union, false for intersection
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= unified_y_count) return;

            int y = unified_y[idx];
            
            int a_row_idx = a.y_coords ? find_row_index(a.y_coords, a.y_coords_count, y) : -1;
            int b_row_idx = b.y_coords ? find_row_index(b.y_coords, b.y_coords_count, y) : -1;

            const Interval* a_begin = (a_row_idx != -1) ? &a.intervals[a.intervals_ptr[a_row_idx]] : nullptr;
            const Interval* a_end = (a_row_idx != -1) ? &a.intervals[a.intervals_ptr[a_row_idx+1]] : a_begin;
            
            const Interval* b_begin = (b_row_idx != -1) ? &b.intervals[b.intervals_ptr[b_row_idx]] : nullptr;
            const Interval* b_end = (b_row_idx != -1) ? &b.intervals[b.intervals_ptr[b_row_idx+1]] : b_begin;

            std::size_t count = 0;
            Interval* result_ptr = compute ? &result.intervals[row_output_offsets[idx]] : nullptr;
            
            if (is_union) {
                union_1d_device(a_begin, a_end, b_begin, b_end, result_ptr, count, compute);
            } else {
                intersection_1d_device(a_begin, a_end, b_begin, b_end, result_ptr, count, compute);
            }
            
            if (!compute) {
                row_output_counts[idx] = count;
            }
        }

        // --- Host-side 2D Union (Refactored) ---
        CSIR_Level_Host union_(const CSIR_Level_Host& a, const CSIR_Level_Host& b)
        {
            if (a.level != b.level) return CSIR_Level_Host();
            if (a.empty()) return b;
            if (b.empty()) return a;

            // 1. Create unified Y-coordinate list on host
            std::vector<int> unified_y = a.y_coords;
            unified_y.insert(unified_y.end(), b.y_coords.begin(), b.y_coords.end());
            std::sort(unified_y.begin(), unified_y.end());
            unified_y.erase(std::unique(unified_y.begin(), unified_y.end()), unified_y.end());

            // 2. Copy inputs to device
            CSIR_Level_Device d_a = a.to_device();
            CSIR_Level_Device d_b = b.to_device();
            int* d_unified_y;
            CUDA_CHECK(cudaMalloc(&d_unified_y, unified_y.size() * sizeof(int)));
            CUDA_CHECK(cudaMemcpy(d_unified_y, unified_y.data(), unified_y.size() * sizeof(int), cudaMemcpyHostToDevice));

            // 3. Sizing phase
            std::size_t* d_row_output_counts;
            CUDA_CHECK(cudaMalloc(&d_row_output_counts, unified_y.size() * sizeof(std::size_t)));
            
            int threads_per_block = 256;
            int blocks = (unified_y.size() + threads_per_block - 1) / threads_per_block;
            set_op_2d_kernel<<<blocks, threads_per_block>>>(d_a, d_b, d_unified_y, unified_y.size(), {}, nullptr, d_row_output_counts, false, true);
            CUDA_CHECK(cudaGetLastError());

            // 4. Prefix sum on host
            std::vector<std::size_t> h_row_output_counts(unified_y.size());
            CUDA_CHECK(cudaMemcpy(h_row_output_counts.data(), d_row_output_counts, unified_y.size() * sizeof(std::size_t), cudaMemcpyDeviceToHost));

            CSIR_Level_Host result_host;
            result_host.level = a.level;
            result_host.intervals_ptr.push_back(0);
            std::size_t total_intervals = 0;

            std::vector<std::size_t> h_row_output_offsets;
            h_row_output_offsets.reserve(unified_y.size());

            for(size_t i = 0; i < unified_y.size(); ++i) {
                if (h_row_output_counts[i] > 0) {
                    result_host.y_coords.push_back(unified_y[i]);
                    h_row_output_offsets.push_back(total_intervals);
                    total_intervals += h_row_output_counts[i];
                    result_host.intervals_ptr.push_back(total_intervals);
                }
            }
            
            if (total_intervals == 0) { // Handle case where result is empty
                d_a.free(); d_b.free(); CUDA_CHECK(cudaFree(d_unified_y)); CUDA_CHECK(cudaFree(d_row_output_counts));
                return result_host;
            }

            // 5. Compute phase
            CSIR_Level_Device d_result = {};
            d_result.level = result_host.level;
            d_result.y_coords_count = result_host.y_coords.size();
            d_result.intervals_ptr_count = result_host.intervals_ptr.size();
            d_result.intervals_count = total_intervals;

            CUDA_CHECK(cudaMalloc(&d_result.y_coords, d_result.y_coords_count * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_result.intervals_ptr, d_result.intervals_ptr_count * sizeof(std::size_t)));
            CUDA_CHECK(cudaMalloc(&d_result.intervals, d_result.intervals_count * sizeof(Interval)));
            
            CUDA_CHECK(cudaMemcpy(d_result.y_coords, result_host.y_coords.data(), d_result.y_coords_count * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_result.intervals_ptr, result_host.intervals_ptr.data(), d_result.intervals_ptr_count * sizeof(std::size_t), cudaMemcpyHostToDevice));

            std::size_t* d_row_output_offsets;
            CUDA_CHECK(cudaMalloc(&d_row_output_offsets, h_row_output_offsets.size() * sizeof(std::size_t)));
            CUDA_CHECK(cudaMemcpy(d_row_output_offsets, h_row_output_offsets.data(), h_row_output_offsets.size() * sizeof(std::size_t), cudaMemcpyHostToDevice));

            int* d_active_unified_y;
            CUDA_CHECK(cudaMalloc(&d_active_unified_y, result_host.y_coords.size() * sizeof(int)));
            CUDA_CHECK(cudaMemcpy(d_active_unified_y, result_host.y_coords.data(), result_host.y_coords.size() * sizeof(int), cudaMemcpyHostToDevice));

            int active_blocks = (result_host.y_coords.size() + threads_per_block - 1) / threads_per_block;
            set_op_2d_kernel<<<active_blocks, threads_per_block>>>(d_a, d_b, d_active_unified_y, result_host.y_coords.size(), d_result, d_row_output_offsets, nullptr, true, true);
            CUDA_CHECK(cudaGetLastError());

            // 6. Copy result back to host
            result_host.from_device(d_result);

            // 7. Cleanup
            d_a.free(); d_b.free(); d_result.free();
            CUDA_CHECK(cudaFree(d_unified_y));
            CUDA_CHECK(cudaFree(d_active_unified_y));
            CUDA_CHECK(cudaFree(d_row_output_counts));
            CUDA_CHECK(cudaFree(d_row_output_offsets));

            return result_host;
        }

        // --- Host-side 2D Intersection ---
        CSIR_Level_Host intersection(const CSIR_Level_Host& a, const CSIR_Level_Host& b)
        {
            if (a.level != b.level || a.empty() || b.empty()) return CSIR_Level_Host();

            // 1. Create intersecting Y-coordinate list on host
            std::vector<int> intersecting_y;
            std::set_intersection(a.y_coords.begin(), a.y_coords.end(),
                                  b.y_coords.begin(), b.y_coords.end(),
                                  std::back_inserter(intersecting_y));

            if (intersecting_y.empty()) return CSIR_Level_Host();

            // 2. Copy inputs to device
            CSIR_Level_Device d_a = a.to_device();
            CSIR_Level_Device d_b = b.to_device();
            int* d_intersecting_y;
            CUDA_CHECK(cudaMalloc(&d_intersecting_y, intersecting_y.size() * sizeof(int)));
            CUDA_CHECK(cudaMemcpy(d_intersecting_y, intersecting_y.data(), intersecting_y.size() * sizeof(int), cudaMemcpyHostToDevice));

            // 3. Sizing phase
            std::size_t* d_row_output_counts;
            CUDA_CHECK(cudaMalloc(&d_row_output_counts, intersecting_y.size() * sizeof(std::size_t)));
            
            int threads_per_block = 256;
            int blocks = (intersecting_y.size() + threads_per_block - 1) / threads_per_block;
            set_op_2d_kernel<<<blocks, threads_per_block>>>(d_a, d_b, d_intersecting_y, intersecting_y.size(), {}, nullptr, d_row_output_counts, false, false);
            CUDA_CHECK(cudaGetLastError());

            // 4. Prefix sum on host
            std::vector<std::size_t> h_row_output_counts(intersecting_y.size());
            CUDA_CHECK(cudaMemcpy(h_row_output_counts.data(), d_row_output_counts, intersecting_y.size() * sizeof(std::size_t), cudaMemcpyDeviceToHost));

            CSIR_Level_Host result_host;
            result_host.level = a.level;
            result_host.intervals_ptr.push_back(0);
            std::size_t total_intervals = 0;

            std::vector<std::size_t> h_row_output_offsets;
            h_row_output_offsets.reserve(intersecting_y.size());

            for(size_t i = 0; i < intersecting_y.size(); ++i) {
                if (h_row_output_counts[i] > 0) {
                    result_host.y_coords.push_back(intersecting_y[i]);
                    h_row_output_offsets.push_back(total_intervals);
                    total_intervals += h_row_output_counts[i];
                    result_host.intervals_ptr.push_back(total_intervals);
                }
            }
            
            if (total_intervals == 0) { // Handle case where result is empty
                d_a.free(); d_b.free(); CUDA_CHECK(cudaFree(d_intersecting_y)); CUDA_CHECK(cudaFree(d_row_output_counts));
                return result_host;
            }

            // 5. Compute phase
            CSIR_Level_Device d_result = {};
            d_result.level = result_host.level;
            d_result.y_coords_count = result_host.y_coords.size();
            d_result.intervals_ptr_count = result_host.intervals_ptr.size();
            d_result.intervals_count = total_intervals;

            CUDA_CHECK(cudaMalloc(&d_result.y_coords, d_result.y_coords_count * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_result.intervals_ptr, d_result.intervals_ptr_count * sizeof(std::size_t)));
            CUDA_CHECK(cudaMalloc(&d_result.intervals, d_result.intervals_count * sizeof(Interval)));
            
            CUDA_CHECK(cudaMemcpy(d_result.y_coords, result_host.y_coords.data(), d_result.y_coords_count * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_result.intervals_ptr, result_host.intervals_ptr.data(), d_result.intervals_ptr_count * sizeof(std::size_t), cudaMemcpyHostToDevice));

            std::size_t* d_row_output_offsets;
            CUDA_CHECK(cudaMalloc(&d_row_output_offsets, h_row_output_offsets.size() * sizeof(std::size_t)));
            CUDA_CHECK(cudaMemcpy(d_row_output_offsets, h_row_output_offsets.data(), h_row_output_offsets.size() * sizeof(std::size_t), cudaMemcpyHostToDevice));

            int* d_active_intersecting_y;
            CUDA_CHECK(cudaMalloc(&d_active_intersecting_y, result_host.y_coords.size() * sizeof(int)));
            CUDA_CHECK(cudaMemcpy(d_active_intersecting_y, result_host.y_coords.data(), result_host.y_coords.size() * sizeof(int), cudaMemcpyHostToDevice));

            int active_blocks = (result_host.y_coords.size() + threads_per_block - 1) / threads_per_block;
            set_op_2d_kernel<<<active_blocks, threads_per_block>>>(d_a, d_b, d_active_intersecting_y, result_host.y_coords.size(), d_result, d_row_output_offsets, nullptr, true, false);
            CUDA_CHECK(cudaGetLastError());

            // 6. Copy result back to host
            result_host.from_device(d_result);

            // 7. Cleanup
            d_a.free(); d_b.free(); d_result.free();
            CUDA_CHECK(cudaFree(d_intersecting_y));
            CUDA_CHECK(cudaFree(d_active_intersecting_y));
            CUDA_CHECK(cudaFree(d_row_output_counts));
            CUDA_CHECK(cudaFree(d_row_output_offsets));

            return result_host;
        }
    } // namespace cuda
} // namespace csir