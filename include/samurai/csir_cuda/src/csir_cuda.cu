#include "csir_cuda.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <vector>
#include <array>

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

        // --- 1D Host/Device transfer utilities ---
        CSIR_Level_1D_Device CSIR_Level_1D_Host::to_device() const {
            CSIR_Level_1D_Device device_level = {};
            device_level.level = level;
            device_level.intervals_count = intervals.size();
            if (device_level.intervals_count > 0) {
                CUDA_CHECK(cudaMalloc(&device_level.intervals, device_level.intervals_count * sizeof(Interval)));
                CUDA_CHECK(cudaMemcpy(
                    device_level.intervals,
                    intervals.data(),
                    device_level.intervals_count * sizeof(Interval),
                    cudaMemcpyHostToDevice));
            } else {
                device_level.intervals = nullptr;
            }
            return device_level;
        }

        void CSIR_Level_1D_Host::from_device(const CSIR_Level_1D_Device& device_level) {
            level = device_level.level;
            intervals.resize(device_level.intervals_count);
            if (device_level.intervals_count > 0 && device_level.intervals != nullptr) {
                CUDA_CHECK(cudaMemcpy(
                    intervals.data(),
                    device_level.intervals,
                    device_level.intervals_count * sizeof(Interval),
                    cudaMemcpyDeviceToHost));
            } else {
                intervals.clear();
            }
        }

        void CSIR_Level_1D_Device::free() {
            if (intervals) {
                CUDA_CHECK(cudaFree(intervals));
                intervals = nullptr;
            }
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

        // --- Device-side 1D Difference ---
        __device__ void difference_1d_device(
            const Interval* a_begin, const Interval* a_end,
            const Interval* b_begin, const Interval* b_end,
            Interval* result, std::size_t& result_count, bool compute)
        {
            result_count = 0;
            if (a_begin == a_end) return;
            if (b_begin == b_end) { // If B is empty, result is A
                result_count = (a_end - a_begin);
                if (compute) for (int i=0; i<result_count; ++i) result[i] = a_begin[i];
                return;
            }

            auto ia = a_begin;
            auto ib = b_begin;
            while (ia != a_end) {
                int a_s = ia->start;
                const int a_e = ia->end;

                while (ib != b_end && ib->end <= a_s) ++ib;

                int cur = a_s;
                while (ib != b_end && ib->start < a_e) {
                    if (ib->start > cur) {
                        if (compute) result[result_count] = {cur, min(ib->start, a_e)};
                        result_count++;
                    }
                    if (ib->end >= a_e) {
                        cur = a_e;
                        break;
                    }
                    cur = max(cur, ib->end);
                    ++ib;
                }
                if (cur < a_e) {
                    if (compute) result[result_count] = {cur, a_e};
                    result_count++;
                }
                ++ia;
            }
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

        // --- Generic 2D Kernel (for Union, Intersection, Difference) ---
        __global__ void set_op_2d_kernel(
            CSIR_Level_Device a, CSIR_Level_Device b,
            const int* unified_y, int unified_y_count,
            CSIR_Level_Device result,
            const std::size_t* row_output_offsets,
            std::size_t* row_output_counts,
            bool compute,
            int op_type) // 0 for union, 1 for intersection, 2 for difference
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

            if (op_type == 0) { // Union
                union_1d_device(a_begin, a_end, b_begin, b_end, result_ptr, count, compute);
            } else if (op_type == 1) { // Intersection
                intersection_1d_device(a_begin, a_end, b_begin, b_end, result_ptr, count, compute);
            } else if (op_type == 2) { // Difference
                difference_1d_device(a_begin, a_end, b_begin, b_end, result_ptr, count, compute);
            }

            if (!compute) {
                row_output_counts[idx] = count;
            }
        }

        // Forward declarations for geometric kernels
        __global__ void translate_2d_kernel(CSIR_Level_Device src, CSIR_Level_Device out, int dx, int dy);
        __global__ void project_to_level_sizing_kernel(CSIR_Level_Device source, std::size_t* row_output_counts, int scale, bool is_upscaling);
        __global__ void project_to_level_compute_kernel(CSIR_Level_Device source, CSIR_Level_Device result, const std::size_t* row_output_offsets, int scale, bool is_upscaling);

        // --- Host-side 2D Operations ---
        // Helper function to abstract common logic for set operations
        CSIR_Level_Host perform_set_op_2d(
            const CSIR_Level_Host& a, const CSIR_Level_Host& b,
            int op_type, // 0: union, 1: intersection, 2: difference
            const std::vector<int>& initial_y_coords) // Y-coords to process
        {
            if (a.level != b.level) return CSIR_Level_Host();

            // Handle empty cases specific to each operation
            if (op_type == 0) { // Union
                if (a.empty()) return b;
                if (b.empty()) return a;
            } else if (op_type == 1) { // Intersection
                if (a.empty() || b.empty()) return CSIR_Level_Host();
            } else if (op_type == 2) { // Difference
                if (a.empty()) return CSIR_Level_Host();
                if (b.empty()) return a; // A - empty = A
            }

            // 1. Prepare Y-coordinate list on host
            std::vector<int> processed_y_coords = initial_y_coords;
            std::sort(processed_y_coords.begin(), processed_y_coords.end());
            processed_y_coords.erase(std::unique(processed_y_coords.begin(), processed_y_coords.end()), processed_y_coords.end());

            if (processed_y_coords.empty()) return CSIR_Level_Host();

            // 2. Copy inputs to device
            CSIR_Level_Device d_a = a.to_device();
            CSIR_Level_Device d_b = b.to_device();
            int* d_processed_y_coords;
            CUDA_CHECK(cudaMalloc(&d_processed_y_coords, processed_y_coords.size() * sizeof(int)));
            CUDA_CHECK(cudaMemcpy(d_processed_y_coords, processed_y_coords.data(), processed_y_coords.size() * sizeof(int), cudaMemcpyHostToDevice));

            // 3. Sizing phase
            std::size_t* d_row_output_counts;
            CUDA_CHECK(cudaMalloc(&d_row_output_counts, processed_y_coords.size() * sizeof(std::size_t)));

            int threads_per_block = 256;
            int blocks = (processed_y_coords.size() + threads_per_block - 1) / threads_per_block;
            set_op_2d_kernel<<<blocks, threads_per_block>>>(d_a, d_b, d_processed_y_coords, processed_y_coords.size(), {}, nullptr, d_row_output_counts, false, op_type);
            CUDA_CHECK(cudaGetLastError());

            // 4. Prefix sum on host
            std::vector<std::size_t> h_row_output_counts(processed_y_coords.size());
            CUDA_CHECK(cudaMemcpy(h_row_output_counts.data(), d_row_output_counts, processed_y_coords.size() * sizeof(std::size_t), cudaMemcpyDeviceToHost));

            CSIR_Level_Host result_host;
            result_host.level = a.level;
            result_host.intervals_ptr.push_back(0);
            std::size_t total_intervals = 0;

            std::vector<std::size_t> h_row_output_offsets;
            h_row_output_offsets.reserve(processed_y_coords.size());

            for(size_t i = 0; i < processed_y_coords.size(); ++i) {
                if (h_row_output_counts[i] > 0) {
                    result_host.y_coords.push_back(processed_y_coords[i]);
                    h_row_output_offsets.push_back(total_intervals);
                    total_intervals += h_row_output_counts[i];
                    result_host.intervals_ptr.push_back(total_intervals);
                }
            }

            if (total_intervals == 0) { // Handle case where result is empty
                d_a.free(); d_b.free(); CUDA_CHECK(cudaFree(d_processed_y_coords)); CUDA_CHECK(cudaFree(d_row_output_counts));
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

            std::size_t* d_row_output_offsets_gpu;
            CUDA_CHECK(cudaMalloc(&d_row_output_offsets_gpu, h_row_output_offsets.size() * sizeof(std::size_t)));
            CUDA_CHECK(cudaMemcpy(d_row_output_offsets_gpu, h_row_output_offsets.data(), h_row_output_offsets.size() * sizeof(std::size_t), cudaMemcpyHostToDevice));

            int* d_active_processed_y_coords;
            CUDA_CHECK(cudaMalloc(&d_active_processed_y_coords, result_host.y_coords.size() * sizeof(int)));
            CUDA_CHECK(cudaMemcpy(d_active_processed_y_coords, result_host.y_coords.data(), result_host.y_coords.size() * sizeof(int), cudaMemcpyHostToDevice));

            int active_blocks = (result_host.y_coords.size() + threads_per_block - 1) / threads_per_block;
            set_op_2d_kernel<<<active_blocks, threads_per_block>>>(d_a, d_b, d_active_processed_y_coords, result_host.y_coords.size(), d_result, d_row_output_offsets_gpu, nullptr, true, op_type);
            CUDA_CHECK(cudaGetLastError());

            // 6. Copy result back to host
            result_host.from_device(d_result);

            // 7. Cleanup
            d_a.free(); d_b.free(); d_result.free();
            CUDA_CHECK(cudaFree(d_processed_y_coords));
            CUDA_CHECK(cudaFree(d_active_processed_y_coords));
            CUDA_CHECK(cudaFree(d_row_output_counts));
            CUDA_CHECK(cudaFree(d_row_output_offsets_gpu));

            return result_host;
        }

        CSIR_Level_Host union_(const CSIR_Level_Host& a, const CSIR_Level_Host& b)
        {
            std::vector<int> unified_y = a.y_coords;
            unified_y.insert(unified_y.end(), b.y_coords.begin(), b.y_coords.end());
            return perform_set_op_2d(a, b, 0, unified_y);
        }

        CSIR_Level_Host intersection(const CSIR_Level_Host& a, const CSIR_Level_Host& b)
        {
            std::vector<int> intersecting_y;
            std::set_intersection(a.y_coords.begin(), a.y_coords.end(),
                                  b.y_coords.begin(), b.y_coords.end(),
                                  std::back_inserter(intersecting_y));
            return perform_set_op_2d(a, b, 1, intersecting_y);
        }

        CSIR_Level_Host difference(const CSIR_Level_Host& a, const CSIR_Level_Host& b)
        {
            // For difference (A - B), we only need to process y-coordinates present in A
            return perform_set_op_2d(a, b, 2, a.y_coords);
        }

        // --- Geometric Operations (2D) ---
        CSIR_Level_Host translate(const CSIR_Level_Host& src, int dx, int dy)
        {
            CSIR_Level_Host out;
            out.level = src.level;
            if (src.empty()) return out;

            // Allocate device memory for source
            CSIR_Level_Device d_src = src.to_device();

            // Allocate device memory for output (same size as source)
            CSIR_Level_Device d_out = {};
            d_out.level = src.level;
            d_out.y_coords_count = src.y_coords.size();
            d_out.intervals_ptr_count = src.intervals_ptr.size();
            d_out.intervals_count = src.intervals.size();

            CUDA_CHECK(cudaMalloc(&d_out.y_coords, d_out.y_coords_count * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_out.intervals_ptr, d_out.intervals_ptr_count * sizeof(std::size_t)));
            CUDA_CHECK(cudaMalloc(&d_out.intervals, d_out.intervals_count * sizeof(Interval)));

            // Launch kernel for translation
            int threads_per_block = 256;
            int blocks = (src.y_coords.size() + threads_per_block - 1) / threads_per_block;
            translate_2d_kernel<<<blocks, threads_per_block>>>(d_src, d_out, dx, dy);
            CUDA_CHECK(cudaGetLastError());

            // Copy result back to host
            out.from_device(d_out);

            // Cleanup
            d_src.free();
            d_out.free();

            return out;
        }

        __global__ void translate_2d_kernel(
            CSIR_Level_Device src, CSIR_Level_Device out,
            int dx, int dy)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= src.y_coords_count) return;

            // Translate y_coord
            out.y_coords[idx] = src.y_coords[idx] + dy;

            // Copy and translate intervals
            out.intervals_ptr[idx] = src.intervals_ptr[idx]; // This will be adjusted later if needed
            if (idx + 1 < src.intervals_ptr_count) {
                out.intervals_ptr[idx+1] = src.intervals_ptr[idx+1];
            }

            std::size_t start_idx = src.intervals_ptr[idx];
            std::size_t end_idx = src.intervals_ptr[idx+1];

            for (std::size_t i = start_idx; i < end_idx; ++i) {
                out.intervals[i].start = src.intervals[i].start + dx;
                out.intervals[i].end = src.intervals[i].end + dx;
            }
        }

        CSIR_Level_Host project_to_level(const CSIR_Level_Host& source, std::size_t target_level)
        {
            if (source.level == target_level) return source;

            CSIR_Level_Host result_host;
            result_host.level = target_level;

            // Allocate device memory for source
            CSIR_Level_Device d_source = source.to_device();

            if (source.level < target_level) {
                // Upscaling
                int scale = 1 << (target_level - source.level);

                // Sizing kernel for upscaling
                std::size_t* d_row_output_counts;
                CUDA_CHECK(cudaMalloc(&d_row_output_counts, source.y_coords.size() * sizeof(std::size_t)));

                int threads_per_block = 256;
                int blocks = (source.y_coords.size() + threads_per_block - 1) / threads_per_block;
                project_to_level_sizing_kernel<<<blocks, threads_per_block>>>(d_source, d_row_output_counts, scale, true);
                CUDA_CHECK(cudaGetLastError());

                std::vector<std::size_t> h_row_output_counts(source.y_coords.size());
                CUDA_CHECK(cudaMemcpy(h_row_output_counts.data(), d_row_output_counts, source.y_coords.size() * sizeof(std::size_t), cudaMemcpyDeviceToHost));

                std::size_t total_intervals = 0;
                std::vector<std::size_t> h_row_output_offsets;
                h_row_output_offsets.reserve(source.y_coords.size() * scale);

                for(size_t i = 0; i < source.y_coords.size(); ++i) {
                    for (int j = 0; j < scale; ++j) {
                        result_host.y_coords.push_back(source.y_coords[i] * scale + j);
                        h_row_output_offsets.push_back(total_intervals);
                        total_intervals += h_row_output_counts[i]; // Each fine row gets the same number of intervals as the source row
                        result_host.intervals_ptr.push_back(total_intervals);
                    }
                }

                if (total_intervals == 0) {
                    d_source.free(); CUDA_CHECK(cudaFree(d_row_output_counts));
                    return result_host;
                }

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

                std::size_t* d_row_output_offsets_gpu;
                CUDA_CHECK(cudaMalloc(&d_row_output_offsets_gpu, h_row_output_offsets.size() * sizeof(std::size_t)));
                CUDA_CHECK(cudaMemcpy(d_row_output_offsets_gpu, h_row_output_offsets.data(), h_row_output_offsets.size() * sizeof(std::size_t), cudaMemcpyHostToDevice));

                int active_blocks = (result_host.y_coords.size() + threads_per_block - 1) / threads_per_block;
                project_to_level_compute_kernel<<<active_blocks, threads_per_block>>>(d_source, d_result, d_row_output_offsets_gpu, scale, true);
                CUDA_CHECK(cudaGetLastError());

                result_host.from_device(d_result);
                d_result.free();
                CUDA_CHECK(cudaFree(d_row_output_counts));
                CUDA_CHECK(cudaFree(d_row_output_offsets_gpu));

            } else { // Downscaling
                int scale = 1 << (source.level - target_level);

                // Sizing kernel for downscaling
                std::size_t* d_row_output_counts;
                CUDA_CHECK(cudaMalloc(&d_row_output_counts, source.y_coords.size() * sizeof(std::size_t)));

                int threads_per_block = 256;
                int blocks = (source.y_coords.size() + threads_per_block - 1) / threads_per_block;
                project_to_level_sizing_kernel<<<blocks, threads_per_block>>>(d_source, d_row_output_counts, scale, false);
                CUDA_CHECK(cudaGetLastError());

                std::vector<std::size_t> h_row_output_counts(source.y_coords.size());
                CUDA_CHECK(cudaMemcpy(h_row_output_counts.data(), d_row_output_counts, source.y_coords.size() * sizeof(std::size_t), cudaMemcpyDeviceToHost));

                // Group contiguous fine rows mapping to the same coarse row
                std::vector<int> coarse_y_coords;
                std::vector<std::size_t> coarse_row_start_indices;
                coarse_row_start_indices.push_back(0);
                std::size_t current_coarse_y_idx = 0;
                std::size_t total_intervals = 0;

                auto floor_div = [scale](int v) { return v >= 0 ? (v / scale) : -(((-v + scale - 1) / scale)); };

                if (!source.y_coords.empty()) {
                    int current_cy = floor_div(source.y_coords[0]);
                    coarse_y_coords.push_back(current_cy);

                    for (size_t i = 0; i < source.y_coords.size(); ++i) {
                        int y_f = source.y_coords[i];
                        int y_c = floor_div(y_f);
                        if (y_c != current_cy) {
                            current_cy = y_c;
                            coarse_y_coords.push_back(current_cy);
                            coarse_row_start_indices.push_back(total_intervals);
                        }
                        total_intervals += h_row_output_counts[i];
                    }
                    coarse_row_start_indices.push_back(total_intervals);
                }

                if (total_intervals == 0) {
                    d_source.free(); CUDA_CHECK(cudaFree(d_row_output_counts));
                    return result_host;
                }

                CSIR_Level_Device d_result = {};
                d_result.level = result_host.level;
                d_result.y_coords_count = coarse_y_coords.size();
                d_result.intervals_ptr_count = coarse_row_start_indices.size();
                d_result.intervals_count = total_intervals;

                CUDA_CHECK(cudaMalloc(&d_result.y_coords, d_result.y_coords_count * sizeof(int)));
                CUDA_CHECK(cudaMalloc(&d_result.intervals_ptr, d_result.intervals_ptr_count * sizeof(std::size_t)));
                CUDA_CHECK(cudaMalloc(&d_result.intervals, d_result.intervals_count * sizeof(Interval)));

                CUDA_CHECK(cudaMemcpy(d_result.y_coords, coarse_y_coords.data(), d_result.y_coords_count * sizeof(int), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_result.intervals_ptr, coarse_row_start_indices.data(), d_result.intervals_ptr_count * sizeof(std::size_t), cudaMemcpyHostToDevice));

                std::size_t* d_row_output_offsets_gpu;
                CUDA_CHECK(cudaMalloc(&d_row_output_offsets_gpu, h_row_output_counts.size() * sizeof(std::size_t)));
                CUDA_CHECK(cudaMemcpy(d_row_output_offsets_gpu, h_row_output_counts.data(), h_row_output_counts.size() * sizeof(std::size_t), cudaMemcpyHostToDevice));

                int active_blocks = (source.y_coords.size() + threads_per_block - 1) / threads_per_block;
                project_to_level_compute_kernel<<<active_blocks, threads_per_block>>>(d_source, d_result, d_row_output_offsets_gpu, scale, false);
                CUDA_CHECK(cudaGetLastError());

                result_host.from_device(d_result);
                d_result.free();
                CUDA_CHECK(cudaFree(d_row_output_counts));
                CUDA_CHECK(cudaFree(d_row_output_offsets_gpu));
            }

            d_source.free();
            return result_host;
        }

        __global__ void project_to_level_sizing_kernel(
            CSIR_Level_Device source,
            std::size_t* row_output_counts,
            int scale,
            bool is_upscaling) // true for upscaling, false for downscaling
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= source.y_coords_count) return;

            std::size_t count = 0;
            std::size_t start_idx = source.intervals_ptr[idx];
            std::size_t end_idx = source.intervals_ptr[idx+1];

            if (is_upscaling) {
                // Each interval from source row is scaled, so count is the same
                count = end_idx - start_idx;
            } else {
                // Downscaling: intervals might merge or disappear
                auto floor_div = [scale](int v) { return v >= 0 ? (v / scale) : -(((-v + scale - 1) / scale)); };
                auto ceil_div  = [scale](int v) { return v >= 0 ? ((v + scale - 1) / scale) : -((-v) / scale); };

                // Simulate 1D projection and count resulting intervals
                // This is a simplified version, a full 1D projection would involve merging
                // intervals that become contiguous after scaling.
                // For now, we just count how many intervals remain after scaling.
                for (std::size_t i = start_idx; i < end_idx; ++i) {
                    int s = source.intervals[i].start;
                    int e = source.intervals[i].end;
                    int cs = floor_div(s);
                    int ce = ceil_div(e);
                    if (cs < ce) {
                        count++;
                    }
                }
            }
            row_output_counts[idx] = count;
        }

        __global__ void project_to_level_compute_kernel(
            CSIR_Level_Device source,
            CSIR_Level_Device result,
            const std::size_t* row_output_offsets,
            int scale,
            bool is_upscaling) // true for upscaling, false for downscaling
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= source.y_coords_count) return;

            std::size_t start_idx_src = source.intervals_ptr[idx];
            std::size_t end_idx_src = source.intervals_ptr[idx+1];

            std::size_t current_result_offset = row_output_offsets[idx];

            if (is_upscaling) {
                // Upscaling: replicate intervals and scale
                for (std::size_t i = start_idx_src; i < end_idx_src; ++i) {
                    result.intervals[current_result_offset].start = source.intervals[i].start * scale;
                    result.intervals[current_result_offset].end = source.intervals[i].end * scale;
                    current_result_offset++;
                }
            } else {
                // Downscaling: scale intervals and potentially merge
                auto floor_div = [scale](int v) { return v >= 0 ? (v / scale) : -(((-v + scale - 1) / scale)); };
                auto ceil_div  = [scale](int v) { return v >= 0 ? ((v + scale - 1) / scale) : -((-v) / scale); };

                // This part needs to perform the 1D projection and merge intervals
                // For now, a simplified copy without merging
                for (std::size_t i = start_idx_src; i < end_idx_src; ++i) {
                    int s = source.intervals[i].start;
                    int e = source.intervals[i].end;
                    int cs = floor_div(s);
                    int ce = ceil_div(e);
                    if (cs < ce) {
                        result.intervals[current_result_offset].start = cs;
                        result.intervals[current_result_offset].end = ce;
                        current_result_offset++;
                    }
                }
            }
        }

        // --- Host-side 2D Contract & Expand ---
        // These operations are defined in terms of translate, union, and intersection
        // So, once translate is implemented, these can be implemented by calling the CUDA versions
        // of translate, union, and intersection.

        CSIR_Level_Host contract(const CSIR_Level_Host& set, std::size_t width, const bool dir_mask[2])
        {
            if (set.empty()) return CSIR_Level_Host();
            if (width == 0) return set;
            CSIR_Level_Host res = set;
            // X
            if (dir_mask[0])
            {
                auto plus  = translate(set, static_cast<int>(width), 0);
                auto minus = translate(set, -static_cast<int>(width), 0);
                res = intersection(res, plus);
                res = intersection(res, minus);
            }
            // Y
            if (dir_mask[1])
            {
                auto plus  = translate(set, 0, static_cast<int>(width));
                auto minus = translate(set, 0, -static_cast<int>(width));
                res = intersection(res, plus);
                res = intersection(res, minus);
            }
            return res;
        }

        CSIR_Level_Host contract(const CSIR_Level_Host& set, std::size_t width)
        {
            return contract(set, width, (const bool[2]){true, true});
        }

        CSIR_Level_Host expand(const CSIR_Level_Host& set, std::size_t width, const bool dir_mask[2])
        {
            if (set.empty() || width == 0) { return set; }
            CSIR_Level_Host res = set;
            // X
            if (dir_mask[0])
            {
                for (std::size_t k = 1; k <= width; ++k)
                {
                    res = union_(res, translate(set, static_cast<int>(k), 0));
                    res = union_(res, translate(set, -static_cast<int>(k), 0));
                }
            }
            // Y
            if (dir_mask[1])
            {
                for (std::size_t k = 1; k <= width; ++k)
                {
                    res = union_(res, translate(set, 0, static_cast<int>(k)));
                    res = union_(res, translate(set, 0, -static_cast<int>(k)));
                }
            }
            return res;
        }

        CSIR_Level_Host expand(const CSIR_Level_Host& set, std::size_t width)
        {
            return expand(set, width, (const bool[2]){true, true});
        }



        // --- 1D Operations ---
        // Generic 1D Kernel for set operations
        __global__ void set_op_1d_kernel(
            CSIR_Level_1D_Device a, CSIR_Level_1D_Device b,
            CSIR_Level_1D_Device result,
            std::size_t* d_result_count_ptr, // Pointer to store result count for sizing
            bool compute,
            int op_type) // 0: union, 1: intersection, 2: difference
        {
            // Only one thread needed for 1D operations
            std::size_t count = 0;
            Interval* result_ptr = compute ? result.intervals : nullptr;

            if (op_type == 0) { // Union
                union_1d_device(a.intervals, a.intervals + a.intervals_count, b.intervals, b.intervals + b.intervals_count, result_ptr, count, compute);
            } else if (op_type == 1) { // Intersection
                intersection_1d_device(a.intervals, a.intervals + a.intervals_count, b.intervals, b.intervals + b.intervals_count, result_ptr, count, compute);
            } else if (op_type == 2) { // Difference
                difference_1d_device(a.intervals, a.intervals + a.intervals_count, b.intervals, b.intervals + b.intervals_count, result_ptr, count, compute);
            }

            if (!compute) {
                *d_result_count_ptr = count;
            }
        }

        // Helper function for 1D set operations
        CSIR_Level_1D_Host perform_set_op_1d(
            const CSIR_Level_1D_Host& a, const CSIR_Level_1D_Host& b,
            int op_type) // 0: union, 1: intersection, 2: difference
        {
            if (a.level != b.level) return CSIR_Level_1D_Host();

            // Handle empty cases specific to each operation
            if (op_type == 0) { // Union
                if (a.empty()) return b;
                if (b.empty()) return a;
            } else if (op_type == 1) { // Intersection
                if (a.empty() || b.empty()) return CSIR_Level_1D_Host();
            } else if (op_type == 2) { // Difference
                if (a.empty()) return CSIR_Level_1D_Host();
                if (b.empty()) return a; // A - empty = A
            }

            CSIR_Level_1D_Device d_a = a.to_device();
            CSIR_Level_1D_Device d_b = b.to_device();

            // Sizing phase
            std::size_t* d_result_count_ptr;
            CUDA_CHECK(cudaMalloc(&d_result_count_ptr, sizeof(std::size_t)));

            set_op_1d_kernel<<<1, 1>>>(d_a, d_b, {}, d_result_count_ptr, false, op_type);
            CUDA_CHECK(cudaGetLastError());

            std::size_t h_result_count;
            CUDA_CHECK(cudaMemcpy(&h_result_count, d_result_count_ptr, sizeof(std::size_t), cudaMemcpyDeviceToHost));

            CSIR_Level_1D_Host result_host;
            result_host.level = a.level;
            if (h_result_count == 0) {
                d_a.free(); d_b.free(); CUDA_CHECK(cudaFree(d_result_count_ptr));
                return result_host;
            }

            // Compute phase
            CSIR_Level_1D_Device d_result = {};
            d_result.level = result_host.level;
            d_result.intervals_count = h_result_count;
            CUDA_CHECK(cudaMalloc(&d_result.intervals, h_result_count * sizeof(Interval)));

            set_op_1d_kernel<<<1, 1>>>(d_a, d_b, d_result, nullptr, true, op_type);
            CUDA_CHECK(cudaGetLastError());

            result_host.from_device(d_result);

            d_a.free(); d_b.free(); d_result.free();
            CUDA_CHECK(cudaFree(d_result_count_ptr));
            return result_host;
        }

        CSIR_Level_1D_Host union_1d(const CSIR_Level_1D_Host& a, const CSIR_Level_1D_Host& b)
        {
            return perform_set_op_1d(a, b, 0);
        }

        CSIR_Level_1D_Host intersection_1d(const CSIR_Level_1D_Host& a, const CSIR_Level_1D_Host& b)
        {
            return perform_set_op_1d(a, b, 1);
        }

        CSIR_Level_1D_Host difference_1d(const CSIR_Level_1D_Host& a, const CSIR_Level_1D_Host& b)
        {
            return perform_set_op_1d(a, b, 2);
        }

        __global__ void translate_1d_kernel(
            CSIR_Level_1D_Device src, CSIR_Level_1D_Device out,
            int dx)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= src.intervals_count) return;

            out.intervals[idx].start = src.intervals[idx].start + dx;
            out.intervals[idx].end = src.intervals[idx].end + dx;
        }

        CSIR_Level_1D_Host translate_1d(const CSIR_Level_1D_Host& src, int dx)
        {
            CSIR_Level_1D_Host out;
            out.level = src.level;
            if (src.empty()) return out;

            CSIR_Level_1D_Device d_src = src.to_device();

            CSIR_Level_1D_Device d_out = {};
            d_out.level = src.level;
            d_out.intervals_count = src.intervals.size();
            CUDA_CHECK(cudaMalloc(&d_out.intervals, d_out.intervals_count * sizeof(Interval)));

            int threads_per_block = 256;
            int blocks = (src.intervals.size() + threads_per_block - 1) / threads_per_block;
            translate_1d_kernel<<<blocks, threads_per_block>>>(d_src, d_out, dx);
            CUDA_CHECK(cudaGetLastError());

            out.from_device(d_out);

            d_src.free();
            d_out.free();

            return out;
        }

        CSIR_Level_1D_Host contract_1d(const CSIR_Level_1D_Host& set, std::size_t width)
        {
            if (set.empty()) return CSIR_Level_1D_Host();
            if (width == 0) return set;

            // This will involve multiple GPU calls, similar to 2D contract/expand
            // For now, a direct port of the CPU logic using CUDA 1D ops
            CSIR_Level_1D_Host res = set;
            auto plus  = translate_1d(set, static_cast<int>(width));
            auto minus = translate_1d(set, -static_cast<int>(width));
            res = intersection_1d(res, plus);
            res = intersection_1d(res, minus);
            return res;
        }

        CSIR_Level_1D_Host expand_1d(const CSIR_Level_1D_Host& set, std::size_t width)
        {
            if (set.empty() || width == 0) return set;
            CSIR_Level_1D_Host res = set;
            for (std::size_t k = 1; k <= width; ++k)
            {
                res = union_1d(res, translate_1d(set, static_cast<int>(k)));
                res = union_1d(res, translate_1d(set, -static_cast<int>(k)));
            }
            return res;
        }

        __global__ void project_to_level_1d_kernel(
            CSIR_Level_1D_Device source,
            CSIR_Level_1D_Device result,
            int scale,
            bool is_upscaling) // true for upscaling, false for downscaling
        {
            int idx = threadIdx.x;
            if (idx >= source.intervals_count) return;

            if (is_upscaling) {
                result.intervals[idx].start = source.intervals[idx].start * scale;
                result.intervals[idx].end = source.intervals[idx].end * scale;
            } else {
                auto floor_div = [scale](int v) { return v >= 0 ? (v / scale) : -(((-v + scale - 1) / scale)); };
                auto ceil_div  = [scale](int v) { return v >= 0 ? ((v + scale - 1) / scale) : -((-v) / scale); };

                int s = source.intervals[idx].start;
                int e = source.intervals[idx].end;
                int cs = floor_div(s);
                int ce = ceil_div(e);

                // This is a simplified copy. A full 1D projection would involve merging
                // intervals that become contiguous after scaling.
                if (cs < ce) {
                    result.intervals[idx].start = cs;
                    result.intervals[idx].end = ce;
                }
            }
        }

        CSIR_Level_1D_Host project_to_level_1d(const CSIR_Level_1D_Host& source, std::size_t target_level)
        {
            if (source.level == target_level) return source;

            CSIR_Level_1D_Host result_host;
            result_host.level = target_level;

            CSIR_Level_1D_Device d_source = source.to_device();

            if (source.level < target_level) {
                // Upscaling
                int scale = 1 << (target_level - source.level);
                result_host.intervals.resize(source.intervals.size());

                CSIR_Level_1D_Device d_result = {};
                d_result.level = result_host.level;
                d_result.intervals_count = result_host.intervals.size();
                CUDA_CHECK(cudaMalloc(&d_result.intervals, d_result.intervals_count * sizeof(Interval)));

                int threads_per_block = 256;
                int blocks = (source.intervals.size() + threads_per_block - 1) / threads_per_block;
                project_to_level_1d_kernel<<<blocks, threads_per_block>>>(d_source, d_result, scale, true);
                CUDA_CHECK(cudaGetLastError());

                result_host.from_device(d_result);
                d_result.free();
            } else {
                // Downscaling
                int scale = 1 << (source.level - target_level);

                // Sizing phase for downscaling (on host for 1D)
                std::vector<Interval> tmp_intervals;
                tmp_intervals.reserve(source.intervals.size());
                auto floor_div = [scale](int v) { return v >= 0 ? (v / scale) : -(((-v + scale - 1) / scale)); };
                auto ceil_div  = [scale](int v) { return v >= 0 ? ((v + scale - 1) / scale) : -((-v) / scale); };

                for (const auto& itv : source.intervals) {
                    int s = floor_div(itv.start);
                    int e = ceil_div(itv.end);
                    if (s < e) tmp_intervals.emplace_back(Interval{s, e});
                }

                if (tmp_intervals.empty()) {
                    d_source.free();
                    return result_host;
                }

                // Sort and merge on host for now (can be optimized with Thrust or custom kernel)
                std::sort(tmp_intervals.begin(), tmp_intervals.end(), [](const Interval& a, const Interval& b){ return a.start < b.start; });
                Interval cur = tmp_intervals.front();
                for (std::size_t i = 1; i < tmp_intervals.size(); ++i) {
                    if (tmp_intervals[i].start <= cur.end) cur.end = std::max(cur.end, tmp_intervals[i].end);
                    else { result_host.intervals.push_back(cur); cur = tmp_intervals[i]; }
                }
                result_host.intervals.push_back(cur);
            }

            d_source.free();
            return result_host;
        }

    } // namespace cuda
} // namespace csir
