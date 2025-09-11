#pragma once

#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <map>

namespace csir
{
    // --- 2D Structures ---
    struct Interval
    {
        int start, end;
    };

    struct CSIR_Level
    {
        std::vector<int> y_coords;
        std::vector<std::size_t> intervals_ptr;
        std::vector<Interval> intervals;
        std::size_t level = 0;

        bool empty() const { return intervals.empty(); }
    };

    // --- 3D Structures ---
    struct CSIR_Level_3D {
        std::map<int, CSIR_Level> slices;
        std::size_t level = 0;
    };

    // --- Forward declarations ---
    CSIR_Level intersection(const CSIR_Level& a, const CSIR_Level& b);
    CSIR_Level project_to_level(const CSIR_Level& source, std::size_t target_level);

    // --- Print Helpers ---
    void print_level(const CSIR_Level& csir)
    {
        if (csir.empty()) {
            std::cout << "    <empty slice>" << std::endl;
            return;
        }
        for (size_t i = 0; i < csir.y_coords.size(); ++i)
        {
            std::cout << "    y = " << csir.y_coords[i] << ": ";
            auto start_idx = csir.intervals_ptr[i];
            auto end_idx = csir.intervals_ptr[i+1];
            for (size_t j = start_idx; j < end_idx; ++j)
            {
                std::cout << "[" << csir.intervals[j].start << ", " << csir.intervals[j].end << ") ";
            }
            std::cout << std::endl;
        }
    }

    void print_level_3d(const CSIR_Level_3D& csir_3d) {
        std::cout << "--- CSIR_Level_3D (Level: " << csir_3d.level << ") ---" << std::endl;
        if (csir_3d.slices.empty()) {
            std::cout << "<empty>" << std::endl;
        } else {
            for(const auto& pair : csir_3d.slices) {
                std::cout << "  Z = " << pair.first << ":" << std::endl;
                print_level(pair.second);
            }
        }
        std::cout << "-------------------------------------" << std::endl;
    }

    // --- Intersection (Optimized v2: Loop Unrolling) ---
    void intersection_1d(std::vector<Interval>::const_iterator a_begin, std::vector<Interval>::const_iterator a_end,
                         std::vector<Interval>::const_iterator b_begin, std::vector<Interval>::const_iterator b_end,
                         std::vector<Interval>& result)
    {
        // Process 2 intervals at a time
        while (a_begin + 1 < a_end && b_begin + 1 < b_end)
        {
            // Unroll 1
            auto max_start1 = std::max(a_begin->start, b_begin->start);
            auto min_end1 = std::min(a_begin->end, b_begin->end);
            if (max_start1 < min_end1) { result.push_back({max_start1, min_end1}); }

            // Unroll 2
            auto max_start2 = std::max((a_begin+1)->start, (b_begin+1)->start);
            auto min_end2 = std::min((a_begin+1)->end, (b_begin+1)->end);
            if (max_start2 < min_end2) { result.push_back({max_start2, min_end2}); }

            if (a_begin->end < b_begin->end) { ++a_begin; } else { ++b_begin; }
            if ((a_begin+1)->end < (b_begin+1)->end) { ++a_begin; } else { ++b_begin; }
        }

        // Cleanup loop for remaining intervals
        while (a_begin != a_end && b_begin != b_end)
        {
            auto max_start = std::max(a_begin->start, b_begin->start);
            auto min_end = std::min(a_begin->end, b_begin->end);
            if (max_start < min_end) { result.push_back({max_start, min_end}); }
            if (a_begin->end < b_begin->end) { ++a_begin; } else { ++b_begin; }
        }
    }

    CSIR_Level intersection(const CSIR_Level& a, const CSIR_Level& b)
    {
        CSIR_Level result;
        result.level = a.level;
        if (a.empty() || b.empty()) return result;

        result.intervals_ptr.push_back(0);
        auto it_a = 0; 
        auto it_b = 0;

        while(it_a < a.y_coords.size() && it_b < b.y_coords.size())
        {
            if (a.y_coords[it_a] < b.y_coords[it_b]) {
                it_a++;
            } else if (b.y_coords[it_b] < a.y_coords[it_a]) {
                it_b++;
            } else { 
                auto y = a.y_coords[it_a];
                
                // No longer creating temporary vectors. Pass iterators directly.
                intersection_1d(a.intervals.begin() + a.intervals_ptr[it_a], a.intervals.begin() + a.intervals_ptr[it_a+1],
                              b.intervals.begin() + b.intervals_ptr[it_b], b.intervals.begin() + b.intervals_ptr[it_b+1],
                              result.intervals); // Append directly to the final interval list

                // The logic to update y_coords and ptrs needs adjustment
                if (result.intervals.size() > result.intervals_ptr.back())
                {
                    result.y_coords.push_back(y);
                    result.intervals_ptr.push_back(result.intervals.size());
                }
                it_a++;
                it_b++;
            }
        }
        return result;
    }

    CSIR_Level project_to_level(const CSIR_Level& source, std::size_t target_level) {
        if (source.level == target_level) return source;
        if (source.level > target_level) return source; 

        CSIR_Level result;
        result.level = target_level;
        int scale = 1 << (target_level - source.level);
        
        // Optimization: Pre-allocate memory
        result.y_coords.reserve(source.y_coords.size() * scale);
        result.intervals.reserve(source.intervals.size() * scale);
        result.intervals_ptr.reserve(source.y_coords.size() * scale + 1);

        result.intervals_ptr.push_back(0);

        for (size_t i = 0; i < source.y_coords.size(); ++i) {
            int y = source.y_coords[i];
            auto start_idx = source.intervals_ptr[i];
            auto end_idx = source.intervals_ptr[i+1];
            size_t num_intervals_per_row = end_idx - start_idx;

            for (int j = 0; j < scale; ++j) {
                result.y_coords.push_back(y * scale + j);
                for (size_t k = start_idx; k < end_idx; ++k) {
                    const auto& interval = source.intervals[k];
                    result.intervals.push_back({interval.start * scale, interval.end * scale});
                }
                result.intervals_ptr.push_back(result.intervals.size());
            }
        }
        return result;
    }

    // --- 3D Algorithms ---
    CSIR_Level_3D project_to_level_3d(const CSIR_Level_3D& source, std::size_t target_level) {
        CSIR_Level_3D result;
        result.level = target_level;
        if (source.level == target_level) return source;
        if (source.level > target_level) return source;

        int scale = 1 << (target_level - source.level);
        for(const auto& pair : source.slices) {
            int z = pair.first;
            CSIR_Level projected_slice_2d = project_to_level(pair.second, target_level);
            for (int i = 0; i < scale; ++i) {
                result.slices[z * scale + i] = projected_slice_2d;
            }
        }
        return result;
    }

    CSIR_Level_3D intersection_3d(const CSIR_Level_3D& a, const CSIR_Level_3D& b) {
        CSIR_Level_3D result;
        result.level = a.level;
        if (a.level != b.level) return result;

        auto it_a = a.slices.begin();
        auto it_b = b.slices.begin();

        while (it_a != a.slices.end() && it_b != b.slices.end()) {
            if (it_a->first < it_b->first) {
                ++it_a;
            } else if (it_b->first < it_a->first) {
                ++it_b;
            } else { 
                auto intersection_2d = intersection(it_a->second, it_b->second);
                if (!intersection_2d.empty()) {
                    result.slices[it_a->first] = intersection_2d;
                }
                ++it_a;
                ++it_b;
            }
        }
        return result;
    }

} // namespace csir