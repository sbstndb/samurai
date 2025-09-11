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

    // --- Union (Optimized) ---
    void union_1d(std::vector<Interval>::const_iterator a_begin, std::vector<Interval>::const_iterator a_end,
                  std::vector<Interval>::const_iterator b_begin, std::vector<Interval>::const_iterator b_end,
                  std::vector<Interval>& result)
    {
        if (a_begin == a_end) { result.insert(result.end(), b_begin, b_end); return; }
        if (b_begin == b_end) { result.insert(result.end(), a_begin, a_end); return; }

        Interval current_interval;
        if (a_begin->start < b_begin->start) {
            current_interval = *a_begin++;
        } else {
            current_interval = *b_begin++;
        }

        while (a_begin != a_end || b_begin != b_end) {
            const Interval* next_interval = nullptr;
            if (a_begin != a_end && (b_begin == b_end || a_begin->start < b_begin->start)) {
                next_interval = &(*a_begin++);
            } else if (b_begin != b_end) {
                next_interval = &(*b_begin++);
            }

            if (next_interval->start <= current_interval.end) { 
                current_interval.end = std::max(current_interval.end, next_interval->end);
            } else { 
                result.push_back(current_interval);
                current_interval = *next_interval;
            }
        }
        result.push_back(current_interval);
    }

    CSIR_Level union_(const CSIR_Level& a, const CSIR_Level& b)
    {
        CSIR_Level result;
        if (a.level != b.level) { return result; }
        result.level = a.level;
        if (a.empty()) return b;
        if (b.empty()) return a;

        result.intervals_ptr.push_back(0);
        auto it_a = 0;
        auto it_b = 0;

        while(it_a < a.y_coords.size() || it_b < b.y_coords.size())
        {
            int y;
            size_t current_interval_count = result.intervals.size();

            if (it_a < a.y_coords.size() && (it_b >= b.y_coords.size() || a.y_coords[it_a] < b.y_coords[it_b])) {
                y = a.y_coords[it_a];
                result.intervals.insert(result.intervals.end(), a.intervals.begin() + a.intervals_ptr[it_a], a.intervals.begin() + a.intervals_ptr[it_a+1]);
                it_a++;
            } else if (it_b < b.y_coords.size() && (it_a >= a.y_coords.size() || b.y_coords[it_b] < a.y_coords[it_a])) {
                y = b.y_coords[it_b];
                result.intervals.insert(result.intervals.end(), b.intervals.begin() + b.intervals_ptr[it_b], b.intervals.begin() + b.intervals_ptr[it_b+1]);
                it_b++;
            } else { 
                y = a.y_coords[it_a];
                union_1d(a.intervals.begin() + a.intervals_ptr[it_a], a.intervals.begin() + a.intervals_ptr[it_a+1],
                         b.intervals.begin() + b.intervals_ptr[it_b], b.intervals.begin() + b.intervals_ptr[it_b+1],
                         result.intervals);
                it_a++;
                it_b++;
            }

            if (result.intervals.size() > current_interval_count)
            {
                result.y_coords.push_back(y);
                result.intervals_ptr.push_back(result.intervals.size());
            }
        }
        return result;
    }

    // --- Difference (Optimized) ---
    void difference_1d(std::vector<Interval>::const_iterator a_begin, std::vector<Interval>::const_iterator a_end,
                         std::vector<Interval>::const_iterator b_begin, std::vector<Interval>::const_iterator b_end,
                         std::vector<Interval>& result)
    {
        while (a_begin != a_end) {
            int current_start = a_begin->start;
            int current_end = a_begin->end;
            auto temp_b_begin = b_begin;

            while (temp_b_begin != b_end && temp_b_begin->start < current_end) {
                if (temp_b_begin->end > current_start) {
                    if (current_start < temp_b_begin->start) {
                        result.push_back({current_start, temp_b_begin->start});
                    }
                    current_start = std::max(current_start, temp_b_begin->end);
                }
                ++temp_b_begin;
            }

            if (current_start < current_end) {
                result.push_back({current_start, current_end});
            }
            ++a_begin;
        }
    }

    CSIR_Level difference(const CSIR_Level& a, const CSIR_Level& b)
    {
        CSIR_Level result;
        if (a.level != b.level) { return result; }
        result.level = a.level;
        if (a.empty()) return result;
        if (b.empty()) return a;

        result.intervals_ptr.push_back(0);
        auto it_a = 0;
        auto it_b = 0;

        while(it_a < a.y_coords.size())
        {
            int y = a.y_coords[it_a];
            size_t current_interval_count = result.intervals.size();

            while(it_b < b.y_coords.size() && b.y_coords[it_b] < y) {
                it_b++;
            }

            if (it_b < b.y_coords.size() && b.y_coords[it_b] == y) {
                difference_1d(a.intervals.begin() + a.intervals_ptr[it_a], a.intervals.begin() + a.intervals_ptr[it_a+1],
                              b.intervals.begin() + b.intervals_ptr[it_b], b.intervals.begin() + b.intervals_ptr[it_b+1],
                              result.intervals);
            } else {
                result.intervals.insert(result.intervals.end(), a.intervals.begin() + a.intervals_ptr[it_a], a.intervals.begin() + a.intervals_ptr[it_a+1]);
            }

            if (result.intervals.size() > current_interval_count) {
                result.y_coords.push_back(y);
                result.intervals_ptr.push_back(result.intervals.size());
            }
            it_a++;
        }
        return result;
    }

    // --- Geometric Operations ---
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

    // --- Intersection (Optimized) ---
    // Now accepts iterators to avoid temporary vector copies
    void intersection_1d(std::vector<Interval>::const_iterator a_begin, std::vector<Interval>::const_iterator a_end,
                         std::vector<Interval>::const_iterator b_begin, std::vector<Interval>::const_iterator b_end,
                         std::vector<Interval>& result)
    {
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