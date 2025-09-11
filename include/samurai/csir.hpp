
// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <vector>
#include <map>
#include <algorithm>

#include "cell_array.hpp"
#include "level_cell_array.hpp"
#include "list_of_intervals.hpp"

namespace samurai
{
    namespace csir
    {
        // The basic interval structure, identical to samurai::Interval
        using samurai::Interval;

        // CSIR data structure for a single level in 2D
        template<class TInterval = samurai::default_config::interval_t>
        struct CSIR_Level
        {
            using interval_t = TInterval;
            using value_t = typename interval_t::value_t;

            // The y-coordinates that have active cells.
            // For a dense representation, this could be omitted and implied by the index.
            std::vector<value_t> y_coords;

            // For each y_coord, `intervals_ptr[i]` is the starting index
            // in the `intervals` array.
            std::vector<std::size_t> intervals_ptr;

            // A single flat array of all intervals for this level.
            std::vector<interval_t> intervals;

            // Metadata
            std::size_t level = 0;
            value_t min_y, max_y;

            bool empty() const
            {
                return intervals.empty();
            }
        };

        // A multi-level CSIR mesh, mirroring the structure of samurai::CellArray
        template<class TInterval = samurai::default_config::interval_t>
        struct CSIR_Mesh
        {
            using interval_t = TInterval;
            std::map<std::size_t, CSIR_Level<interval_t>> levels;
        };

        // Forward declarations
        template<class TInterval>
        CSIR_Level<TInterval> to_csir_level(const samurai::LevelCellArray<2, TInterval>& lca);

        template<class TInterval>
        samurai::LevelCellArray<2, TInterval> from_csir_level(const CSIR_Level<TInterval>& csir_level);


        // Conversion from samurai::LevelCellArray<2> to CSIR_Level
        template<class TInterval>
        CSIR_Level<TInterval> to_csir_level(const samurai::LevelCellArray<2, TInterval>& lca)
        {
            CSIR_Level<TInterval> csir;
            csir.level = lca.level();
            if (lca.empty())
            {
                csir.min_y = 0;
                csir.max_y = -1;
                csir.intervals_ptr.push_back(0);
                return csir;
            }

            const auto& y_intervals = lca[1]; // In 2D, dim 1 is y
            const auto& x_intervals = lca[0]; // In 2D, dim 0 is x

            csir.min_y = y_intervals.front().start;
            csir.max_y = y_intervals.back().end - 1;

            csir.intervals_ptr.push_back(0);

            for(const auto& y_interval : y_intervals)
            {
                for(auto y = y_interval.start; y < y_interval.end; ++y)
                {
                    csir.y_coords.push_back(y);
                    auto y_index = y_interval.index + y;

                    auto x_start_offset = lca.offsets(1)[y_index];
                    auto x_end_offset = lca.offsets(1)[y_index + 1];

                    for (std::size_t i = x_start_offset; i < x_end_offset; ++i)
                    {
                        csir.intervals.push_back(x_intervals[i]);
                    }
                    csir.intervals_ptr.push_back(csir.intervals.size());
                }
            }
            return csir;
        }

        // Conversion from CSIR_Level to samurai::LevelCellArray<2>
        template<class TInterval>
        samurai::LevelCellArray<2, TInterval> from_csir_level(const CSIR_Level<TInterval>& csir_level)
        {
            samurai::LevelCellArray<2, TInterval> lca(csir_level.level);
            if (csir_level.empty())
            {
                return lca;
            }

            samurai::ListOfIntervals<typename TInterval::value_t> y_list;
            for(const auto& y : csir_level.y_coords)
            {
                y_list.add_interval({y, y + 1});
            }

            std::vector<samurai::ListOfIntervals<typename TInterval::value_t>> x_lists(csir_level.y_coords.size());
            for(std::size_t i = 0; i < csir_level.y_coords.size(); ++i)
            {
                auto start_ptr = csir_level.intervals_ptr[i];
                auto end_ptr = csir_level.intervals_ptr[i+1];
                for(std::size_t j = start_ptr; j < end_ptr; ++j)
                {
                    x_lists[i].add_interval(csir_level.intervals[j]);
                }
            }

            lca.construct(x_lists, y_list);
            return lca;
        }


        // 1D interval intersection logic
        template<class TInterval>
        void intersection_1d(const std::vector<TInterval>& list_a, const std::vector<TInterval>& list_b, std::vector<TInterval>& result)
        {
            auto it_a = list_a.begin();
            auto it_b = list_b.begin();
            while (it_a != list_a.end() && it_b != list_b.end())
            {
                auto max_start = std::max(it_a->start, it_b->start);
                auto min_end = std::min(it_a->end, it_b->end);

                if (max_start < min_end)
                {
                    result.push_back({max_start, min_end});
                }

                if (it_a->end < it_b->end)
                {
                    ++it_a;
                }
                else
                {
                    ++it_b;
                }
            }
        }

        // Intersection of two CSIR_Level objects
        template<class TInterval>
        CSIR_Level<TInterval> intersection(const CSIR_Level<TInterval>& a, const CSIR_Level<TInterval>& b)
        {
            CSIR_Level<TInterval> result;
            result.level = a.level;
            if (a.empty() || b.empty())
            {
                return result;
            }

            result.intervals_ptr.push_back(0);

            auto it_a = 0; 
            auto it_b = 0;

            while(it_a < a.y_coords.size() && it_b < b.y_coords.size())
            {
                if (a.y_coords[it_a] < b.y_coords[it_b])
                {
                    it_a++;
                }
                else if (b.y_coords[it_b] < a.y_coords[it_a])
                {
                    it_b++;
                }
                else // y-coordinates are equal
                {
                    auto y = a.y_coords[it_a];
                    
                    // Get slices of intervals for this y
                    std::vector<TInterval> intervals_a, intervals_b, intersection_result;
                    
                    auto start_a = a.intervals_ptr[it_a];
                    auto end_a = a.intervals_ptr[it_a+1];
                    for(size_t i = start_a; i < end_a; ++i) intervals_a.push_back(a.intervals[i]);

                    auto start_b = b.intervals_ptr[it_b];
                    auto end_b = b.intervals_ptr[it_b+1];
                    for(size_t i = start_b; i < end_b; ++i) intervals_b.push_back(b.intervals[i]);

                    // Compute 1D intersection
                    intersection_1d(intervals_a, intervals_b, intersection_result);

                    if (!intersection_result.empty())
                    {
                        result.y_coords.push_back(y);
                        for(const auto& interval : intersection_result)
                        {
                            result.intervals.push_back(interval);
                        }
                        result.intervals_ptr.push_back(result.intervals.size());
                    }

                    it_a++;
                    it_b++;
                }
            }

            if (!result.y_coords.empty())
            {
                result.min_y = result.y_coords.front();
                result.max_y = result.y_coords.back();
            }

            return result;
        }


        // 1D union logic
        template<class TInterval>
        void union_1d(const std::vector<TInterval>& list_a, const std::vector<TInterval>& list_b, std::vector<TInterval>& result)
        {
            auto it_a = list_a.begin();
            auto it_b = list_b.begin();
            TInterval current_interval;

            if (it_a == list_a.end()) { result = list_b; return; }
            if (it_b == list_b.end()) { result = list_a; return; }

            // Initialize with the first interval
            if (it_a->start < it_b->start) {
                current_interval = *it_a++;
            } else {
                current_interval = *it_b++;
            }

            while (it_a != list_a.end() || it_b != list_b.end()) {
                const TInterval* next_interval = nullptr;
                if (it_a != list_a.end() && (it_b == list_b.end() || it_a->start < it_b->start)) {
                    next_interval = &(*it_a++);
                } else if (it_b != list_b.end()) {
                    next_interval = &(*it_b++);
                }

                if (next_interval->start < current_interval.end) { // Overlap or contiguous
                    current_interval.end = std::max(current_interval.end, next_interval->end);
                } else { // Disjoint
                    result.push_back(current_interval);
                    current_interval = *next_interval;
                }
            }
            result.push_back(current_interval);
        }

        // 1D difference logic (a - b)
        template<class TInterval>
        void difference_1d(const std::vector<TInterval>& list_a, const std::vector<TInterval>& list_b, std::vector<TInterval>& result)
        {
            auto it_a = list_a.begin();
            auto it_b = list_b.begin();

            while (it_a != list_a.end()) {
                auto current_start = it_a->start;
                auto current_end = it_a->end;

                while (it_b != list_b.end() && it_b->end < current_start) {
                    it_b++;
                }

                while (it_b != list_b.end() && it_b->start < current_end) {
                    if (current_start < it_b->start) {
                        result.push_back({current_start, it_b->start});
                    }
                    current_start = std::max(current_start, it_b->end);
                    if (current_start >= it_b->end) {
                        it_b++;
                    }
                }

                if (current_start < current_end) {
                    result.push_back({current_start, current_end});
                }
                it_a++;
            }
        }

        // Union of two CSIR_Level objects
        template<class TInterval>
        CSIR_Level<TInterval> union_(const CSIR_Level<TInterval>& a, const CSIR_Level<TInterval>& b)
        {
            CSIR_Level<TInterval> result;
            result.level = a.level;
            if (a.empty()) return b;
            if (b.empty()) return a;

            result.intervals_ptr.push_back(0);

            auto it_a = 0;
            auto it_b = 0;

            while(it_a < a.y_coords.size() || it_b < b.y_coords.size())
            {
                typename TInterval::value_t y;
                std::vector<TInterval> intervals_a, intervals_b, union_result;

                if (it_a < a.y_coords.size() && (it_b >= b.y_coords.size() || a.y_coords[it_a] < b.y_coords[it_b])) {
                    y = a.y_coords[it_a];
                    auto start_a = a.intervals_ptr[it_a];
                    auto end_a = a.intervals_ptr[it_a+1];
                    for(size_t i = start_a; i < end_a; ++i) union_result.push_back(a.intervals[i]);
                    it_a++;
                } else if (it_b < b.y_coords.size() && (it_a >= a.y_coords.size() || b.y_coords[it_b] < a.y_coords[it_a])) {
                    y = b.y_coords[it_b];
                    auto start_b = b.intervals_ptr[it_b];
                    auto end_b = b.intervals_ptr[it_b+1];
                    for(size_t i = start_b; i < end_b; ++i) union_result.push_back(b.intervals[i]);
                    it_b++;
                } else { // y-coordinates are equal
                    y = a.y_coords[it_a];
                    auto start_a = a.intervals_ptr[it_a];
                    auto end_a = a.intervals_ptr[it_a+1];
                    for(size_t i = start_a; i < end_a; ++i) intervals_a.push_back(a.intervals[i]);

                    auto start_b = b.intervals_ptr[it_b];
                    auto end_b = b.intervals_ptr[it_b+1];
                    for(size_t i = start_b; i < end_b; ++i) intervals_b.push_back(b.intervals[i]);

                    union_1d(intervals_a, intervals_b, union_result);
                    it_a++;
                    it_b++;
                }

                if (!union_result.empty())
                {
                    result.y_coords.push_back(y);
                    for(const auto& interval : union_result)
                    {
                        result.intervals.push_back(interval);
                    }
                    result.intervals_ptr.push_back(result.intervals.size());
                }
            }
            return result;
        }

        // Difference of two CSIR_Level objects (a - b)
        template<class TInterval>
        CSIR_Level<TInterval> difference(const CSIR_Level<TInterval>& a, const CSIR_Level<TInterval>& b)
        {
            CSIR_Level<TInterval> result;
            result.level = a.level;
            if (a.empty()) return result;
            if (b.empty()) return a;

            result.intervals_ptr.push_back(0);

            auto it_a = 0;
            auto it_b = 0;

            while(it_a < a.y_coords.size())
            {
                auto y = a.y_coords[it_a];
                std::vector<TInterval> intervals_a, intervals_b, diff_result;

                auto start_a = a.intervals_ptr[it_a];
                auto end_a = a.intervals_ptr[it_a+1];
                for(size_t i = start_a; i < end_a; ++i) intervals_a.push_back(a.intervals[i]);

                // Find corresponding y in b
                while(it_b < b.y_coords.size() && b.y_coords[it_b] < y) {
                    it_b++;
                }

                if (it_b < b.y_coords.size() && b.y_coords[it_b] == y) {
                    auto start_b = b.intervals_ptr[it_b];
                    auto end_b = b.intervals_ptr[it_b+1];
                    for(size_t i = start_b; i < end_b; ++i) intervals_b.push_back(b.intervals[i]);
                    difference_1d(intervals_a, intervals_b, diff_result);
                } else {
                    diff_result = intervals_a;
                }

                if (!diff_result.empty())
                {
                    result.y_coords.push_back(y);
                    for(const auto& interval : diff_result)
                    {
                        result.intervals.push_back(interval);
                    }
                    result.intervals_ptr.push_back(result.intervals.size());
                }
                it_a++;
            }
            return result;
        }

    } // namespace csir
} // namespace samurai
