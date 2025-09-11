#pragma once

#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <map>
#include <array>

namespace csir
{
    // Base interval type used in 1D/2D
    struct Interval
    {
        int start, end;
    };

    // 1D code is placed after 2D helpers below
    // --- 2D Structures ---
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
    inline CSIR_Level intersection(const CSIR_Level& a, const CSIR_Level& b);

    // --- Union (Optimized) ---
    inline void union_1d(std::vector<Interval>::const_iterator a_begin, std::vector<Interval>::const_iterator a_end,
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

    inline CSIR_Level union_(const CSIR_Level& a, const CSIR_Level& b)
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
    inline void difference_1d(std::vector<Interval>::const_iterator a_begin, std::vector<Interval>::const_iterator a_end,
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

    inline CSIR_Level difference(const CSIR_Level& a, const CSIR_Level& b)
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

    // ---------------------- 1D (after helpers) ----------------------
    struct CSIR_Level_1D
    {
        std::vector<Interval> intervals;
        std::size_t level = 0;
        bool empty() const { return intervals.empty(); }
    };

    inline void print_level_1d(const CSIR_Level_1D& csir)
    {
        if (csir.empty()) { std::cout << "    <empty>" << std::endl; return; }
        std::cout << "    ";
        for (const auto& itv : csir.intervals) { std::cout << "[" << itv.start << ", " << itv.end << ") "; }
        std::cout << std::endl;
    }

    inline CSIR_Level_1D intersection(const CSIR_Level_1D& a, const CSIR_Level_1D& b)
    {
        CSIR_Level_1D out; out.level = a.level;
        if (a.level != b.level || a.empty() || b.empty()) return out;
        auto ia = a.intervals.begin();
        auto ib = b.intervals.begin();
        while (ia != a.intervals.end() && ib != b.intervals.end())
        {
            int s = std::max(ia->start, ib->start);
            int e = std::min(ia->end,   ib->end);
            if (s < e) out.intervals.push_back({s,e});
            if (ia->end < ib->end) ++ia; else ++ib;
        }
        return out;
    }

    inline CSIR_Level_1D union_(const CSIR_Level_1D& a, const CSIR_Level_1D& b)
    {
        CSIR_Level_1D out; if (a.level != b.level) return out; out.level = a.level;
        if (a.empty()) { out = b; return out; }
        if (b.empty()) { out = a; return out; }
        union_1d(a.intervals.begin(), a.intervals.end(), b.intervals.begin(), b.intervals.end(), out.intervals);
        return out;
    }

    inline CSIR_Level_1D difference(const CSIR_Level_1D& a, const CSIR_Level_1D& b)
    {
        CSIR_Level_1D out; if (a.level != b.level) return out; out.level = a.level;
        if (a.empty()) return out; if (b.empty()) { out = a; return out; }
        difference_1d(a.intervals.begin(), a.intervals.end(), b.intervals.begin(), b.intervals.end(), out.intervals);
        return out;
    }

    inline CSIR_Level_1D translate(const CSIR_Level_1D& src, int dx)
    {
        CSIR_Level_1D out; out.level = src.level;
        out.intervals.reserve(src.intervals.size());
        for (auto itv : src.intervals) { itv.start += dx; itv.end += dx; out.intervals.push_back(itv); }
        return out;
    }

    inline CSIR_Level_1D contract(const CSIR_Level_1D& set, std::size_t width)
    {
        if (set.empty()) return set;
        auto r = intersection(set, translate(set, static_cast<int>(width)));
        r = intersection(r, translate(set, -static_cast<int>(width)));
        return r;
    }

    inline CSIR_Level_1D expand(const CSIR_Level_1D& set, std::size_t width)
    {
        if (set.empty() || width == 0) return set;
        auto r = set;
        for (std::size_t k = 1; k <= width; ++k)
        {
            r = union_(r, translate(set, static_cast<int>(k)));
            r = union_(r, translate(set, -static_cast<int>(k)));
        }
        return r;
    }

    inline CSIR_Level_1D project_to_level(const CSIR_Level_1D& source, std::size_t target_level)
    {
        if (source.level == target_level) return source;
        CSIR_Level_1D out; out.level = target_level;
        if (source.level < target_level)
        {
            int scale = 1 << (target_level - source.level);
            out.intervals.reserve(source.intervals.size());
            for (auto itv : source.intervals)
            {
                out.intervals.push_back({itv.start * scale, itv.end * scale});
            }
            return out;
        }
        else
        {
            int scale = 1 << (source.level - target_level);
            auto floor_div = [scale](int v) { return v >= 0 ? (v / scale) : -(((-v + scale - 1) / scale)); };
            auto ceil_div  = [scale](int v) { return v >= 0 ? ((v + scale - 1) / scale) : -((-v) / scale); };
            std::vector<Interval> tmp;
            for (auto itv : source.intervals)
            {
                int s = floor_div(itv.start);
                int e = ceil_div(itv.end);
                if (s < e) tmp.push_back({s,e});
            }
            if (!tmp.empty())
            {
                std::sort(tmp.begin(), tmp.end(), [](const Interval& a, const Interval& b){ return a.start < b.start; });
                Interval cur = tmp.front();
                for (std::size_t i = 1; i < tmp.size(); ++i)
                {
                    if (tmp[i].start <= cur.end) cur.end = std::max(cur.end, tmp[i].end);
                    else { out.intervals.push_back(cur); cur = tmp[i]; }
                }
                out.intervals.push_back(cur);
            }
            return out;
        }
    }
    // ---------------------- Geometric ops (2D) ----------------------
    inline CSIR_Level translate(const CSIR_Level& src, int dx, int dy)
    {
        CSIR_Level out;
        out.level = src.level;
        if (src.empty())
        {
            out.intervals_ptr.push_back(0);
            return out;
        }

        out.y_coords.reserve(src.y_coords.size());
        out.intervals.reserve(src.intervals.size());
        out.intervals_ptr.reserve(src.intervals_ptr.size());
        out.intervals_ptr.push_back(0);

        for (std::size_t i = 0; i < src.y_coords.size(); ++i)
        {
            out.y_coords.push_back(src.y_coords[i] + dy);
            auto s = src.intervals_ptr[i];
            auto e = src.intervals_ptr[i + 1];
            for (std::size_t k = s; k < e; ++k)
            {
                auto itv = src.intervals[k];
                itv.start += dx;
                itv.end += dx;
                out.intervals.push_back(itv);
            }
            out.intervals_ptr.push_back(out.intervals.size());
        }
        return out;
    }

    inline CSIR_Level contract(const CSIR_Level& set, std::size_t width, const std::array<bool, 2>& dir_mask)
    {
        if (set.empty())
        {
            CSIR_Level out; out.level = set.level; out.intervals_ptr.push_back(0); return out;
        }
        CSIR_Level res = set;
        if (width == 0) { return res; }
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

    inline CSIR_Level contract(const CSIR_Level& set, std::size_t width)
    {
        return contract(set, width, std::array<bool, 2>{true, true});
    }

    inline CSIR_Level expand(const CSIR_Level& set, std::size_t width, const std::array<bool, 2>& dir_mask)
    {
        if (set.empty() || width == 0) { return set; }
        CSIR_Level res = set;
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

    inline CSIR_Level expand(const CSIR_Level& set, std::size_t width)
    {
        return expand(set, width, std::array<bool, 2>{true, true});
    }

    // --- Geometric Operations ---
    inline CSIR_Level project_to_level(const CSIR_Level& source, std::size_t target_level) {
        if (source.level == target_level) return source;

        CSIR_Level result;
        result.level = target_level;

        if (source.level < target_level) {
            // Upscaling
            int scale = 1 << (target_level - source.level);

            // Pre-allocate
            result.y_coords.reserve(source.y_coords.size() * static_cast<std::size_t>(scale));
            result.intervals.reserve(source.intervals.size() * static_cast<std::size_t>(scale));
            result.intervals_ptr.reserve(source.y_coords.size() * static_cast<std::size_t>(scale) + 1);

            result.intervals_ptr.push_back(0);

            for (size_t i = 0; i < source.y_coords.size(); ++i) {
                int y = source.y_coords[i];
                auto start_idx = source.intervals_ptr[i];
                auto end_idx = source.intervals_ptr[i+1];

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
        } else {
            // Downscaling (coarsening)
            int scale = 1 << (source.level - target_level);

            // Aggregate intervals per coarse y
            std::map<int, std::vector<Interval>> rows;

            auto floor_div = [scale](int v) {
                return v >= 0 ? (v / scale) : -(((-v + scale - 1) / scale));
            };
            auto ceil_div = [scale](int v) {
                return v >= 0 ? ((v + scale - 1) / scale) : -((-v) / scale);
            };

            for (size_t i = 0; i < source.y_coords.size(); ++i) {
                int y_f = source.y_coords[i];
                int y_c = floor_div(y_f);

                auto start_idx = source.intervals_ptr[i];
                auto end_idx = source.intervals_ptr[i+1];
                for (size_t k = start_idx; k < end_idx; ++k) {
                    int s = source.intervals[k].start;
                    int e = source.intervals[k].end;
                    int cs = floor_div(s);
                    int ce = ceil_div(e);
                    if (cs < ce) rows[y_c].push_back({cs, ce});
                }
            }

            // Merge and flush
            result.intervals_ptr.push_back(0);
            for (auto& it : rows) {
                int y = it.first;
                auto& list = it.second;
                std::sort(list.begin(), list.end(), [](const Interval& a, const Interval& b){ return a.start < b.start; });
                std::vector<Interval> merged;
                if (!list.empty()) {
                    Interval cur = list.front();
                    for (size_t i = 1; i < list.size(); ++i) {
                        if (list[i].start <= cur.end) {
                            cur.end = std::max(cur.end, list[i].end);
                        } else {
                            merged.push_back(cur);
                            cur = list[i];
                        }
                    }
                    merged.push_back(cur);
                }
                if (!merged.empty()) {
                    result.y_coords.push_back(y);
                    result.intervals.insert(result.intervals.end(), merged.begin(), merged.end());
                    result.intervals_ptr.push_back(result.intervals.size());
                }
            }
            return result;
        }
    }

    // --- Print Helpers ---
    inline void print_level(const CSIR_Level& csir)
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

    inline void print_level_3d(const CSIR_Level_3D& csir_3d) {
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
    inline void intersection_1d(std::vector<Interval>::const_iterator a_begin, std::vector<Interval>::const_iterator a_end,
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

    inline CSIR_Level intersection(const CSIR_Level& a, const CSIR_Level& b)
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
                
                intersection_1d(a.intervals.begin() + a.intervals_ptr[it_a], a.intervals.begin() + a.intervals_ptr[it_a+1],
                              b.intervals.begin() + b.intervals_ptr[it_b], b.intervals.begin() + b.intervals_ptr[it_b+1],
                              result.intervals);

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

    // --- 3D Algorithms ---
    inline CSIR_Level_3D project_to_level_3d(const CSIR_Level_3D& source, std::size_t target_level) {
        if (source.level == target_level) return source;
        CSIR_Level_3D result;
        result.level = target_level;

        if (source.level < target_level) {
            // Upscaling: replicate slices and upscale XY
            int scale = 1 << (target_level - source.level);
            for(const auto& pair : source.slices) {
                int z = pair.first;
                CSIR_Level projected_slice_2d = project_to_level(pair.second, target_level);
                for (int i = 0; i < scale; ++i) {
                    result.slices[z * scale + i] = projected_slice_2d;
                }
            }
            return result;
        } else {
            // Downscaling: map fine z to coarse z and union
            int scale = 1 << (source.level - target_level);
            auto floor_div = [scale](int v) {
                return v >= 0 ? (v / scale) : -(((-v + scale - 1) / scale));
            };
            for (const auto& pair : source.slices) {
                int zf = pair.first;
                int zc = floor_div(zf);
                CSIR_Level proj2d = project_to_level(pair.second, target_level);
                auto it = result.slices.find(zc);
                if (it == result.slices.end()) {
                    result.slices[zc] = proj2d;
                } else {
                    it->second = union_(it->second, proj2d);
                }
            }
            return result;
        }
    }

    inline CSIR_Level_3D intersection_3d(const CSIR_Level_3D& a, const CSIR_Level_3D& b) {
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

    // -------------------------- 3D ops ---------------------------
    inline CSIR_Level_3D translate(const CSIR_Level_3D& src, int dx, int dy, int dz)
    {
        CSIR_Level_3D out;
        out.level = src.level;
        for (const auto& [z, slice] : src.slices)
        {
            out.slices[z + dz] = translate(slice, dx, dy);
        }
        return out;
    }

    inline CSIR_Level_3D union_3d(const CSIR_Level_3D& a, const CSIR_Level_3D& b)
    {
        CSIR_Level_3D result;
        if (a.level != b.level) return result;
        result.level = a.level;
        auto it_a = a.slices.begin();
        auto it_b = b.slices.begin();
        while (it_a != a.slices.end() || it_b != b.slices.end())
        {
            if (it_b == b.slices.end() || (it_a != a.slices.end() && it_a->first < it_b->first))
            {
                result.slices[it_a->first] = it_a->second;
                ++it_a;
            }
            else if (it_a == a.slices.end() || it_b->first < it_a->first)
            {
                result.slices[it_b->first] = it_b->second;
                ++it_b;
            }
            else // same z
            {
                result.slices[it_a->first] = union_(it_a->second, it_b->second);
                ++it_a;
                ++it_b;
            }
        }
        return result;
    }

    inline CSIR_Level_3D contract(const CSIR_Level_3D& set, std::size_t width, const std::array<bool, 3>& mask)
    {
        if (width == 0) return set;
        CSIR_Level_3D res = set;
        // X
        if (mask[0])
        {
            auto plus  = translate(set, static_cast<int>(width), 0, 0);
            auto minus = translate(set, -static_cast<int>(width), 0, 0);
            res = intersection_3d(res, plus);
            res = intersection_3d(res, minus);
        }
        // Y
        if (mask[1])
        {
            auto plus  = translate(set, 0, static_cast<int>(width), 0);
            auto minus = translate(set, 0, -static_cast<int>(width), 0);
            res = intersection_3d(res, plus);
            res = intersection_3d(res, minus);
        }
        // Z
        if (mask[2])
        {
            auto plus  = translate(set, 0, 0, static_cast<int>(width));
            auto minus = translate(set, 0, 0, -static_cast<int>(width));
            res = intersection_3d(res, plus);
            res = intersection_3d(res, minus);
        }
        return res;
    }

    inline CSIR_Level_3D contract(const CSIR_Level_3D& set, std::size_t width)
    {
        return contract(set, width, std::array<bool, 3>{true, true, true});
    }

    inline CSIR_Level_3D expand(const CSIR_Level_3D& set, std::size_t width, const std::array<bool, 3>& mask)
    {
        if (width == 0) return set;
        CSIR_Level_3D res = set;
        // X
        if (mask[0])
        {
            for (std::size_t k = 1; k <= width; ++k)
            {
                res = union_3d(res, translate(set, static_cast<int>(k), 0, 0));
                res = union_3d(res, translate(set, -static_cast<int>(k), 0, 0));
            }
        }
        // Y
        if (mask[1])
        {
            for (std::size_t k = 1; k <= width; ++k)
            {
                res = union_3d(res, translate(set, 0, static_cast<int>(k), 0));
                res = union_3d(res, translate(set, 0, -static_cast<int>(k), 0));
            }
        }
        // Z
        if (mask[2])
        {
            for (std::size_t k = 1; k <= width; ++k)
            {
                res = union_3d(res, translate(set, 0, 0, static_cast<int>(k)));
                res = union_3d(res, translate(set, 0, 0, -static_cast<int>(k)));
            }
        }
        return res;
    }

    inline CSIR_Level_3D expand(const CSIR_Level_3D& set, std::size_t width)
    {
        return expand(set, width, std::array<bool, 3>{true, true, true});
    }
} // namespace csir
