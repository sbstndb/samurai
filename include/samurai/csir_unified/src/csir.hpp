#pragma once

#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <map>
#include <array>
#include <limits>
#include <xtensor/xfixed.hpp>

// Samurai conversions (LCA <-> CSIR)
// These helpers allow using CSIR set algebra with Samurai data structures.
#include "../../level_cell_array.hpp"
#include "../../box.hpp"

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
    inline CSIR_Level_3D intersection(const CSIR_Level_3D& a, const CSIR_Level_3D& b);
    // Samurai interop forward decls (needed for templates below)
    template <class TInterval>
    CSIR_Level to_csir_level(const samurai::LevelCellArray<2, TInterval>& lca);
    template <class TInterval>
    CSIR_Level_3D to_csir_level(const samurai::LevelCellArray<3, TInterval>& lca);
    inline CSIR_Level_3D project_to_level(const CSIR_Level_3D& source, std::size_t target_level);

    // --- Union (Optimized) ---
    inline void union_1d(std::vector<Interval>::const_iterator a_begin, std::vector<Interval>::const_iterator a_end,
                  std::vector<Interval>::const_iterator b_begin, std::vector<Interval>::const_iterator b_end,
                  std::vector<Interval>& result)
    {
        // Reserve to avoid repeated reallocations (conservative upper bound)
        const auto a_len = static_cast<std::size_t>(std::distance(a_begin, a_end));
        const auto b_len = static_cast<std::size_t>(std::distance(b_begin, b_end));
        if (a_len + b_len > 0) {
            result.reserve(result.size() + a_len + b_len);
        }
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

        // Pre-allocate conservatively to reduce reallocations
        result.y_coords.reserve(a.y_coords.size() + b.y_coords.size());
        result.intervals.reserve(a.intervals.size() + b.intervals.size());
        result.intervals_ptr.push_back(0);
        std::size_t it_a = 0;
        std::size_t it_b = 0;

        while(it_a < a.y_coords.size() || it_b < b.y_coords.size())
        {
            int y;
            size_t current_interval_count = result.intervals.size();

            if (it_a < a.y_coords.size() && (it_b >= b.y_coords.size() || a.y_coords[it_a] < b.y_coords[it_b])) {
                y = a.y_coords[it_a];
                auto first = a.intervals.begin() + static_cast<std::ptrdiff_t>(a.intervals_ptr[it_a]);
                auto last  = a.intervals.begin() + static_cast<std::ptrdiff_t>(a.intervals_ptr[it_a + 1]);
                result.intervals.insert(result.intervals.end(), first, last);
                it_a++;
            } else if (it_b < b.y_coords.size() && (it_a >= a.y_coords.size() || b.y_coords[it_b] < a.y_coords[it_a])) {
                y = b.y_coords[it_b];
                auto first = b.intervals.begin() + static_cast<std::ptrdiff_t>(b.intervals_ptr[it_b]);
                auto last  = b.intervals.begin() + static_cast<std::ptrdiff_t>(b.intervals_ptr[it_b + 1]);
                result.intervals.insert(result.intervals.end(), first, last);
                it_b++;
            } else { 
                y = a.y_coords[it_a];
                auto a_first = a.intervals.begin() + static_cast<std::ptrdiff_t>(a.intervals_ptr[it_a]);
                auto a_last  = a.intervals.begin() + static_cast<std::ptrdiff_t>(a.intervals_ptr[it_a + 1]);
                auto b_first = b.intervals.begin() + static_cast<std::ptrdiff_t>(b.intervals_ptr[it_b]);
                auto b_last  = b.intervals.begin() + static_cast<std::ptrdiff_t>(b.intervals_ptr[it_b + 1]);
                union_1d(a_first, a_last,
                         b_first, b_last,
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
        // Linear-time difference leveraging sorted, non-overlapping inputs.
        // Reserve conservatively to reduce reallocations.
        const auto a_len = static_cast<std::size_t>(std::distance(a_begin, a_end));
        const auto b_len = static_cast<std::size_t>(std::distance(b_begin, b_end));
        if (a_len + b_len > 0) {
            result.reserve(result.size() + a_len + b_len);
        }

        auto ia = a_begin;
        auto ib = b_begin;
        while (ia != a_end) {
            int a_s = ia->start;
            const int a_e = ia->end;

            // Skip all B intervals that end before or at a_s
            while (ib != b_end && ib->end <= a_s) ++ib;

            int cur = a_s;
            while (ib != b_end && ib->start < a_e) {
                // Emit gap before current overlapping B interval
                if (ib->start > cur) {
                    result.emplace_back(Interval{cur, std::min(ib->start, a_e)});
                }
                // Advance cur past the overlap
                if (ib->end >= a_e) {
                    cur = a_e; // fully consumed
                    break;
                }
                cur = std::max(cur, ib->end);
                ++ib;
            }
            if (cur < a_e) {
                result.emplace_back(Interval{cur, a_e});
            }
            ++ia;
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
        std::size_t it_a = 0;
        std::size_t it_b = 0;

        while(it_a < a.y_coords.size())
        {
            int y = a.y_coords[it_a];
            size_t current_interval_count = result.intervals.size();

            while(it_b < b.y_coords.size() && b.y_coords[it_b] < y) {
                it_b++;
            }

            if (it_b < b.y_coords.size() && b.y_coords[it_b] == y) {
                auto a_first = a.intervals.begin() + static_cast<std::ptrdiff_t>(a.intervals_ptr[it_a]);
                auto a_last  = a.intervals.begin() + static_cast<std::ptrdiff_t>(a.intervals_ptr[it_a + 1]);
                auto b_first = b.intervals.begin() + static_cast<std::ptrdiff_t>(b.intervals_ptr[it_b]);
                auto b_last  = b.intervals.begin() + static_cast<std::ptrdiff_t>(b.intervals_ptr[it_b + 1]);
                difference_1d(a_first, a_last,
                              b_first, b_last,
                              result.intervals);
            } else {
                auto first = a.intervals.begin() + static_cast<std::ptrdiff_t>(a.intervals_ptr[it_a]);
                auto last  = a.intervals.begin() + static_cast<std::ptrdiff_t>(a.intervals_ptr[it_a + 1]);
                result.intervals.insert(result.intervals.end(), first, last);
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
        out.intervals.reserve(std::min(a.intervals.size(), b.intervals.size()));
        auto ia = a.intervals.begin();
        auto ib = b.intervals.begin();
        while (ia != a.intervals.end() && ib != b.intervals.end())
        {
            int s = std::max(ia->start, ib->start);
            int e = std::min(ia->end,   ib->end);
            if (s < e) out.intervals.emplace_back(Interval{s, e});
            if (ia->end < ib->end) ++ia; else ++ib;
        }
        return out;
    }

    inline CSIR_Level_1D union_(const CSIR_Level_1D& a, const CSIR_Level_1D& b)
    {
        CSIR_Level_1D out; if (a.level != b.level) return out; out.level = a.level;
        if (a.empty()) { out = b; return out; }
        if (b.empty()) { out = a; return out; }
        out.intervals.reserve(a.intervals.size() + b.intervals.size());
        union_1d(a.intervals.begin(), a.intervals.end(), b.intervals.begin(), b.intervals.end(), out.intervals);
        return out;
    }

    inline CSIR_Level_1D difference(const CSIR_Level_1D& a, const CSIR_Level_1D& b)
    {
        CSIR_Level_1D out;
        if (a.level != b.level) return out;
        out.level = a.level;
        if (a.empty()) {
            return out;
        }
        if (b.empty()) {
            out = a;
            return out;
        }
        out.intervals.reserve(a.intervals.size() + b.intervals.size());
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

    inline CSIR_Level_1D translate(const CSIR_Level_1D& src, const std::array<int, 1>& d)
    {
        return translate(src, d[0]);
    }

    inline CSIR_Level_1D translate(const CSIR_Level_1D& src, const xt::xtensor_fixed<int, xt::xshape<1>>& d)
    {
        return translate(src, d(0));
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

    inline CSIR_Level_1D expand(const CSIR_Level_1D& set, std::size_t width, const std::array<bool, 1>& mask)
    {
        if (!mask[0]) return set;
        return expand(set, width);
    }

    // Nested expand: union of translations by {-w,0,+w} per axis (includes diagonals)
    inline CSIR_Level_1D nested_expand(const CSIR_Level_1D& set, std::size_t width)
    {
        if (width == 0 || set.empty()) return set;
        CSIR_Level_1D out = set;
        int w = static_cast<int>(width);
        out = union_(out, translate(set, -w));
        out = union_(out, translate(set,  w));
        return out;
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
            tmp.reserve(source.intervals.size());
            for (auto itv : source.intervals)
            {
                int s = floor_div(itv.start);
                int e = ceil_div(itv.end);
                if (s < e) tmp.emplace_back(Interval{s, e});
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

    // Generalized translate overloads (2D)
    inline CSIR_Level translate(const CSIR_Level& src, const std::array<int, 2>& d)
    {
        return translate(src, d[0], d[1]);
    }

    inline CSIR_Level translate(const CSIR_Level& src, const xt::xtensor_fixed<int, xt::xshape<2>>& d)
    {
        return translate(src, d(0), d(1));
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

    inline CSIR_Level nested_expand(const CSIR_Level& set, std::size_t width, const std::array<bool, 2>& mask)
    {
        if (set.empty()) return set;
        CSIR_Level out = set;
        std::array<int,3> vals = { -static_cast<int>(width), 0, static_cast<int>(width) };
        for (int dx : (mask[0] ? std::array<int,3>{vals[0], vals[1], vals[2]} : std::array<int,3>{0,0,0}))
        {
            for (int dy : (mask[1] ? std::array<int,3>{vals[0], vals[1], vals[2]} : std::array<int,3>{0,0,0}))
            {
                if (dx == 0 && dy == 0) continue;
                out = union_(out, translate(set, dx, dy));
            }
        }
        return out;
    }

    inline CSIR_Level nested_expand(const CSIR_Level& set, std::size_t width)
    {
        return nested_expand(set, width, std::array<bool,2>{true, true});
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

            auto floor_div = [scale](int v) {
                return v >= 0 ? (v / scale) : -(((-v + scale - 1) / scale));
            };
            auto ceil_div = [scale](int v) {
                return v >= 0 ? ((v + scale - 1) / scale) : -((-v) / scale);
            };

            // Group contiguous fine rows mapping to the same coarse row
            result.intervals_ptr.push_back(0);
            bool have_group = false;
            int current_cy = 0;
            std::vector<Interval> bucket;
            bucket.reserve(64);

            auto flush = [&](int y) {
                if (bucket.empty()) return;
                std::sort(bucket.begin(), bucket.end(), [](const Interval& a, const Interval& b){ return a.start < b.start; });
                size_t start_idx = result.intervals.size();
                Interval cur = bucket.front();
                for (size_t i = 1; i < bucket.size(); ++i) {
                    if (bucket[i].start <= cur.end) {
                        cur.end = std::max(cur.end, bucket[i].end);
                    } else {
                        result.intervals.push_back(cur);
                        cur = bucket[i];
                    }
                }
                result.intervals.push_back(cur);
                if (result.intervals.size() > start_idx) {
                    result.y_coords.push_back(y);
                    result.intervals_ptr.push_back(result.intervals.size());
                }
                bucket.clear();
            };

            for (size_t i = 0; i < source.y_coords.size(); ++i) {
                int y_f = source.y_coords[i];
                int y_c = floor_div(y_f);
                if (!have_group) { have_group = true; current_cy = y_c; }
                if (y_c != current_cy) {
                    flush(current_cy);
                    current_cy = y_c;
                }
                auto start_idx = source.intervals_ptr[i];
                auto end_idx = source.intervals_ptr[i+1];
                for (size_t k = start_idx; k < end_idx; ++k) {
                    int s = source.intervals[k].start;
                    int e = source.intervals[k].end;
                    int cs = floor_div(s);
                    int ce = ceil_div(e);
                    if (cs < ce) bucket.emplace_back(Interval{cs, ce});
                }
            }
            if (have_group) {
                flush(current_cy);
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
        // Reserve up to min sizes; intersection size cannot exceed min count
        const auto a_len = static_cast<std::size_t>(std::distance(a_begin, a_end));
        const auto b_len = static_cast<std::size_t>(std::distance(b_begin, b_end));
        if (a_len > 0 && b_len > 0) {
            result.reserve(result.size() + std::min(a_len, b_len));
        }
        while (a_begin != a_end && b_begin != b_end)
        {
            auto max_start = std::max(a_begin->start, b_begin->start);
            auto min_end = std::min(a_begin->end, b_begin->end);
            if (max_start < min_end) { result.emplace_back(Interval{max_start, min_end}); }
            if (a_begin->end < b_begin->end) { ++a_begin; } else { ++b_begin; }
        }
    }

    inline CSIR_Level intersection(const CSIR_Level& a, const CSIR_Level& b)
    {
        CSIR_Level result;
        result.level = a.level;
        if (a.empty() || b.empty()) return result;

        // Pre-allocate conservatively
        result.y_coords.reserve(std::min(a.y_coords.size(), b.y_coords.size()));
        result.intervals.reserve(std::min(a.intervals.size(), b.intervals.size()));
        result.intervals_ptr.reserve(std::min(a.y_coords.size(), b.y_coords.size()) + 1);
        result.intervals_ptr.push_back(0);
        std::size_t it_a = 0; 
        std::size_t it_b = 0;

        while(it_a < a.y_coords.size() && it_b < b.y_coords.size())
        {
            if (a.y_coords[it_a] < b.y_coords[it_b]) {
                it_a++;
            } else if (b.y_coords[it_b] < a.y_coords[it_a]) {
                it_b++;
            } else { 
                auto y = a.y_coords[it_a];
                
                auto a_first = a.intervals.begin() + static_cast<std::ptrdiff_t>(a.intervals_ptr[it_a]);
                auto a_last  = a.intervals.begin() + static_cast<std::ptrdiff_t>(a.intervals_ptr[it_a + 1]);
                auto b_first = b.intervals.begin() + static_cast<std::ptrdiff_t>(b.intervals_ptr[it_b]);
                auto b_last  = b.intervals.begin() + static_cast<std::ptrdiff_t>(b.intervals_ptr[it_b + 1]);
                intersection_1d(a_first, a_last,
                              b_first, b_last,
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

    // Generalized translate overloads (3D)
    inline CSIR_Level_3D translate(const CSIR_Level_3D& src, const std::array<int, 3>& d)
    {
        return translate(src, d[0], d[1], d[2]);
    }

    inline CSIR_Level_3D translate(const CSIR_Level_3D& src, const xt::xtensor_fixed<int, xt::xshape<3>>& d)
    {
        return translate(src, d(0), d(1), d(2));
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

    // Dimension-generic alias for 3D union
    inline CSIR_Level_3D union_(const CSIR_Level_3D& a, const CSIR_Level_3D& b)
    {
        return union_3d(a, b);
    }

    // 3D difference: slice-wise difference using 2D difference
    inline CSIR_Level_3D difference_3d(const CSIR_Level_3D& a, const CSIR_Level_3D& b)
    {
        CSIR_Level_3D result;
        if (a.level != b.level) return result;
        result.level = a.level;

        auto it_a = a.slices.begin();
        auto it_b = b.slices.begin();

        while (it_a != a.slices.end())
        {
            while (it_b != b.slices.end() && it_b->first < it_a->first)
            {
                ++it_b;
            }
            if (it_b != b.slices.end() && it_b->first == it_a->first)
            {
                auto diff2d = difference(it_a->second, it_b->second);
                if (!diff2d.empty())
                {
                    result.slices[it_a->first] = diff2d;
                }
            }
            else
            {
                result.slices[it_a->first] = it_a->second;
            }
            ++it_a;
        }
        return result;
    }

    // Dimension-generic alias for 3D difference
    inline CSIR_Level_3D difference(const CSIR_Level_3D& a, const CSIR_Level_3D& b)
    {
        return difference_3d(a, b);
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

    inline CSIR_Level_3D nested_expand(const CSIR_Level_3D& set, std::size_t width, const std::array<bool,3>& mask)
    {
        CSIR_Level_3D out = set;
        int w = static_cast<int>(width);
        std::array<int,3> vals = { -w, 0, w };
        auto xs = mask[0] ? std::array<int,3>{vals[0], vals[1], vals[2]} : std::array<int,3>{0,0,0};
        auto ys = mask[1] ? std::array<int,3>{vals[0], vals[1], vals[2]} : std::array<int,3>{0,0,0};
        auto zs = mask[2] ? std::array<int,3>{vals[0], vals[1], vals[2]} : std::array<int,3>{0,0,0};
        for (int dx : xs)
        {
            for (int dy : ys)
            {
                for (int dz : zs)
                {
                    if (dx == 0 && dy == 0 && dz == 0) continue;
                    out = union_(out, translate(set, dx, dy, dz));
                }
            }
        }
        return out;
    }

    inline CSIR_Level_3D nested_expand(const CSIR_Level_3D& set, std::size_t width)
    {
        return nested_expand(set, width, std::array<bool,3>{true, true, true});
    }

    // Dimension-generic alias for project_to_level
    template <class Set>
    inline Set restrict_to_level(const Set& source, std::size_t target_level)
    {
        return project_to_level(source, target_level);
    }

    // Box / domain helpers
    template <class T>
    inline CSIR_Level to_csir_box(const samurai::Box<T, 2>& box, std::size_t level)
    {
        samurai::LevelCellArray<2> lca(level, box);
        return to_csir_level(lca);
    }

    template <class T>
    inline CSIR_Level_3D to_csir_box(const samurai::Box<T, 3>& box, std::size_t level)
    {
        samurai::LevelCellArray<3> lca(level, box);
        return to_csir_level(lca);
    }

    // Restrict a set to a domain LCA at a target level (2D)
    template <class TInterval>
    inline CSIR_Level restrict_to_domain(const CSIR_Level& set,
                                         const samurai::LevelCellArray<2, TInterval>& domain_lca,
                                         std::size_t target_level)
    {
        auto set_on_l     = project_to_level(set, target_level);
        auto dom_csir     = to_csir_level(domain_lca);
        auto dom_on_l     = project_to_level(dom_csir, target_level);
        return intersection(set_on_l, dom_on_l);
    }

    // Restrict a set to a domain LCA at a target level (3D)
    template <class TInterval>
    inline CSIR_Level_3D restrict_to_domain(const CSIR_Level_3D& set,
                                            const samurai::LevelCellArray<3, TInterval>& domain_lca,
                                            std::size_t target_level)
    {
        auto set_on_l     = project_to_level(set, target_level);
        auto dom_csir     = to_csir_level(domain_lca);
        auto dom_on_l     = project_to_level(dom_csir, target_level);
        return intersection(set_on_l, dom_on_l);
    }

    // Convenience overloads for dimension-generic calls
    inline CSIR_Level_3D project_to_level(const CSIR_Level_3D& source, std::size_t target_level)
    {
        return project_to_level_3d(source, target_level);
    }

    inline CSIR_Level_3D intersection(const CSIR_Level_3D& a, const CSIR_Level_3D& b)
    {
        return intersection_3d(a, b);
    }

    inline CSIR_Level_1D contract(const CSIR_Level_1D& set, std::size_t width, const std::array<bool, 1>& mask)
    {
        if (width == 0 || (!mask[0])) return set;
        return contract(set, width);
    }

    // ----------------- Samurai interop (LCA <-> CSIR) -----------------
    // 1D
    template <class TInterval>
    inline CSIR_Level_1D to_csir_level(const samurai::LevelCellArray<1, TInterval>& lca)
    {
        CSIR_Level_1D out;
        out.level = lca.level();
        if (lca.empty()) return out;
        const auto& x_intervals = lca[0];
        out.intervals.reserve(x_intervals.size());
        for (const auto& itv : x_intervals)
        {
            out.intervals.push_back({itv.start, itv.end});
        }
        return out;
    }

    inline samurai::LevelCellArray<1> from_csir_level(const CSIR_Level_1D& level_1d)
    {
        samurai::LevelCellArray<1> out(level_1d.level);
        using value_t = typename samurai::LevelCellArray<1>::value_t;
        xt::xtensor_fixed<value_t, xt::xshape<0>> yz; // empty yz for 1D
        for (const auto& itv : level_1d.intervals)
        {
            out.add_interval_back({itv.start, itv.end}, yz);
        }
        return out;
    }

    // Geometry-aware overloads: preserve origin/scaling of the mesh
    template <class Origin, class Scale>
    inline samurai::LevelCellArray<1> from_csir_level(const CSIR_Level_1D& level_1d, const Origin& origin, const Scale& scaling)
    {
        samurai::LevelCellArray<1> out(level_1d.level, origin, scaling);
        using value_t = typename samurai::LevelCellArray<1>::value_t;
        xt::xtensor_fixed<value_t, xt::xshape<0>> yz;
        for (const auto& itv : level_1d.intervals)
        {
            out.add_interval_back({itv.start, itv.end}, yz);
        }
        return out;
    }

    // 2D
    template <class TInterval>
    inline CSIR_Level to_csir_level(const samurai::LevelCellArray<2, TInterval>& lca)
    {
        CSIR_Level result;
        result.level = lca.level();
        if (lca.empty()) { result.intervals_ptr.push_back(0); return result; }

        result.intervals_ptr.reserve(lca.shape()[1] + 1);
        result.intervals_ptr.push_back(0);

        bool have_row = false;
        int current_y = 0;
        std::size_t before = 0;

        for (auto it = lca.cbegin(); it != lca.cend(); ++it)
        {
            int y = it.index()[0];
            if (!have_row)
            {
                have_row = true;
                current_y = y;
                before = result.intervals.size();
            }
            else if (y != current_y)
            {
                if (result.intervals.size() > before)
                {
                    result.y_coords.push_back(current_y);
                    result.intervals_ptr.push_back(result.intervals.size());
                }
                current_y = y;
                before = result.intervals.size();
            }

            const auto& itv = *it;
            result.intervals.push_back({itv.start, itv.end});
        }
        // flush last row
        if (have_row && result.intervals.size() > result.intervals_ptr.back())
        {
            result.y_coords.push_back(current_y);
            result.intervals_ptr.push_back(result.intervals.size());
        }
        return result;
    }

    inline samurai::LevelCellArray<2> from_csir_level(const CSIR_Level& level_2d)
    {
        samurai::LevelCellArray<2> out(level_2d.level);
        using value_t = typename samurai::LevelCellArray<2>::value_t;
        xt::xtensor_fixed<value_t, xt::xshape<1>> yz;
        for (std::size_t ri = 0; ri < level_2d.y_coords.size(); ++ri)
        {
            yz[0] = level_2d.y_coords[ri];
            auto s = level_2d.intervals_ptr[ri];
            auto e = level_2d.intervals_ptr[ri + 1];
            for (std::size_t k = s; k < e; ++k)
            {
                const auto& itv = level_2d.intervals[k];
                out.add_interval_back({itv.start, itv.end}, yz);
            }
        }
        return out;
    }

    template <class Origin, class Scale>
    inline samurai::LevelCellArray<2> from_csir_level(const CSIR_Level& level_2d, const Origin& origin, const Scale& scaling)
    {
        samurai::LevelCellArray<2> out(level_2d.level, origin, scaling);
        using value_t = typename samurai::LevelCellArray<2>::value_t;
        xt::xtensor_fixed<value_t, xt::xshape<1>> yz;
        for (std::size_t ri = 0; ri < level_2d.y_coords.size(); ++ri)
        {
            yz[0] = level_2d.y_coords[ri];
            auto s = level_2d.intervals_ptr[ri];
            auto e = level_2d.intervals_ptr[ri + 1];
            for (std::size_t k = s; k < e; ++k)
            {
                const auto& itv = level_2d.intervals[k];
                out.add_interval_back({itv.start, itv.end}, yz);
            }
        }
        return out;
    }

    // 3D
    template <class TInterval>
    inline CSIR_Level_3D to_csir_level(const samurai::LevelCellArray<3, TInterval>& lca)
    {
        CSIR_Level_3D out;
        out.level = lca.level();
        if (lca.empty()) return out;

        // Build slices by z (k)
        struct RowAccum { int y; std::size_t before; bool active; RowAccum(): y(0), before(0), active(false) {} };
        std::map<int, RowAccum> accums;

        int current_k = std::numeric_limits<int>::min();
        for (auto it = lca.cbegin(); it != lca.cend(); ++it)
        {
            int y = it.index()[0];
            int z = it.index()[1];
            auto& slice = out.slices[z];
            if (slice.intervals_ptr.empty()) slice.intervals_ptr.push_back(0);

            // detect z change
            if (z != current_k)
            {
                // flush any pending row of previous z
                auto it_acc = accums.find(current_k);
                if (it_acc != accums.end() && it_acc->second.active)
                {
                    auto& prev_slice = out.slices[current_k];
                    if (prev_slice.intervals.size() > it_acc->second.before)
                    {
                        prev_slice.y_coords.push_back(it_acc->second.y);
                        prev_slice.intervals_ptr.push_back(prev_slice.intervals.size());
                    }
                    it_acc->second.active = false;
                }
                current_k = z;
                accums[z] = RowAccum{};
            }

            auto& row = accums[z];
            if (!row.active)
            {
                row.active = true;
                row.y = y;
                row.before = slice.intervals.size();
            }
            else if (y != row.y)
            {
                if (slice.intervals.size() > row.before)
                {
                    slice.y_coords.push_back(row.y);
                    slice.intervals_ptr.push_back(slice.intervals.size());
                }
                row.y = y;
                row.before = slice.intervals.size();
            }

            const auto& itv = *it;
            slice.intervals.push_back({itv.start, itv.end});
        }
        // flush last row of last slice
        auto it_acc = accums.find(current_k);
        if (it_acc != accums.end() && it_acc->second.active)
        {
            auto& slice = out.slices[current_k];
            if (slice.intervals.size() > it_acc->second.before)
            {
                slice.y_coords.push_back(it_acc->second.y);
                slice.intervals_ptr.push_back(slice.intervals.size());
            }
        }
        return out;
    }

    inline samurai::LevelCellArray<3> from_csir_level(const CSIR_Level_3D& level_3d)
    {
        samurai::LevelCellArray<3> out(level_3d.level);
        using value_t = typename samurai::LevelCellArray<3>::value_t;
        xt::xtensor_fixed<value_t, xt::xshape<2>> yz;
        for (const auto& [z, slice] : level_3d.slices)
        {
            yz[1] = z;
            for (std::size_t ri = 0; ri < slice.y_coords.size(); ++ri)
            {
                yz[0] = slice.y_coords[ri];
                auto s = slice.intervals_ptr[ri];
                auto e = slice.intervals_ptr[ri + 1];
                for (std::size_t k = s; k < e; ++k)
                {
                    const auto& itv = slice.intervals[k];
                    out.add_interval_back({itv.start, itv.end}, yz);
                }
            }
        }
        return out;
    }

    template <class Origin, class Scale>
    inline samurai::LevelCellArray<3> from_csir_level(const CSIR_Level_3D& level_3d, const Origin& origin, const Scale& scaling)
    {
        samurai::LevelCellArray<3> out(level_3d.level, origin, scaling);
        using value_t = typename samurai::LevelCellArray<3>::value_t;
        xt::xtensor_fixed<value_t, xt::xshape<2>> yz;
        for (const auto& [z, slice] : level_3d.slices)
        {
            yz[1] = z;
            for (std::size_t ri = 0; ri < slice.y_coords.size(); ++ri)
            {
                yz[0] = slice.y_coords[ri];
                auto s = slice.intervals_ptr[ri];
                auto e = slice.intervals_ptr[ri + 1];
                for (std::size_t k = s; k < e; ++k)
                {
                    const auto& itv = slice.intervals[k];
                    out.add_interval_back({itv.start, itv.end}, yz);
                }
            }
        }
        return out;
    }
} // namespace csir
