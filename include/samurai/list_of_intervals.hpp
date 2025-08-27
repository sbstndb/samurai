// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <algorithm>
#include <iostream>
#include <vector>

#include "interval.hpp"
#include "samurai_config.hpp"

namespace samurai
{

    ////////////////////////////////
    // ListOfIntervals definition //
    ////////////////////////////////

    /** @class ListOfIntervals
     *  @brief Ordered container of intervals.
     *
     * @tparam TValue  The coordinate type (must be signed).
     * @tparam TIndex  The index type (must be signed).
     */
    template <typename TValue, typename TIndex = default_config::index_t>
    struct ListOfIntervals : private std::vector<Interval<TValue, TIndex>>
    {
        using value_t    = TValue;
        using index_t    = TIndex;
        using interval_t = Interval<value_t, index_t>;

        using list_t = std::vector<interval_t>;
        using list_t::begin;
        using list_t::cbegin;
        using list_t::cend;
        using list_t::empty;
        using list_t::end;

        using const_iterator = typename list_t::const_iterator;
        using iterator       = typename list_t::iterator;
        using value_type     = typename list_t::value_type;

        std::size_t size() const;

        void add_point(value_t point);
        void add_interval(const interval_t& interval);
    };

    ////////////////////////////////////
    // ListOfIntervals implementation //
    ////////////////////////////////////

    /// Number of intervals stored in the list.
    template <typename TValue, typename TIndex>
    inline std::size_t ListOfIntervals<TValue, TIndex>::size() const
    {
        return list_t::size();
    }

    /// Add a point inside the list.
    template <typename TValue, typename TIndex>
    inline void ListOfIntervals<TValue, TIndex>::add_point(value_t point)
    {
        add_interval({point, point + 1});
    }

    /// Add an interval inside the list.
    template <typename TValue, typename TIndex>
    inline void ListOfIntervals<TValue, TIndex>::add_interval(const interval_t& interval)
    {
        if (!interval.is_valid())
        {
            return;
        }

        auto it = std::lower_bound(begin(),
                                   end(),
                                   interval,
                                   [](const auto& value, const auto& inter)
                                   {
                                       return value.end < inter.start;
                                   });

        if (it == end() || interval.end < it->start)
        {
            this->insert(it, interval);
            return;
        }

        it->start = std::min(it->start, interval.start);
        it->end   = std::max(it->end, interval.end);

        auto jt = it;
        ++jt;
        while (jt != end() && it->end >= jt->start)
        {
            it->end = std::max(it->end, jt->end);
            jt      = this->erase(jt);
        }
    }

    template <typename value_t, typename index_t>
    inline std::ostream& operator<<(std::ostream& out, const ListOfIntervals<value_t, index_t>& interval_list)
    {
        for (const auto& interval : interval_list)
        {
            out << interval << " ";
        }
        return out;
    }
} // namespace samurai
