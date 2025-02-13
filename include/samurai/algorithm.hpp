// Copyright 2018-2024 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#ifdef SAMURAI_WITH_OPENMP
#include <omp.h>
#endif
#include <type_traits>

#include "cell.hpp"
#include "mesh_holder.hpp"
#include "mesh_interval.hpp"

namespace samurai
{
    template <std::size_t dim_, class TInterval, std::size_t max_size_>
    class CellArray;

    template <std::size_t dim_, class TInterval>
    class LevelCellArray;

    template <class D, class Config>
    class Mesh_base;

    template <class F, class... CT>
    class subset_operator;

    enum class Run
    {
        Sequential,
        Parallel
    };

    enum class Get
    {
        Cells,
        Intervals
    };

    ///////////////////////////////////
    // for_each_level implementation //
    ///////////////////////////////////

    template <std::size_t dim, class TInterval, std::size_t max_size, class Func>
    inline void for_each_level(const CellArray<dim, TInterval, max_size>& ca, Func&& f, bool include_empty_levels = false)
    {
        for (std::size_t level = ca.min_level(); level <= ca.max_level(); ++level)
        {
            if (include_empty_levels || !ca[level].empty())
            {
                f(level);
            }
        }
    }

    template <class Mesh, class Func>
    inline void for_each_level(Mesh& mesh, Func&& f, bool include_empty_levels = false)
    {
        using mesh_id_t = typename Mesh::mesh_id_t;
        for_each_level(mesh[mesh_id_t::cells], std::forward<Func>(f), include_empty_levels);
    }

    //////////////////////////////////////
    // for_each_interval implementation //
    //////////////////////////////////////

    template <std::size_t dim, class TInterval, class Func>
    inline void for_each_interval(const LevelCellArray<dim, TInterval>& lca, Func&& f)
    {
        if (!lca.empty())
        {
            for (auto it = lca.cbegin(); it != lca.cend(); ++it)
            {
                f(lca.level(), *it, it.index());
            }
        }
    }

    template <std::size_t dim, class TInterval, class Func>
    inline void for_each_interval(LevelCellArray<dim, TInterval>& lca, Func&& f)
    {
        if (!lca.empty())
        {
            for (auto it = lca.begin(); it != lca.end(); ++it)
            {
                f(lca.level(), *it, it.index());
            }
        }
    }

    template <std::size_t dim, class TInterval, std::size_t max_size, class Func>
    inline void for_each_interval(const CellArray<dim, TInterval, max_size>& ca, Func&& f)
    {
        for (std::size_t level = ca.min_level(); level <= ca.max_level(); ++level)
        {
            for_each_interval(ca[level], std::forward<Func>(f));
        }
    }

    template <std::size_t dim, class TInterval, std::size_t max_size, class Func>
    inline void for_each_interval(CellArray<dim, TInterval, max_size>& ca, Func&& f)
    {
        for (std::size_t level = ca.min_level(); level <= ca.max_level(); ++level)
        {
            for_each_interval(ca[level], std::forward<Func>(f));
        }
    }

    template <class Mesh, class Func>
    inline void for_each_interval(const Mesh& mesh, Func&& f)
    {
        using mesh_id_t = typename Mesh::config::mesh_id_t;
        for_each_interval(mesh[mesh_id_t::cells], std::forward<Func>(f));
    }

    template <class F, class... CT>
    class subset_operator;

    template <class Func, class F, class... CT>
    inline void for_each_interval(subset_operator<F, CT...>& set, Func&& f)
    {
        set(
            [&](const auto& i, const auto& index)
            {
                f(set.level(), i, index);
            });
    }

    //////////////////////////////////////////
    // for_each_meshinterval implementation //
    //////////////////////////////////////////

    template <std::size_t dim, class TInterval, class Func>
    inline void for_each_meshinterval(const LevelCellArray<dim, TInterval>& lca, Func&& f)
    {
        using MeshInterval = typename LevelCellArray<dim, TInterval>::mesh_interval_t;

        for (auto it = lca.cbegin(); it != lca.cend(); ++it)
        {
            f(MeshInterval(lca.level(), *it, it.index()));
        }
    }

    template <std::size_t dim, class TInterval, std::size_t max_size, class Func>
    inline void for_each_meshinterval(const CellArray<dim, TInterval, max_size>& ca, Func&& f)
    {
        for (std::size_t level = ca.min_level(); level <= ca.max_level(); ++level)
        {
            if (!ca[level].empty())
            {
                for_each_meshinterval(ca[level], std::forward<Func>(f));
            }
        }
    }

    template <class MeshIntervalType, class SetType, class Func>
    inline void for_each_meshinterval(SetType& set, Func&& f)
    {
        MeshIntervalType mesh_interval(set.level());
        set(
            [&](const auto& i, const auto& index)
            {
                mesh_interval.i     = i;
                mesh_interval.index = index;
                f(mesh_interval);
            });
    }

    template <class MeshIntervalType, class SetType, class Func>
    inline void parallel_for_each_meshinterval(SetType& set, Func&& f)
    {
#pragma omp parallel
#pragma omp single nowait
        set(
            [&](const auto& i, const auto& index)
            {
#pragma omp task
                {
                    MeshIntervalType mesh_interval(set.level());
                    mesh_interval.i     = i;
                    mesh_interval.index = index;
                    f(mesh_interval);
                }
            });
    }

    template <class MeshIntervalType, Run run_type, class SetType, class Func>
    inline void for_each_meshinterval(SetType& set, Func&& f)
    {
        if constexpr (run_type == Run::Parallel)
        {
            parallel_for_each_meshinterval<MeshIntervalType>(set, std::forward<Func>(f));
        }
        else
        {
            for_each_meshinterval<MeshIntervalType>(set, std::forward<Func>(f));
        }
    }

    //////////////////////////////////
    // for_each_cell implementation //
    //////////////////////////////////

    template <std::size_t dim, class TInterval, class Func>
    inline void for_each_cell(const LevelCellArray<dim, TInterval>& lca, Func&& f)
    {
        using cell_t        = Cell<dim, TInterval>;
        using index_value_t = typename cell_t::value_t;
        typename cell_t::indices_t index;

        for (auto it = lca.cbegin(); it != lca.cend(); ++it)
        {
            for (std::size_t d = 0; d < dim - 1; ++d)
            {
                index[d + 1] = it.index()[d];
            }

            for (index_value_t i = it->start; i < it->end; ++i)
            {
                index[0] = i;
                cell_t cell{lca.origin_point(), lca.scaling_factor(), lca.level(), index, it->index + i};
                f(cell);
            }
        }
    }

    template <std::size_t dim, class TInterval, class Func>
    inline void parallel_for_each_cell(const LevelCellArray<dim, TInterval>& lca, Func&& f)
    {
        using cell_t        = Cell<dim, TInterval>;
        using index_value_t = typename cell_t::value_t;

#pragma omp parallel
#pragma omp single nowait
        {
            for (auto it = lca.cbegin(); it != lca.cend(); ++it)
            {
#pragma omp task
                for (index_value_t i = it->start; i < it->end; ++i)
                {
                    typename cell_t::indices_t index;
                    for (std::size_t d = 0; d < dim - 1; ++d)
                    {
                        index[d + 1] = it.index()[d];
                    }
                    index[0] = i;
                    cell_t cell{lca.origin_point(), lca.scaling_factor(), lca.level(), index, it->index + i};
                    f(cell);
                }
            }
        }
    }

    template <Run run_type, std::size_t dim, class TInterval, class Func>
    inline void for_each_cell(const LevelCellArray<dim, TInterval>& lca, Func&& f)
    {
        if constexpr (run_type == Run::Parallel)
        {
            parallel_for_each_cell(lca, std::forward<Func>(f));
        }
        else
        {
            for_each_cell(lca, std::forward<Func>(f));
        }
    }

    template <std::size_t dim, class TInterval, class Func, class F, class... CT>
    inline void for_each_cell(const LevelCellArray<dim, TInterval>& lca, subset_operator<F, CT...> set, Func&& f)
    {
        using cell_t        = Cell<dim, TInterval>;
        using index_value_t = typename cell_t::value_t;
        typename cell_t::indices_t index;

        set(
            [&](const auto& interval, const auto& index_yz)
            {
                index[0]                         = interval.start;
                auto cell_index                  = lca.get_index(index);
                xt::view(index, xt::range(1, _)) = index_yz;
                for (index_value_t i = interval.start; i < interval.end; ++i)
                {
                    index[0] = i;
                    cell_t cell{lca.origin_point(), lca.scaling_factor(), set.level(), index, cell_index++};
                    f(cell);
                }
            });
    }

    template <Run run_type, std::size_t dim, class TInterval, std::size_t max_size, class Func>
    inline void for_each_cell(const CellArray<dim, TInterval, max_size>& ca, Func&& f)
    {
        for (std::size_t level = ca.min_level(); level <= ca.max_level(); ++level)
        {
            if (!ca[level].empty())
            {
                for_each_cell<run_type>(ca[level], std::forward<Func>(f));
            }
        }
    }

    template <std::size_t dim, class TInterval, std::size_t max_size, class Func>
    inline void for_each_cell(const CellArray<dim, TInterval, max_size>& ca, Func&& f)
    {
        for_each_cell<Run::Sequential>(ca, std::forward<Func>(f));
    }

    template <Run run_type, class Mesh, class Func>
    inline void for_each_cell(const Mesh& mesh, Func&& f)
    {
        using mesh_id_t = typename Mesh::mesh_id_t;
        for_each_cell<run_type>(mesh[mesh_id_t::cells], std::forward<Func>(f));
    }

    template <class Mesh, class Func>
    inline void for_each_cell(const Mesh& mesh, Func&& f)
    {
        for_each_cell<Run::Sequential>(mesh, std::forward<Func>(f));
    }

    template <class Mesh, class Func>
    inline void for_each_cell(const hold<Mesh>& mesh, Func&& f)
    {
        for_each_cell(mesh.get(), std::forward<Func>(f));
    }

    template <class Mesh, class coord_type, class Func>
    inline void for_each_cell(const Mesh& mesh, std::size_t level, const typename Mesh::interval_t& i, const coord_type& index, Func&& f)
    {
        static constexpr std::size_t dim = Mesh::dim;
        using cell_t                     = Cell<dim, typename Mesh::interval_t>;
        using index_value_t              = typename cell_t::value_t;
        typename cell_t::indices_t coord;

        coord[0] = i.start;
        for (std::size_t d = 0; d < dim - 1; ++d)
        {
            coord[d + 1] = index[d];
        }
        auto cell_index = mesh.get_index(level, coord);
        cell_t cell{mesh.origin_point(), mesh.scaling_factor(), level, coord, cell_index};
        for (index_value_t ii = 0; ii < static_cast<index_value_t>(i.size()); ++ii)
        {
            f(cell);
            cell.indices[0]++; // increment x coordinate
            cell.index++;      // increment cell index
        }
    }

    template <class Mesh, class SetType, class Func>
    inline void for_each_cell(const Mesh& mesh, SetType& set, Func&& f)
    {
        set(
            [&](const auto& i, const auto& index)
            {
                for_each_cell(mesh, set.level(), i, index, std::forward<Func>(f));
            });
    }

    /////////////////////////
    // find implementation //
    /////////////////////////

    namespace detail
    {
        // template <class ForwardIt, class T>
        // auto interval_search(ForwardIt first, ForwardIt last, const T& value)
        // {
        //     auto comp = [](const auto& interval, auto v)
        //     {
        //         return interval.end < v;
        //     };

        //     auto result = std::lower_bound(first, last, value, comp);

        //     if (!(result == last) && !(comp(*result, value)))
        //     {
        //         if (result->contains(value))
        //         {
        //             return static_cast<int>(std::distance(first, result));
        //         }
        //     }
        //     return -1;
        // }

        template <class ForwardIt, class T>
        inline auto interval_search(ForwardIt first, ForwardIt last, const T& value)
        {
            for (int dist = 0; first != last; ++first, ++dist)
            {
                if (first->contains(value))
                {
                    return dist;
                }
            }
            return -1;
        }

// ... existing code ...

	/**
template <class Container, class T>
inline auto interval_search(const Container& intervals, std::size_t start_index, std::size_t end_index, const T& value) 
{
    // Comparateur simple pour vérifier si la valeur est dans l'intervalle
    auto comp = [](const auto& interval, auto v) {
        return interval.end <= v;
    };

    auto comp2 = [](const auto& interval, auto v) {
        return interval.start > v;
    };
    // Recherche linéaire dans la plage spécifiée
    for (std::size_t i = start_index; i < end_index; i+=2) {
//        if (!comp(intervals[i], value) & (!comp2(intervals[i], value))){
	if (intervals[i].start <= value){
		if (intervals[i].end > value){
	        	return static_cast<int>(i - start_index);
		}
        }
        if (intervals[i+1].start <= value){
                if (intervals[i+1].end > value){
                        return static_cast<int>(i+1 - start_index);
                }
        }
    }
    return -1;
}
**/


#include <immintrin.h>
#include <cstddef>

template <class Container, class T>
inline auto interval_search(const Container& intervals, std::size_t start_index, std::size_t end_index, const T& value)
{
    // On suppose que 'value' peut être converti en int.
    for (std::size_t i = start_index; i < end_index; ++i) {
        const auto& interval = intervals[i];

        // Charger les valeurs 'start' et 'end' dans des registres SSE scalaires.
        __m128i start_reg = _mm_cvtsi32_si128(interval.start);
        __m128i end_reg   = _mm_cvtsi32_si128(interval.end);
        __m128i value_reg = _mm_cvtsi32_si128(static_cast<int>(value));

        // Vérifier la condition : interval.start <= value < interval.end
        // Pour "start <= value", on teste !(start > value)
        __m128i cmp_start = _mm_cmpgt_epi32(start_reg, value_reg); // renvoie 0xFFFFFFFF si start > value, sinon 0.
        // Pour "value < end", on vérifie que end > value.
        __m128i cmp_end = _mm_cmpgt_epi32(end_reg, value_reg);       // renvoie 0xFFFFFFFF si end > value, sinon 0.

        // Extraire le résultat des comparaisons depuis le registre (la valeur dans les 32 bits inférieurs)
        int res_start = _mm_cvtsi128_si32(cmp_start); // 0 si start <= value
        int res_end   = _mm_cvtsi128_si32(cmp_end);   // non nul si end > value

        // Si l'intervalle correspond, retourner l'indice relatif (par rapport à start_index)
        if (res_start == 0 && res_end != 0) {
            return static_cast<int>(i - start_index);
        }
    }
    return -1;
}


#include <immintrin.h>
#include <cstddef>

/**
// Exemple de structure d'intervalle
struct Interval {
    int start;      // Interval start.
    int end;        // Interval end + 1.
    int step;       // Step inside the interval.
    long long index; // Storage index so that interval's content starts at index + start.
};
**/
/**

template <class Container, class T>
inline int interval_search(const Container& intervals, std::size_t start_index, std::size_t end_index, const T& value) {
    // Nombre total d'intervalles à parcourir
    const int n = static_cast<int>(end_index - start_index);
    constexpr int vec_size = 8; // 8 entiers (32 bits) par __m256i

    // On travaille sur les champs start et end qui sont des int.
    // On va utiliser le gather sur un tableau d'int obtenu en considérant que les Interval sont stockés en mémoire de manière contiguë.
    using IntervalType = typename Container::value_type;
    const int* base = reinterpret_cast<const int*>(&intervals[start_index]);
    // La distance (en int) entre deux intervalles est:
    constexpr int stride = sizeof(IntervalType) / sizeof(int);
    // On suppose que 'start' est au décalage 0 et 'end' à 1 (en int)
    constexpr int offset_start = 0;
    constexpr int offset_end   = 1;

    // Préparation d'un vecteur contenant la valeur recherchée (pour la comparaison)
    __m256i value_vec = _mm256_set1_epi32(static_cast<int>(value));
    int i = 0;
    for (; i <= n - vec_size; i += vec_size) {
        // Calcul des indices pour charger les champs 'start'
        __m256i indices_start = _mm256_setr_epi32(
            i * stride + offset_start,
            (i + 1) * stride + offset_start,
            (i + 2) * stride + offset_start,
            (i + 3) * stride + offset_start,
            (i + 4) * stride + offset_start,
            (i + 5) * stride + offset_start,
            (i + 6) * stride + offset_start,
            (i + 7) * stride + offset_start
        );
        __m256i starts = _mm256_i32gather_epi32(base, indices_start, 1);

        // Calcul des indices pour charger les champs 'end'
        __m256i indices_end = _mm256_setr_epi32(
            i * stride + offset_end,
            (i + 1) * stride + offset_end,
            (i + 2) * stride + offset_end,
            (i + 3) * stride + offset_end,
            (i + 4) * stride + offset_end,
            (i + 5) * stride + offset_end,
            (i + 6) * stride + offset_end,
            (i + 7) * stride + offset_end
        );
        __m256i ends = _mm256_i32gather_epi32(base, indices_end, 1);

        // Conditions à vérifier pour chaque intervalle :
        //   (start <= value) et (end > value)
        // Pour start <= value, on calcule ~(start > value)
        __m256i cmp_start = _mm256_cmpgt_epi32(starts, value_vec); // renvoie 0xFFFFFFFF si start > value
        __m256i cond_start = _mm256_andnot_si256(cmp_start, _mm256_set1_epi32(-1)); // ~cmp_start

        // Pour end > value
        __m256i cond_end = _mm256_cmpgt_epi32(ends, value_vec);

        // On combine les conditions
        __m256i valid = _mm256_and_si256(cond_start, cond_end);

        // On teste si au moins un intervalle du bloc satisfait la condition.
        int mask = _mm256_movemask_epi8(valid);
        if (mask != 0) {
            // Un ou plusieurs éléments du vecteur sont valides.
            // On récupère le vecteur dans un tableau pour trouver le premier indice.
            alignas(32) int valid_array[vec_size];
            _mm256_store_si256(reinterpret_cast<__m256i*>(valid_array), valid);
            for (int j = 0; j < vec_size; ++j) {
                if (valid_array[j] != 0) {
                    return i + j; // indice relatif à start_index
                }
            }
        }
    }
    // Boucle scalaire pour les éléments restants (si n n'est pas un multiple de 8)
    for (; i < n; i++) {
        const IntervalType& inter = intervals[start_index + i];
        if (inter.start <= value && inter.end > value)
            return i;
    }
    return -1;
}
**/


// ... existing code ...




        // template <class ForwardIt, class T>
        // inline auto interval_search(ForwardIt first, ForwardIt last, const T& value)
        // {
        //     auto it = std::find_if(first,
        //                            last,
        //                            [value](const auto& e)
        //                            {
        //                                return e.contains(value);
        //                            });
        //     return (it == last) ? -1 : static_cast<int>(std::distance(first, it));
        // }

        template <std::size_t dim, class TInterval, class index_t = typename TInterval::index_t, class coord_index_t = typename TInterval::coord_index_t>
        inline auto find_impl(const LevelCellArray<dim, TInterval>& lca,
                              std::size_t start_index,
                              std::size_t end_index,
                              const xt::xtensor_fixed<coord_index_t, xt::xshape<dim>>& coord,
                              std::integral_constant<std::size_t, 0>) -> index_t
        {
            using lca_t     = const LevelCellArray<dim, TInterval>;
            using diff_t    = typename lca_t::const_iterator::difference_type;

            index_t find_index = interval_search(lca[0], start_index, end_index, coord[0]);


            return (find_index != -1) ? find_index + static_cast<diff_t>(start_index) : find_index;
        }

        template <std::size_t dim,
                  class TInterval,
                  class index_t       = typename TInterval::index_t,
                  class coord_index_t = typename TInterval::coord_index_t,
                  std::size_t N>
        inline auto find_impl(const LevelCellArray<dim, TInterval>& lca,
                              std::size_t start_index,
                              std::size_t end_index,
                              const xt::xtensor_fixed<coord_index_t, xt::xshape<dim>>& coord,
                              std::integral_constant<std::size_t, N>) -> index_t
        {
            using lca_t        = const LevelCellArray<dim, TInterval>;
            using diff_t       = typename lca_t::const_iterator::difference_type;
            auto find_index = interval_search(lca[N], start_index, end_index,  coord[N]);

            if (find_index != -1)
            {
                auto off_ind = static_cast<std::size_t>(lca[N][static_cast<std::size_t>(find_index) + start_index].index + coord[N]);
                find_index   = find_impl(lca,
                                       lca.offsets(N)[off_ind],
                                       lca.offsets(N)[off_ind + 1],
                                       coord,
                                       std::integral_constant<std::size_t, N - 1>{});
                find_index   = find_impl(lca,
                                       lca.offsets(N)[off_ind],
                                       lca.offsets(N)[off_ind + 1],
                                       coord,
                                       std::integral_constant<std::size_t, N - 1>{});


            }
            return find_index;
        }
    } // namespace detail

    template <std::size_t dim, class TInterval, class index_t = typename TInterval::index_t, class coord_index_t = typename TInterval::coord_index_t>
    inline auto find(const LevelCellArray<dim, TInterval>& lca, const xt::xtensor_fixed<coord_index_t, xt::xshape<dim>>& coord) -> index_t
    {
        return detail::find_impl(lca, 0, lca[dim - 1].size(), coord, std::integral_constant<std::size_t, dim - 1>{});
    }

    template <std::size_t dim, class TInterval, class coord_index_t = typename TInterval::coord_index_t, class index_t = typename TInterval::index_t>
    inline auto
    find_on_dim(const LevelCellArray<dim, TInterval>& lca, std::size_t d, std::size_t start_index, std::size_t end_index, coord_index_t coord)
    {
        using lca_t        = const LevelCellArray<dim, TInterval>;
        using diff_t       = typename lca_t::const_iterator::difference_type;
        index_t find_index = detail::interval_search(lca[d].cbegin() + static_cast<diff_t>(start_index),
                                                     lca[d].cbegin() + static_cast<diff_t>(end_index),
                                                     coord);

        return (find_index != -1) ? static_cast<std::size_t>(find_index) + start_index : std::numeric_limits<std::size_t>::max();
    }

} // namespace samurai
