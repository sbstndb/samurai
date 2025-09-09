#pragma once

#ifdef SAMURAI_WITH_MPI

#include <numeric>
#include <vector>

#include "../box.hpp"
#include "../field.hpp"
#include "../mesh.hpp"
#include "../subset/node.hpp"

namespace samurai
{
    /**
     * @brief Describes a contiguous segment of data to be exchanged.
     *
     * The plan is based on the mesh geometry and is independent of the number of components of the fields.
     */
    template <class TMesh>
    struct ExchangeSegment
    {
        using interval_t = typename TMesh::interval_t;
        using value_t    = typename interval_t::value_t;
        static constexpr std::size_t dim = TMesh::dim;
        using indices_t                  = xt::xtensor_fixed<value_t, xt::xshape<dim - 1>>;

        std::size_t level;
        interval_t interval;
        indices_t index;
    };

    /**
     * @brief Communication plan for a single neighbor process.
     *
     * It stores the list of segments to send and receive, and caches the total data size
     * to avoid re-computation. The size is in number of elements (e.g., doubles), not bytes.
     */
    template <class TMesh>
    struct NeighborExchangePlan
    {
        int rank = -1;
        std::vector<ExchangeSegment<TMesh>> send_segments;
        std::vector<ExchangeSegment<TMesh>> recv_segments;
        std::size_t total_send_size = 0; // in number of values
        std::size_t total_recv_size = 0; // in number of values
    };

    /**
     * @brief The complete exchange plan for all neighbors.
     *
     * This object is intended to be built once after each mesh adaptation and reused for
     * all subsequent ghost exchanges until the mesh changes again.
     */
    template <class TMesh>
    struct ExchangePlan
    {
        std::vector<NeighborExchangePlan<TMesh>> neighbors;
    };

    namespace detail
    {
        template <class Field>
        auto build_exchange_plan_impl(const Field& field) -> ExchangePlan<typename Field::mesh_t>
        {
            using mesh_t           = typename Field::mesh_t;
            using Config           = typename mesh_t::config;
            using mesh_id_t        = typename Config::mesh_id_t;
            using interval_t       = typename mesh_t::interval_t;
            using value_t          = typename interval_t::value_t;
            using lca_type         = typename mesh_t::lca_type;
            using box_t            = Box<value_t, Field::dim>;
            constexpr std::size_t dim = Field::dim;

            ExchangePlan<mesh_t> plan;
            auto& mesh = field.mesh();

            if (mesh.mpi_neighbourhood().empty())
            {
                return plan;
            }

            plan.neighbors.resize(mesh.mpi_neighbourhood().size());

            std::size_t i_neigh = 0;
            for (const auto& neighbour : mesh.mpi_neighbourhood())
            {
                plan.neighbors[i_neigh].rank = neighbour.rank;
                auto& neighbor_plan          = plan.neighbors[i_neigh];

                // Part 1: Direct subdomain interfaces and corners
                for (std::size_t level = mesh.min_level(); level <= mesh.max_level(); ++level)
                {
                    if (!mesh[mesh_id_t::reference][level].empty() && !neighbour.mesh[mesh_id_t::reference][level].empty())
                    {
                        // Segments to send
                        auto out_interface = intersection(mesh[mesh_id_t::reference][level],
                                                          neighbour.mesh[mesh_id_t::reference][level],
                                                          mesh.subdomain())
                                                 .on(level);
                        out_interface([&](const auto& i, const auto& index)
                                      {
                                          neighbor_plan.send_segments.push_back({level, i, index});
                                      });

                        auto send_corners = outer_subdomain_corner<true>(level, field, neighbour);
                        for_each_interval(send_corners,
                                          [&](auto, const auto& i, const auto& index)
                                          {
                                              neighbor_plan.send_segments.push_back({level, i, index});
                                          });

                        // Segments to receive
                        auto in_interface = intersection(neighbour.mesh[mesh_id_t::reference][level],
                                                         mesh[mesh_id_t::reference][level],
                                                         neighbour.mesh.subdomain())
                                                .on(level);
                        in_interface([&](const auto& i, const auto& index)
                                     {
                                         neighbor_plan.recv_segments.push_back({level, i, index});
                                     });

                        auto recv_corners = outer_subdomain_corner<false>(level, field, neighbour);
                        for_each_interval(recv_corners,
                                          [&](auto, const auto& i, const auto& index)
                                          {
                                              neighbor_plan.recv_segments.push_back({level, i, index});
                                          });
                    }
                }
                i_neigh++;
            }

            // Part 2: Periodic boundaries involving neighbors
            const auto& mesh_ref    = mesh[mesh_id_t::reference];
            const auto& domain      = mesh.domain();
            const auto& min_indices = domain.min_indices();
            const auto& max_indices = domain.max_indices();

            xt::xtensor_fixed<value_t, xt::xshape<dim>> shift;

            for (std::size_t d = 0; d < dim; ++d)
            {
                if (mesh.is_periodic(d))
                {
                    for (std::size_t level = mesh.min_level(); level <= mesh.max_level(); ++level)
                    {
                        const std::size_t delta_l = domain.level() - level;
                        shift.fill(0);
                        shift[d] = (max_indices[d] - min_indices[d]) >> delta_l;

                        xt::xtensor_fixed<value_t, xt::xshape<dim>> min_corner;
                        xt::xtensor_fixed<value_t, xt::xshape<dim>> max_corner;

                        for (std::size_t id = 0; id < dim; ++id)
                        {
                            min_corner[id] = (min_indices[id] >> delta_l) - Config::ghost_width;
                            max_corner[id] = (max_indices[id] >> delta_l) + Config::ghost_width;
                        }
                        // Define boundary boxes for periodic exchange detection
                        min_corner[d] = (min_indices[d] >> delta_l);
                        max_corner[d] = (min_indices[d] >> delta_l) + Config::ghost_width;
                        lca_type lca_min_p(level, box_t(min_corner, max_corner));

                        min_corner[d] = (max_indices[d] >> delta_l) - Config::ghost_width;
                        max_corner[d] = (max_indices[d] >> delta_l);
                        lca_type lca_max_m(level, box_t(min_corner, max_corner));

                        i_neigh = 0;
                        for (const auto& neighbour : mesh.mpi_neighbourhood())
                        {
                            auto& neighbor_plan           = plan.neighbors[i_neigh];
                            const auto& neighbor_mesh_ref = neighbour.mesh[mesh_id_t::reference];

                            // Data to send from me to neighbor for their periodic ghosts
                            auto send_set1 = intersection(translate(intersection(mesh_ref[level], lca_min_p), shift),
                                                          neighbor_mesh_ref[level]);
                            send_set1([&](const auto& i, const auto& index)
                                      {
                                          neighbor_plan.send_segments.push_back({level, i - shift[0], index - xt::view(shift, xt::range(1, _))});
                                      });
                            auto send_set2 = intersection(translate(intersection(mesh_ref[level], lca_max_m), -shift),
                                                          neighbor_mesh_ref[level]);
                            send_set2([&](const auto& i, const auto& index)
                                      {
                                          neighbor_plan.send_segments.push_back({level, i + shift[0], index + xt::view(shift, xt::range(1, _))});
                                      });

                            // Data to receive on my periodic ghosts from neighbor
                            auto recv_set1 = intersection(mesh_ref[level],
                                                          translate(intersection(neighbor_mesh_ref[level], lca_min_p), shift));
                            recv_set1([&](const auto& i, const auto& index)
                                      {
                                          neighbor_plan.recv_segments.push_back({level, i, index});
                                      });

                            auto recv_set2 = intersection(mesh_ref[level],
                                                          translate(intersection(neighbor_mesh_ref[level], lca_max_m), -shift));
                            recv_set2([&](const auto& i, const auto& index)
                                      {
                                          neighbor_plan.recv_segments.push_back({level, i, index});
                                      });
                            i_neigh++;
                        }
                    }
                }
            }

            // Part 3: Finalize plan by computing sizes and removing duplicates
            for (auto& neighbor_plan : plan.neighbors)
            {
                // Sort and remove duplicates
                std::sort(neighbor_plan.send_segments.begin(),
                          neighbor_plan.send_segments.end(),
                          [](const auto& a, const auto& b)
                          {
                              if (a.level != b.level)
                                  return a.level < b.level;
                              return a.interval.start < b.interval.start;
                          });
                neighbor_plan.send_segments.erase(std::unique(neighbor_plan.send_segments.begin(), neighbor_plan.send_segments.end()),
                                                  neighbor_plan.send_segments.end());

                std::sort(neighbor_plan.recv_segments.begin(),
                          neighbor_plan.recv_segments.end(),
                          [](const auto& a, const auto& b)
                          {
                              if (a.level != b.level)
                                  return a.level < b.level;
                              return a.interval.start < b.interval.start;
                          });
                neighbor_plan.recv_segments.erase(std::unique(neighbor_plan.recv_segments.begin(), neighbor_plan.recv_segments.end()),
                                                  neighbor_plan.recv_segments.end());

                // Compute total sizes
                neighbor_plan.total_send_size = std::accumulate(neighbor_plan.send_segments.begin(),
                                                                neighbor_plan.send_segments.end(),
                                                                std::size_t{0},
                                                                [](std::size_t sum, const auto& seg)
                                                                {
                                                                    return sum + seg.interval.size();
                                                                });
                neighbor_plan.total_recv_size = std::accumulate(neighbor_plan.recv_segments.begin(),
                                                                neighbor_plan.recv_segments.end(),
                                                                std::size_t{0},
                                                                [](std::size_t sum, const auto& seg)
                                                                {
                                                                    return sum + seg.interval.size();
                                                                });
            }

            return plan;
        }

        // Overload for multiple fields (variadic template)
        template <class Plan, class Field, class... Fields>
        void exchange_ghosts_all_levels_impl(const Plan& plan, Field& field, Fields&... other_fields)
        {
            mpi::communicator world;
            std::vector<mpi::request> reqs;

            // Prepare buffers and post non-blocking sends for all neighbors
            std::vector<std::vector<typename Field::value_type>> send_buffers(plan.neighbors.size());
            std::size_t i_neigh = 0;
            for (const auto& neighbor_plan : plan.neighbors)
            {
                send_buffers[i_neigh].reserve(neighbor_plan.total_send_size * Field::n_comp);
                for (const auto& seg : neighbor_plan.send_segments)
                {
                    auto field_data = field(seg.level, seg.interval, seg.index);
                    send_buffers[i_neigh].insert(send_buffers[i_neigh].end(), field_data.begin(), field_data.end());
                }
                reqs.push_back(world.isend(neighbor_plan.rank, 0, send_buffers[i_neigh]));
                i_neigh++;
            }

            // Prepare buffers and post non-blocking receives
            std::vector<std::vector<typename Field::value_type>> recv_buffers(plan.neighbors.size());
            i_neigh = 0;
            for (const auto& neighbor_plan : plan.neighbors)
            {
                recv_buffers[i_neigh].resize(neighbor_plan.total_recv_size * Field::n_comp);
                reqs.push_back(world.irecv(neighbor_plan.rank, 0, recv_buffers[i_neigh]));
                i_neigh++;
            }

            // Wait for all communications to complete
            mpi::wait_all(reqs.begin(), reqs.end());

            // Unpack received data into all fields
            auto unpack = [&](auto& f)
            {
                std::size_t i_n = 0;
                for (const auto& neighbor_plan : plan.neighbors)
                {
                    auto& buffer = recv_buffers[i_n];
                    auto it      = buffer.begin();
                    for (const auto& seg : neighbor_plan.recv_segments)
                    {
                        auto field_view = f(seg.level, seg.interval, seg.index);
                        std::copy(it, it + field_view.size(), field_view.begin());
                        it += field_view.size();
                    }
                    i_n++;
                }
            };

            unpack(field);
            (unpack(other_fields), ...);
        }

    } // namespace detail

    /**
     * @brief Builds a communication plan for ghost cell exchanges.
     *
     * This function analyzes the mesh and its neighborhood to determine which data segments
     * need to be exchanged with each neighbor. It covers both direct subdomain interfaces
     * and periodic boundaries.
     *
     * @tparam Field The type of the field, used to deduce mesh configuration.
     * @param field A field associated with the mesh.
     * @return An ExchangePlan object for the given mesh.
     */
    template <class Field>
    auto build_exchange_plan(const Field& field) -> ExchangePlan<typename Field::mesh_t>
    {
        return detail::build_exchange_plan_impl(field);
    }

    /**
     * @brief Performs an aggregated ghost cell exchange for one or more fields.
     *
     * This function executes the provided communication plan. It packs all required data
     * for each neighbor into a single buffer, performs non-blocking send/receive operations,
     * waits for completion, and then unpacks the received data into the ghost cells of the provided fields.
     *
     * @tparam Field The type of the first field.
     * @tparam Fields The types of any additional fields to be updated simultaneously.
     * @param plan The pre-computed ExchangePlan to execute.
     * @param field The first field to update.
     * @param other_fields Additional fields to update. They must share the same mesh.
     */
    template <class Field, class... Fields>
    void exchange_ghosts_all_levels(const ExchangePlan<typename Field::mesh_t>& plan, Field& field, Fields&... other_fields)
    {
        detail::exchange_ghosts_all_levels_impl(plan, field, other_fields...);
    }

    // Equality operators for segments, used for unique check
    template <class TMesh>
    bool operator==(const ExchangeSegment<TMesh>& a, const ExchangeSegment<TMesh>& b)
    {
        return a.level == b.level && a.interval == b.interval && a.index == b.index;
    }

} // namespace samurai

#endif // SAMURAI_WITH_MPI
