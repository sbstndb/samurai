#pragma once

#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

#include "algorithm.hpp"
#include "algorithm/utils.hpp"
#include "mesh.hpp"
#include "mr/mesh.hpp"
#include "timers.hpp"

#ifdef SAMURAI_WITH_MPI
#include <boost/mpi.hpp>
#endif

#ifdef SAMURAI_WITH_MPI
namespace samurai
{
    enum BalanceElement_t
    {
        CELL
    };

    namespace weight
    {
        template <class Field>
        auto from_field(const Field& f)
        {
            auto weight = samurai::make_scalar_field<double>("weight", f.mesh());
            weight.fill(0.);
            samurai::for_each_cell(f.mesh(),
                                   [&](auto cell)
                                   {
                                       weight[cell] = 1.0 + std::abs(f[cell]);
                                   });
            return weight;
        }

        template <class Mesh>
        auto uniform(const Mesh& mesh)
        {
            using mesh_id_t = typename Mesh::mesh_id_t;
            auto weight     = samurai::make_scalar_field<double>("weight", mesh);
            weight.fill(0.);

            samurai::for_each_cell(mesh[mesh_id_t::cells],
                                   [&](auto cell)
                                   {
                                       weight[cell] = 1.0;
                                   });
            return weight;
        }
    }

    /**
     * Compute the load of the current process based on intervals or cells. It uses the
     * mesh_id_t::cells to only consider leaves.
     */
    template <BalanceElement_t elem, class Mesh_t, class Field_t>
    static double cmptLoad(const Mesh_t& mesh, const Field_t& weight)
    {
        using mesh_id_t                  = typename Mesh_t::mesh_id_t;
        const auto& current_mesh         = mesh[mesh_id_t::cells];
        double current_process_load = 0.;
        // cell-based load with weight.
        samurai::for_each_cell(current_mesh,
                                   [&](const auto& cell)
                                   {
                                       current_process_load += weight[cell];
                                   });
        return current_process_load;
    }

    /**
     * Compute fluxes based on load computing strategy using graph with label
     * propagation algorithm. Returns, for the current process, the flux in terms of
     * load, i.e. the quantity of "load" to transfer to its neighbours. If the load
     * is negative, it means that the process (current) must send load to neighbour,
     * if positive it means that it must receive load.
     *
     * This function uses 2 MPI all_gather calls.
     *
     */
    template <BalanceElement_t elem, class Mesh_t, class Field_t>
    std::vector<double> cmptFluxes(Mesh_t& mesh, const Field_t& weight, int niterations)
    {
        // Start timer for flux computation
        samurai::times::timers.start("load_balancing_flux_computation");
        
        using mpi_subdomain_t = typename Mesh_t::mpi_subdomain_t;
        boost::mpi::communicator world;
        // Give access to geometrically neighbouring process rank and mesh
        std::vector<mpi_subdomain_t>& neighbourhood = mesh.mpi_neighbourhood();
        size_t n_neighbours                         = neighbourhood.size();

        // Load of current process
        double my_load = cmptLoad<elem>(mesh, weight);
        // Fluxes between processes
        std::vector<double> fluxes(n_neighbours, 0.);
        // Load of each process (all processes not only neighbours)
        std::vector<double> loads;
        int iteration_count = 0;
        while (iteration_count < niterations)
        {
            boost::mpi::all_gather(world, my_load, loads);

            // Compute updated my_load for current process based on its neighbourhood
            double my_load_new = my_load;
            bool all_fluxes_zero = true;
            for (std::size_t neighbour_idx = 0; neighbour_idx < n_neighbours; ++neighbour_idx)
            // Get "my_load" from other processes
            {
                std::size_t neighbour_rank = static_cast<std::size_t>(neighbourhood[neighbour_idx].rank);
                double neighbour_load         = loads[neighbour_rank];
                double diff_load = neighbour_load - my_load_new;

                // If transferLoad < 0 -> need to send data, if transferLoad > 0 need to receive data
                // Use diffusion factor 1/(deg+1) for stability
                double transfertLoad = 0.5* diff_load;


                // Accumulate total flux on current edge
                fluxes[neighbour_idx] += transfertLoad;

                // Mark if a non-zero transfer was performed
                if (transfertLoad != 0)
                {
                    all_fluxes_zero = false;
                }

                // Update intermediate local load before processing next neighbour
                my_load_new += transfertLoad;
            }
            
            // Update reference load for next iteration
            my_load = my_load_new;

            // Check if all processes have reached convergence
            bool global_convergence = boost::mpi::all_reduce(world, all_fluxes_zero, std::logical_and<bool>());

            // If all processes have zero fluxes, state will no longer change
            if (global_convergence)
            {
                std::cout << "Process " << world.rank() << " : Global convergence reached at iteration " << iteration_count << std::endl;
                break;
            }
            
            iteration_count++;
        }
        
        // Stop timer for flux computation
        samurai::times::timers.stop("load_balancing_flux_computation");
        
        return fluxes;
    }

    template <class Flavor>
    class LoadBalancer
    {
      public:

        int nloadbalancing;

        template <class Mesh_t, class Field_t>
        void update_field(Mesh_t& new_mesh, Field_t& field)
        {
            using mesh_id_t = typename Mesh_t::mesh_id_t;
            using value_t   = typename Field_t::value_type;
            boost::mpi::communicator world;

            Field_t new_field("new_f", new_mesh);
            new_field.fill(0);

            auto& old_mesh = field.mesh();
            auto min_level = old_mesh.min_level();
            auto max_level = old_mesh.max_level();

            // Copy data of intervals that didn't move
            for (std::size_t level = min_level; level <= max_level; ++level)
            {
                auto intersect_old_new = intersection(old_mesh[mesh_id_t::cells][level], new_mesh[mesh_id_t::cells][level]);
                intersect_old_new.apply_op(samurai::copy(new_field, field));
            }

            std::vector<boost::mpi::request> req, reqs;
            std::vector<std::vector<value_t>> to_send(static_cast<size_t>(world.size()));

            // Here we have to define all_* at size n_neighbours...
            std::vector<Mesh_t> all_new_meshes, all_old_meshes;
            Mesh_t recv_old_mesh, recv_new_mesh;
            for (auto& neighbour : new_mesh.mpi_neighbourhood())
            {
                reqs.push_back(world.isend(neighbour.rank, 0, new_mesh));
                reqs.push_back(world.isend(neighbour.rank, 1, old_mesh));

                world.recv(neighbour.rank, 0, recv_new_mesh);
                world.recv(neighbour.rank, 1, recv_old_mesh);

                all_new_meshes.push_back(recv_new_mesh);
                all_old_meshes.push_back(recv_old_mesh);
            }
            boost::mpi::wait_all(reqs.begin(), reqs.end());

            // Build payload of field that has been sent to neighbour, so compare old mesh with new neighbour mesh
            for (size_t neighbour_idx = 0; neighbour_idx < all_new_meshes.size(); ++neighbour_idx)
            {
                auto& neighbour_new_mesh = all_new_meshes[neighbour_idx];

                for (std::size_t level = min_level; level <= max_level; ++level)
                {
                    if (!old_mesh[mesh_id_t::cells][level].empty() && !neighbour_new_mesh[mesh_id_t::cells][level].empty())
                    {
                        auto intersect_old_mesh_new_neigh = intersection(old_mesh[mesh_id_t::cells][level],
                                                                         neighbour_new_mesh[mesh_id_t::cells][level]);
                        intersect_old_mesh_new_neigh(
                            [&](const auto& interval, const auto& index)
                            {
                                std::copy(field(level, interval, index).begin(),
                                          field(level, interval, index).end(),
                                          std::back_inserter(to_send[neighbour_idx]));
                            });
                    }
                }

                if (to_send[neighbour_idx].size() != 0)
                {
                    auto neighbour_rank = new_mesh.mpi_neighbourhood()[neighbour_idx].rank;
                    req.push_back(world.isend(neighbour_rank, neighbour_rank, to_send[neighbour_idx]));
                }
            }

            // Build payload of field that I need to receive from neighbour, so compare NEW mesh with OLD neighbour mesh
            for (size_t neighbour_idx = 0; neighbour_idx < all_old_meshes.size(); ++neighbour_idx)
            {
                bool isintersect = false;
                for (std::size_t level = min_level; level <= max_level; ++level)
                {
                    if (!new_mesh[mesh_id_t::cells][level].empty() && !all_old_meshes[neighbour_idx][mesh_id_t::cells][level].empty())
                    {
                        std::vector<value_t> to_recv;

                        auto in_interface = intersection(all_old_meshes[neighbour_idx][mesh_id_t::cells][level], new_mesh[mesh_id_t::cells][level]);

                        in_interface(
                            [&]([[maybe_unused]] const auto& i, [[maybe_unused]] const auto& index)
                            {
                                isintersect = true;
                            });

                        if (isintersect)
                        {
                            break;
                        }
                    }
                }

                if (isintersect)
                {
                    std::ptrdiff_t count = 0;
                    std::vector<value_t> to_recv;
                    world.recv(new_mesh.mpi_neighbourhood()[neighbour_idx].rank, world.rank(), to_recv);

                    for (std::size_t level = min_level; level <= max_level; ++level)
                    {
                        if (!new_mesh[mesh_id_t::cells][level].empty() && !all_old_meshes[neighbour_idx][mesh_id_t::cells][level].empty())
                        {
                            auto in_interface = intersection(all_old_meshes[neighbour_idx][mesh_id_t::cells][level], new_mesh[mesh_id_t::cells][level]);

                            in_interface(
                                [&](const auto& i, const auto& index)
                                {
                                    std::copy(to_recv.begin() + count,
                                              to_recv.begin() + count + static_cast<ptrdiff_t>(i.size() * field.n_comp),
                                              new_field(level, i, index).begin());
                                    count += static_cast<ptrdiff_t>(i.size() * field.n_comp);
                                });
                        }
                    }
                }
            }

            if (!req.empty())
            {
                mpi::wait_all(req.begin(), req.end());
            }

            std::swap(field.array(), new_field.array());
        }

        template <class Mesh_t, class Field_t, class... Fields_t>
        void update_fields(Mesh_t& new_mesh, Field_t& field, Fields_t&... kw)

        {
            update_field(new_mesh, field);
            update_fields(new_mesh, kw...);
        }

        template <class Mesh_t>
        void update_fields([[maybe_unused]] Mesh_t& new_mesh)
        {
        }

      public:

        LoadBalancer()
        {
            boost::mpi::communicator world;
            nloadbalancing = 0;
        }

        template <class Mesh_t, class Field_t>
        Mesh_t update_mesh(Mesh_t& mesh, const Field_t& flags)
        {
            // Démarrer le timer pour la mise à jour du maillage
            samurai::times::timers.start("load_balancing_mesh_update");
            
            using CellList_t  = typename Mesh_t::cl_type;
            using CellArray_t = typename Mesh_t::ca_type;

            boost::mpi::communicator world;

            CellList_t new_cl;
            std::vector<CellList_t> payload(static_cast<size_t>(world.size()));

            // Build cell list for the current process && cells lists of cells for other processes
            samurai::for_each_cell(
                mesh[Mesh_t::mesh_id_t::cells],
                [&](const auto& cell)
                {
                    if (flags[cell] == world.rank())
                    {
                        if constexpr (Mesh_t::dim == 1)
                        {
                            new_cl[cell.level][{}].add_point(cell.indices[0]);
                        }
                        if constexpr (Mesh_t::dim == 2)
                        {
                            new_cl[cell.level][{cell.indices[1]}].add_point(cell.indices[0]);
                        }
                        if constexpr (Mesh_t::dim == 3)
                        {
                            new_cl[cell.level][{cell.indices[1], cell.indices[2]}].add_point(cell.indices[0]);
                        }
                    }
                    else
                    {
                        assert(static_cast<size_t>(flags[cell]) < payload.size());

                        if constexpr (Mesh_t::dim == 1)
                        {
                            payload[static_cast<size_t>(flags[cell])][cell.level][{}].add_point(cell.indices[0]);
                        }
                        if constexpr (Mesh_t::dim == 2)
                        {
                            payload[static_cast<size_t>(flags[cell])][cell.level][{cell.indices[1]}].add_point(cell.indices[0]);
                        }
                        if constexpr (Mesh_t::dim == 3)
                        {
                            payload[static_cast<size_t>(flags[cell])][cell.level][{cell.indices[1], cell.indices[2]}].add_point(
                                cell.indices[0]);
                        }
                    }
                });

            std::vector<mpi::request> req;

            // Actual data exchange between processes that need to exchange data
            for (int process_rank = 0; process_rank < world.size(); ++process_rank)
            {
                if (process_rank == world.rank())
                {
                    continue;
                }
                CellArray_t to_send = {payload[static_cast<size_t>(process_rank)], false};
                req.push_back(world.isend(process_rank, 17, to_send));
            }

            for (int process_rank = 0; process_rank < world.size(); ++process_rank)
            {
                if (process_rank == world.rank())
                {
                    continue;
                }
                CellArray_t to_rcv;
                world.recv(process_rank, 17, to_rcv);

                samurai::for_each_interval(to_rcv,
                                           [&](std::size_t level, const auto& interval, const auto& index)
                                           {
                                               new_cl[level][index].add_interval(interval);
                                           });
            }
            boost::mpi::wait_all(req.begin(), req.end());
            Mesh_t new_mesh(new_cl, mesh);
            
            // Stop timer for mesh update
            samurai::times::timers.stop("load_balancing_mesh_update");
            
            return new_mesh;
        }

        template <class Mesh_t, class Weight_t, class Field_t, class... Fields>
        void load_balance(Mesh_t& mesh, Weight_t& weight, Field_t& field, Fields&... kw)
        {
            // Early check: no load balancing with single process
            boost::mpi::communicator world;
            if (world.size() <= 1)
            {
                std::cout << "Process " << world.rank() << " : Single MPI process detected, load balancing ignored" << std::endl;
                return;
            }

            // Start timer for load balancing
            samurai::times::timers.start("load_balancing");

            // Compute flags for this single pass
            auto flags = static_cast<Flavor*>(this)->load_balance_impl(mesh, weight);

            // Update mesh
            auto new_mesh = update_mesh(mesh, flags);

            // Update physical fields
            update_fields(new_mesh, weight, field, kw...);

            // Replace reference mesh
            mesh.swap(new_mesh);

            nloadbalancing += 1;

            // Stop timer
            samurai::times::timers.stop("load_balancing");

            // Final display of cell count after load balancing
            {
                using mesh_id_t = typename Mesh_t::mesh_id_t;
                double total_weight = cmptLoad<BalanceElement_t::CELL>(field.mesh(), weight);
                auto nb_cells = field.mesh().nb_cells(mesh_id_t::cells);
                std::cout << "Process " << world.rank() << " : " << nb_cells << " cells (total weight " << total_weight << ") after load balancing" << std::endl;
            }
        }
    };

} // namespace samurai
#endif
