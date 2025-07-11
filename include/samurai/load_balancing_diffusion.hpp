#include "field.hpp"
#include "load_balancing.hpp"
#include "timers.hpp"

#include <algorithm>
#include <iterator>

#ifdef SAMURAI_WITH_MPI
namespace samurai
{
    /**
     * Diffusion-based load balancer implementation
     * Encapsulates flux computation and cell distribution logic
     */
    class DiffusionLoadBalancer : public samurai::LoadBalancer<DiffusionLoadBalancer>
    {
      public:

        DiffusionLoadBalancer(int max_iterations = 50, double diffusion_factor = 0.5)
            : m_max_iterations(max_iterations)
            , m_diffusion_factor(diffusion_factor)
        {
        }

        /**
         * Compute fluxes based on load computing strategy using graph with label
         * propagation algorithm. Returns, for the current process, the flux in terms of
         * load, i.e. the quantity of "load" to transfer to its neighbours. If the load
         * is negative, it means that the process (current) must send load to neighbour,
         * if positive it means that it must receive load.
         *
         * This function uses 2 MPI all_gather calls.
         */
        template <samurai::BalanceElement_t elem, class Mesh_t, class Field_t>
        std::vector<double> compute_fluxes(Mesh_t& mesh, const Field_t& weight)
        {
            samurai::times::timers.start("load_balancing_flux_computation");
            
            using mpi_subdomain_t = typename Mesh_t::mpi_subdomain_t;
            boost::mpi::communicator world;
            std::vector<mpi_subdomain_t>& neighbourhood = mesh.mpi_neighbourhood();
            size_t n_neighbours                         = neighbourhood.size();

            // Load of current process
            double my_load = samurai::cmptLoad<elem>(mesh, weight);
            // Fluxes between processes
            std::vector<double> fluxes(n_neighbours, 0.);
            // Load of each process (all processes not only neighbours)
            std::vector<double> loads;
            int iteration_count = 0;
            
            while (iteration_count < m_max_iterations)
            {
                boost::mpi::all_gather(world, my_load, loads);

                // Compute updated my_load for current process based on its neighbourhood
                double my_load_new = my_load;
                bool all_fluxes_zero = true;
                
                for (std::size_t neighbour_idx = 0; neighbour_idx < n_neighbours; ++neighbour_idx)
                {
                    std::size_t neighbour_rank = static_cast<std::size_t>(neighbourhood[neighbour_idx].rank);
                    double neighbour_load         = loads[neighbour_rank];
                    double diff_load = neighbour_load - my_load_new;

                    // If transferLoad < 0 -> need to send data, if transferLoad > 0 need to receive data
                    // Use diffusion factor for stability
                    double transfertLoad = m_diffusion_factor * diff_load;

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
            
            samurai::times::timers.stop("load_balancing_flux_computation");
            
            return fluxes;
        }

        template <class Mesh_t, class Weight_t>
        auto load_balance_impl(Mesh_t& mesh, const Weight_t& weight)
        {
            using mesh_id_t = typename Mesh_t::mesh_id_t;
            boost::mpi::communicator world;

            auto flags = samurai::make_scalar_field<int>("diffusion_flag", mesh);
            flags.fill(world.rank());

            // Compute fluxes in terms of load to transfer/receive
            std::vector<double> fluxes = compute_fluxes<samurai::BalanceElement_t::CELL>(mesh, weight);

            using cell_t = typename Mesh_t::cell_t;
            std::vector<cell_t> cells;
            samurai::for_each_cell(mesh[mesh_id_t::cells],
                                   [&](auto cell)
                                   {
                                       cells.emplace_back(cell);
                                   });

            if (cells.empty())
            {
                return flags;
            }

            // Sort cells from "top" to "bottom", then from "left" to "right"
            std::sort(cells.begin(),
                      cells.end(),
                      [](const cell_t& a, const cell_t& b)
                      {
                          auto center_a = a.center();
                          auto center_b = b.center();
                          if (center_a(1) != center_b(1))
                          {
                              return center_a(1) > center_b(1); // First, cells with highest y coordinate
                          }
                          return center_a(0) < center_b(0); // Then, cells with lowest x coordinate
                      });

            auto& neighbourhood = mesh.mpi_neighbourhood();
            auto cell_it        = cells.begin();
            auto cell_rit       = cells.rbegin();

            for (std::size_t i = 0; i < neighbourhood.size(); ++i)
            {
                double flux            = fluxes[i];
                auto neighbour_rank = neighbourhood[i].rank;

                if (flux < 0) // We must send cells
                {
                    double weight_to_send   = -flux;
                    double accumulated_weight = 0;

                    // Send from the "top" to higher ranks, and from the "bottom" to lower ranks
                    if (neighbour_rank > world.rank())
                    {
                        while (cell_it != cell_rit.base() && accumulated_weight < weight_to_send)
                        {
                            accumulated_weight += weight[*cell_it];
                            flags[*cell_it] = neighbour_rank;
                            cell_it++;
                        }
                    }
                    else
                    {
                        while (cell_rit.base() != cell_it && accumulated_weight < weight_to_send)
                        {
                            accumulated_weight += weight[*cell_rit];
                            flags[*cell_rit] = neighbour_rank;
                            cell_rit++;
                        }
                    }
                }
            }
            return flags;
        }

      private:
        int m_max_iterations;
        double m_diffusion_factor;
    };
}
#endif
