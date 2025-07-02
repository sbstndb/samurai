#include "field.hpp"
#include "load_balancing.hpp"
#include "timers.hpp"

#include <algorithm>
#include <iterator>

#ifdef SAMURAI_WITH_MPI
namespace Load_balancing
{

    class Diffusion : public samurai::LoadBalancer<Diffusion>
    {
      public:

        Diffusion() = default;

        template <class Mesh_t, class Weight_t>
        auto load_balance_impl(Mesh_t& mesh, const Weight_t& weight)
        {
            // Démarrer le timer pour l'algorithme de diffusion
            samurai::times::timers.start("load_balancing_diffusion_algorithm");
            
            using mesh_id_t = typename Mesh_t::mesh_id_t;

            boost::mpi::communicator world;

            // compute fluxes in terms of number of intervals to transfer/receive
            std::vector<double> fluxes = samurai::cmptFluxes<samurai::BalanceElement_t::CELL>(mesh, weight, 100);

            // set field "flags" for each rank. Initialized to current for all cells (leaves only)
            auto flags = samurai::make_scalar_field<int>("diffusion_flag", mesh);
            flags.fill(world.rank());

            using cell_t = typename Mesh_t::cell_t;
            std::vector<cell_t> cells;
            cells.reserve(mesh.nb_cells(mesh_id_t::cells));

            // Collecter toutes les cellules
            samurai::for_each_cell(mesh[mesh_id_t::cells],
                                   [&](auto cell)
                                   {
                                       cells.emplace_back(cell);
                                   });

            // Si pas de cellules, retourner directement
            if (cells.empty())
            {
                samurai::times::timers.stop("load_balancing_diffusion_algorithm");
                return flags;
            }

            // Comparateur pour trier en priorité les cellules « en haut puis à gauche »
            const auto comp_cells = [&](const cell_t& a, const cell_t& b)
            {
                auto ca = a.center();
                auto cb = b.center();
                if (ca(1) != cb(1))
                {
                    return ca(1) > cb(1); // plus haut en premier
                }
                else
                {
                    return ca(0) > cb(0); // puis plus à gauche
                }
            };
            std::sort(cells.begin(), cells.end(), comp_cells);

            // Poids à envoyer/recevoir vers chaque voisin
            double weight_to_send_top    = 0.; // vers rang+1
            double weight_to_send_bottom = 0.; // vers rang-1

            if (world.size() > 1)
            {
                if (world.rank() == 0)
                {
                    if (fluxes[0] < 0)
                    {
                        weight_to_send_top = -fluxes[0];
                    }
                }
                else if (world.rank() == world.size() - 1)
                {
                    if (fluxes[0] < 0)
                    {
                        weight_to_send_bottom = -fluxes[0];
                    }
                }
                else
                {
                    if (fluxes[0] < 0)
                    {
                        weight_to_send_bottom = -fluxes[0]; // vers rang-1
                    }
                    if (fluxes[1] < 0)
                    {
                        weight_to_send_top = -fluxes[1]; // vers rang+1
                    }
                }
            }

            // Marquer les cellules à envoyer "en haut"
            if (weight_to_send_top > 0)
            {
                double a_weight = 0.;
                for (std::size_t i = 0; i < cells.size(); ++i)
                {
                    if (a_weight < weight_to_send_top)
                    {
                        a_weight += weight[cells[i]];
                        flags[cells[i]] = world.rank() + 1;
                    }
                    else
                    {
                        break;
                    }
                }
            }

            // Marquer les cellules à envoyer "en bas"
            if (weight_to_send_bottom > 0)
            {
                double a_weight = 0.;
                for (std::size_t i = 0; i < cells.size(); ++i)
                {
                    auto& cell = cells[cells.size() - 1 - i];
                    if (a_weight < weight_to_send_bottom)
                    {
                        a_weight += weight[cell];
                        flags[cell] = world.rank() - 1;
                    }
                    else
                    {
                        break;
                    }
                }
            }

            // Arrêter le timer pour l'algorithme de diffusion
            samurai::times::timers.stop("load_balancing_diffusion_algorithm");
            
            return flags;
        }
    };
}
#endif
