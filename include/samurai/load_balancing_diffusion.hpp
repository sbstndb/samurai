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

        template <class Mesh_t>
        auto load_balance_impl(Mesh_t& mesh)
        {
            // Démarrer le timer pour l'algorithme de diffusion
            samurai::times::timers.start("load_balancing_diffusion_algorithm");
            
            using mesh_id_t = typename Mesh_t::mesh_id_t;

            boost::mpi::communicator world;

            // compute fluxes in terms of number of intervals to transfer/receive
            std::vector<int> fluxes = samurai::cmptFluxes<samurai::BalanceElement_t::CELL>(mesh, 100);

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

            const std::size_t cells_size = cells.size();

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

            // Nombre de cellules à envoyer/recevoir vers chaque voisin
            std::size_t n_top    = 0; // portion « en haut »
            std::size_t n_bottom = 0; // portion « en bas »

            if (world.size() > 1)
            {
                if (world.rank() == 0)
                {
                    if (fluxes[0] < 0)
                    {
                        n_top = static_cast<std::size_t>(std::abs(fluxes[0]));
                    }
                }
                else if (world.rank() == world.size() - 1)
                {
                    if (fluxes[0] < 0)
                    {
                        n_bottom = static_cast<std::size_t>(std::abs(fluxes[0]));
                    }
                }
                else
                {
                    if (fluxes[0] < 0)
                    {
                        n_bottom = static_cast<std::size_t>(std::abs(fluxes[0])); // vers rang-1
                    }
                    if (fluxes[1] < 0)
                    {
                        n_top = static_cast<std::size_t>(std::abs(fluxes[1])); // vers rang+1
                    }
                }
            }

            // Sélection des n_top cellules les plus « hautes »
            if (n_top > 0)
            {
                std::size_t k = std::min(n_top, cells_size);
                auto middle   = cells.begin() + static_cast<std::ptrdiff_t>(k);
                std::nth_element(cells.begin(), middle, cells.end(), comp_cells);
            }

            // Sélection des n_bottom cellules les plus « basses »
            if (n_bottom > 0)
            {
                std::size_t k = std::min(n_bottom, cells_size);
                auto middle   = cells.end() - static_cast<std::ptrdiff_t>(k);
                std::nth_element(cells.begin(), middle, cells.end(), comp_cells);
            }

            // Attribution des flags en fonction des flux calculés
            if (world.size() > 1)
            {
                // Fonction helper pour attribuer les flags
                auto assign_flags = [&](std::size_t count, std::size_t start_idx, int target_rank)
                {
                    if (count > 0)
                    {
                        std::size_t k = std::min(count, cells_size);
                        for (std::size_t i = 0; i < k; ++i)
                        {
                            flags[cells[start_idx + i]] = target_rank;
                        }
                    }
                };

                // Fonction helper pour traiter un flux négatif
                auto process_negative_flux = [&](int flux_idx, std::size_t count, std::size_t start_idx, int target_rank)
                {
                    if (fluxes[flux_idx] < 0)
                    {
                        assign_flags(count, start_idx, target_rank);
                    }
                };

                if (world.rank() == 0)
                {
                    // Processus le plus « bas » : envoie n_top cellules au rang 1
                    process_negative_flux(0, n_top, 0, 1);
                }
                else if (world.rank() == world.size() - 1)
                {
                    // Dernier processus : envoie n_bottom cellules au rang-1
                    process_negative_flux(0, n_bottom, cells_size - n_bottom, world.rank() - 1);
                }
                else
                {
                    // Processus intérieur : deux échanges possibles
                    process_negative_flux(0, n_bottom, cells_size - n_bottom, world.rank() - 1);
                    process_negative_flux(1, n_top, 0, world.rank() + 1);
                }
            }

            // Arrêter le timer pour l'algorithme de diffusion
            samurai::times::timers.stop("load_balancing_diffusion_algorithm");
            
            return flags;
        }
    };
}
#endif
