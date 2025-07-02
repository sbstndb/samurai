#include "field.hpp"
#include "load_balancing.hpp"
#include "timers.hpp"

#include <algorithm>

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

            samurai::for_each_cell(mesh[mesh_id_t::cells],
                                   [&](auto cell)
                                   {
                                       cells.push_back(cell);
                                   });

            // Comparateur pour trier en priorité les cellules « en haut puis à gauche »
            auto comp_cells = [&](const cell_t& a, const cell_t& b)
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
            if (n_top > 0 && !cells.empty())
            {
                std::size_t k = std::min(n_top, cells.size());
                auto middle   = cells.begin() + static_cast<std::ptrdiff_t>(k);
                std::nth_element(cells.begin(), middle, cells.end(), comp_cells);
            }

            // Sélection des n_bottom cellules les plus « basses »
            if (n_bottom > 0 && !cells.empty())
            {
                std::size_t k = std::min(n_bottom, cells.size());
                auto middle   = cells.end() - static_cast<std::ptrdiff_t>(k);
                std::nth_element(cells.begin(), middle, cells.end(), comp_cells);
            }

            // Attribution des flags en fonction des flux calculés
            if (world.size() > 1)
            {
                if (world.rank() == 0)
                {
                    // Processus le plus « bas » : envoie n_top cellules au rang 1
                    if (fluxes[0] < 0)
                    {
                        std::size_t k = std::min(n_top, cells.size());
                        for (std::size_t i = 0; i < k; ++i)
                        {
                            flags[cells[i]] = 1;
                        }
                    }
                }
                else if (world.rank() == world.size() - 1)
                {
                    // Dernier processus : envoie n_bottom cellules au rang-1
                    if (fluxes[0] < 0)
                    {
                        std::size_t k = std::min(n_bottom, cells.size());
                        for (std::size_t i = 0; i < k; ++i)
                        {
                            flags[cells[cells.size() - k + i]] = world.rank() - 1;
                        }
                    }
                }
                else
                {
                    // Processus intérieur : deux échanges possibles
                    if (fluxes[0] < 0)
                    {
                        std::size_t k = std::min(n_bottom, cells.size());
                        for (std::size_t i = 0; i < k; ++i)
                        {
                            flags[cells[cells.size() - k + i]] = world.rank() - 1;
                        }
                    }

                    if (fluxes[1] < 0)
                    {
                        std::size_t k = std::min(n_top, cells.size());
                        for (std::size_t i = 0; i < k; ++i)
                        {
                            flags[cells[i]] = world.rank() + 1;
                        }
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
