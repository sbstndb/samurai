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
            using mesh_id_t = typename Mesh_t::mesh_id_t;
            boost::mpi::communicator world;

            // compute fluxes in terms of load to transfer/receive
            std::vector<double> fluxes = samurai::cmptFluxes<samurai::BalanceElement_t::CELL>(mesh, weight, 100);

            auto flags = samurai::make_scalar_field<int>("diffusion_flag", mesh);
            flags.fill(world.rank());

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

            // Tri des cellules du "haut" vers le "bas", puis de "gauche" à "droite"
            std::sort(cells.begin(),
                      cells.end(),
                      [](const cell_t& a, const cell_t& b)
                      {
                          auto ca = a.center();
                          auto cb = b.center();
                          if (ca(1) != cb(1))
                          {
                              return ca(1) > cb(1); // En premier, les cellules avec la plus grande coordonnée y
                          }
                          return ca(0) < cb(0); // Ensuite, les cellules avec la plus petite coordonnée x
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
    };
}
#endif
