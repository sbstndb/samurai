#pragma once

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <vector>

#include <tuple>
#include <utility>

#include <samurai/mesh.hpp>
#include <samurai/mr/mesh.hpp>

namespace samurai
{

    // print and info functions
    template <class Mesh_t>
    void printLocalLoad(const Mesh_t& mesh)
    {
        mpi::communicator world;
        std::size_t local_cells = mesh.nb_cells();
        std::cout << "subdomain Cells for rank " << world.rank() << " : " << local_cells << std::endl;
    }

    template <class Mesh_t>
    std::vector<std::size_t> gatherLocalLoad(const Mesh_t& mesh)
    {
        // Permet d'obtenir le vecteur de charge pour chaque sous-domaine
        mpi::communicator world;
        std::size_t local_cells = mesh.nb_cells();
        std::vector<std::size_t> distributedCountCells(world.size());
        mpi::all_gather(world, local_cells, distributedCountCells);
        return distributedCountCells;
    }

    template <class Mesh_t>
    std::tuple<double, double> getInbalance(const Mesh_t& mesh)
    {
        std::vector<std::size_t> distributedCountCells = gatherLocalLoad(mesh);

        auto minmax_iterators = std::minmax_element(distributedCountCells.begin(), distributedCountCells.end());

        // Pour obtenir les valeurs, on déréférence les itérateurs retournés
        double min_value = *minmax_iterators.first;  // .first pointe vers le minimum
        double max_value = *minmax_iterators.second; // .second pointe vers le maximum
        return std::make_tuple(min_value, max_value);
    }

    template <class Mesh_t>
    void printInbalance(const Mesh_t& mesh)
    {
        auto minmax_values = getInbalance(mesh); // min and max
        std::cout << " Inbalance : " << std::get<1>(minmax_values) / std::get<0>(minmax_values) << std::endl;
    }

    // Functions that exchange cells

    template <class Mesh_t>
    void send_10_cells(Mesh_t& mesh)
    {
        // aim : send 10 cells from level0 to level 1
        // This is just an experimental function
        // Please run advection-2d with 2 ranks for now.
        //
        using CellList_t  = typename Mesh_t::cl_type;
        using CellArray_t = typename Mesh_t::ca_type;
        mpi::communicator world;

        CellList_t to_keep_cl, to_send_cl;

        int num_cell = 0;
        samurai::for_each_cell(mesh[Mesh_t::mesh_id_t::cells],
                               [&](const auto& cell)
                               {
                                   if (num_cell < 1)
                                   {
                                       to_send_cl[cell.level][cell.indices[1]].add_point(cell.indices[0]);
                                   }
                                   else
                                   {
                                       to_keep_cl[cell.level][cell.indices[1]].add_point(cell.indices[0]);
                                   }
                                   num_cell++;
                               });

        samurai::CellArray<2> to_send_ca, to_recv_ca;
        to_send_ca = samurai::CellArray<2>(to_send_cl, true);
        //      to_recv_ca = samurai::CellArray<2>(to_recv_cl, true);

        Mesh_t to_keep_mesh, to_send_mesh, to_recv_mesh;
        to_keep_mesh = {to_keep_cl, mesh};
        to_send_mesh = {to_send_cl, mesh};
        //      std::cout << "so_send_mesh : " << to_send_mesh << std::endl ;
        std::cout << "size of to_send_ca : " << to_send_ca.nb_cells() << std::endl;
        //                std::cout << "size of to_keep_mesh : " << to_keep_mesh.nb_cells() << std::endl ;

        mpi::request req;

        if (world.rank() == 0)
        {
            // send 0 to 1
            req = world.isend(1, 1, to_send_ca);
            //          req = world.isend(1, 1, to_send_mesh) ;
        }
        else if (world.rank() == 1)
        {
            // recv 1 from 0
            world.recv(0, 1, to_recv_ca);
            //          world.recv(0, 1, to_recv_mesh) ;
        }
        req.wait();
        std::cout << "size of to_recv_ca : " << to_recv_ca.nb_cells() << std::endl;

        // The new mesh is
        //  for rank 0 : mesh - to_send_mesh::cells
        //  for rank 1 : mesh + to_recv_mesh::cells

        using config_t = typename Mesh_t::config;
        //      samurai::MRMesh<config_t> new_mesh;
        for (int level = 0; level < 15; level++)
        {
            if (world.rank() == 0)
            {
                auto restriction_cells                = samurai::difference(mesh[Mesh_t::mesh_id_t::cells][level], to_send_ca[level]);
                mesh[Mesh_t::mesh_id_t::cells][level] = restriction_cells;
            }
            else if (world.rank() == 1)
            {
                auto union_cells                      = samurai::union_(mesh[Mesh_t::mesh_id_t::cells][level], to_recv_ca[level]);
                mesh[Mesh_t::mesh_id_t::cells][level] = union_cells;
            }
        }
        std::cout << " mesh transfert ended for rank : " << std::endl;

        //      samurai::MRMesh<config_t> new_mesh{mesh[Mesh_t::mesh_id_t::cells], mesh };
        mesh.update_sub_mesh_impl();
        std::cout << " mesh transfert2 ended for rank : " << std::endl;

        //      using config_t = typename Mesh_t::config;
        //      samurai::MRMesh<config_t> new_mesh{ mesh[Mesh_t::mesh_id_t::cells], mesh };
        // Échange (swap) du nouvel objet maillage avec le maillage courant
        //          mesh.swap(new_mesh);
        std::cout << " mesh update ended for rank : " << std::endl;

        // enregistrement du maillage
        //    samurai::save(".", fmt::format("{}_size_{}{}", "new_cells", world.size(), ""), new_mesh);
        //    samurai::save(".", fmt::format("{}_size_{}{}", "new_cells_and_ghosts", world.size(), ""),
        //    new_mesh[Mesh_t::mesh_id_t::cells_and_ghosts]); samurai::save(".", fmt::format("{}_size_{}{}", "new_reference", world.size(),
        //    ""), new_mesh[Mesh_t::mesh_id_t::reference]);

        samurai::save(".", fmt::format("{}_size_{}{}", "cells", world.size(), ""), mesh);
        samurai::save(".", fmt::format("{}_size_{}{}", "cells_and_ghosts", world.size(), ""), mesh[Mesh_t::mesh_id_t::cells_and_ghosts]);
        samurai::save(".", fmt::format("{}_size_{}{}", "reference", world.size(), ""), mesh[Mesh_t::mesh_id_t::reference]);

        // Copie des valeurs flottantes
    }

    // diffusion algorithm that define the new boundaries of the current subdomain

    //

}
