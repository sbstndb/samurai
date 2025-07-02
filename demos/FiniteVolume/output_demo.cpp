// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#include <samurai/samurai.hpp>
#include <samurai/io/output.hpp>
#include <thread>
#include <chrono>
#include <random>
#ifdef SAMURAI_WITH_MPI
#include <boost/mpi.hpp>
#endif

int main(int argc, char* argv[])
{
    // Initialisation de Samurai
    samurai::initialize("Démonstration des sorties MPI", argc, argv);

#ifdef SAMURAI_WITH_MPI
    mpi::communicator world;
#endif

    // Simulation d'une boucle de calcul
    for (int iteration = 0; iteration < 5; ++iteration)
    {
        // Message seulement depuis le rang 0
        samurai::output::print("=== Itération {}/5 ===\n", iteration + 1);

        // Simulation d'un calcul
        double local_result = 1.0 + iteration * 0.5;
        // On ajoute une petite part aléatoire pour différencier les rangs
#ifdef SAMURAI_WITH_MPI
        std::mt19937 gen(static_cast<unsigned int>(world.rank() + 123u * iteration));
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        local_result += dist(gen);
#else
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        local_result += dist(gen);
#endif
        
        // Message depuis tous les rangs avec préfixe
        samurai::output::print_all("Résultat local: {:.3f}\n", local_result);

        // Affichage du maximum et du minimum globaux
        samurai::output::print_max(local_result, "Max résultat global: {:.3f}\n");
        samurai::output::print_min(local_result, "Min résultat global: {:.3f}\n");
        samurai::output::print_sum(local_result, "Somme globale: {:.3f}\n");

        // Simulation d'une condition d'erreur
        if (iteration == 2)
        {
            samurai::output::print_error("Condition d'erreur simulée à l'itération {}\n", iteration);
        }

        // Message depuis un rang spécifique (rang 1 si il existe)
#ifdef SAMURAI_WITH_MPI
        if (world.size() > 1)
        {
            samurai::output::print(1, "Message spécial depuis le rang 1\n");
        }
#endif

        // Pause pour voir les messages
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // Message final
    samurai::output::print("Démonstration terminée\n");

    samurai::finalize();
    return 0;
} 