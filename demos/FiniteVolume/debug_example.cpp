// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#include <samurai/samurai.hpp>

int main(int argc, char** argv)
{
    // Initialisation de Samurai
    samurai::initialize("Debug Example Production", argc, argv);

    // Ces appels seront compilés ou non selon SAMURAI_DEBUG_ENABLED
    samurai::debug::initialize_debug("production_test");

    // Exemples d'utilisation du debug
    samurai::debug::debug_log("SAMURAI", "value=", 0);
    samurai::debug::debug_log("LOAD_BALANCING", "ncells = ", 4);
    samurai::debug::debug_log("MESH", "level=", 2, " cells=", 100);

    // Utilisation avec des variables
    int iteration = 5;
    double residual = 1.23e-8;
    std::string status = "converged";

    samurai::debug::debug_log("ITERATION", "step=", iteration, " residual=", residual, " status=", status);

    // Utilisation des variables même en mode production pour éviter les warnings
    (void)iteration;
    (void)residual;
    (void)status;

    // Plus d'exemples d'utilisation
    samurai::debug::debug_log("BOUNDARY_CONDITIONS", "type=periodic", " direction=x");
    samurai::debug::debug_log("TIME_STEPPING", "dt=", 0.001, " time=", 1.5);
    samurai::debug::debug_log("ADAPTATION", "refined_cells=", 15, " coarsened_cells=", 3);

    std::cout << "Debug example completed." << std::endl;
    std::cout << "Compilation info:" << std::endl;

#if SAMURAI_DEBUG_ENABLED
    std::cout << "- Debug system: ENABLED" << std::endl;
    std::cout << "- Check generated debug files" << std::endl;
#else
    std::cout << "- Debug system: DISABLED (production mode)" << std::endl;
    std::cout << "- No debug files generated" << std::endl;
    std::cout << "- Zero overhead" << std::endl;
#endif

    samurai::finalize();
    return 0;
} 