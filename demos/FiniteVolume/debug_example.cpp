// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#include <samurai/samurai.hpp>

int main(int argc, char** argv)
{
    samurai::initialize("Debug Example Production", argc, argv);

    samurai::debug::initialize_debug("production_test");

    samurai::debug::debug_log("SAMURAI", "value=", 0);
    samurai::debug::debug_log("LOAD_BALANCING", "ncells = ", 4);
    samurai::debug::debug_log("MESH", "level=", 2, " cells=", 100);

    int iteration = 5;
    double residual = 1.23e-8;
    std::string status = "converged";

    samurai::debug::debug_log("ITERATION", "step=", iteration, " residual=", residual, " status=", status);

    (void)iteration;
    (void)residual;
    (void)status;

    samurai::debug::debug_log("BOUNDARY_CONDITIONS", "type=periodic", " direction=x");
    samurai::debug::debug_log("TIME_STEPPING", "dt=", 0.001, " time=", 1.5);
    samurai::debug::debug_log("ADAPTATION", "refined_cells=", 15, " coarsened_cells=", 3);



    samurai::finalize();
    return 0;
} 