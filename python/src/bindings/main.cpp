// Samurai Python Bindings - Main Module
//
// This file serves as the entry point for the Python bindings.
// Bindings will be added progressively following the phased approach:
// - Phase 1: Core types (Box, Mesh, Field, Cell)
// - Phase 2: Algorithms (for_each_cell, adapt, BC)
// - Phase 3: Operators (diffusion, upwind, etc.)
// - Phase 4: I/O (HDF5 save/load)

#include <pybind11/pybind11.h>

// Binding initialization headers
#include "adapt_bindings.hpp"
#include "algorithm_bindings.hpp"
#include "bc_bindings.hpp"
#include "box_bindings.hpp"
#include "domain_builder_bindings.hpp"
#include "field_bindings.hpp"
#include "interval_bindings.hpp"
#include "io_bindings.hpp"
#include "mesh_bindings.hpp"
#include "mesh_config_bindings.hpp"
#include "mra_config_bindings.hpp"
#include "operator_bindings.hpp"

namespace py = pybind11;

// Version information (will be read from version.txt in production)
#define SAMURAI_PYTHON_VERSION "0.30.0"

PYBIND11_MODULE(samurai_python, m)
{
    // Module documentation
    m.doc() = R"pbdoc(
        Samurai Python Bindings
        -----------------------

        Adaptive Mesh Refinement (AMR) and Multiresolution Analysis library for Python.

        **IMPORTANT API CHANGE (v0.30.0+):** All types are now organized in submodules.

        Quick Start
        -----------
        >>> import samurai_python as sam
        >>>
        >>> # Create mesh using factory (recommended)
        >>> box = sam.geometry.box([0., 0.], [1., 1.])
        >>> mesh = sam.mesh.make(box, min_level=4, max_level=8)
        >>>
        >>> # Create field using factory
        >>> u = sam.field.scalar(mesh, "u")
        >>> vel = sam.field.vector(mesh, "vel", n_components=2)
        >>>
        >>> # Mesh adaptation
        >>> MRadapt = sam.adaptation.make_MRAdapt(u)
        >>> MRadapt(sam.config.MRAConfig(epsilon=1e-4))

        Submodules
        ----------
        .. autosummary::
           :toctree: _generate

           samurai_python.geometry
           samurai_python.config
           samurai_python.mesh
           samurai_python.field
           samurai_python.interval
           samurai_python.algorithms
           samurai_python.adaptation
           samurai_python.io

        Main Module Functions
        ---------------------
        The following functions remain available at module level for convenience:

        .. autosummary::
           :toctree: _generate

           upwind
           make_dirichlet_bc
           make_neumann_bc

        Examples
        --------
        See the submodule documentation for detailed examples:

        - `sam.geometry` - Geometric primitives (Box, DomainBuilder)
        - `sam.config` - Mesh configuration (MeshConfig, MRAConfig)
        - `sam.mesh` - Mesh types and creation
        - `sam.field` - Field types and creation
        - `sam.adaptation` - Mesh adaptation (MRAdapt, update_ghost_mr)
        - `sam.algorithms` - Iteration algorithms (for_each_cell, for_each_interval)
    )pbdoc";

    // Version attribute
    m.attr("__version__") = SAMURAI_PYTHON_VERSION;

    // Initialize bindings
    init_box_bindings(m);
    init_domain_builder_bindings(m);
    init_mesh_config_bindings(m);
    init_mesh_bindings(m);
    init_field_bindings(m);
    init_interval_bindings(m);
    init_algorithm_bindings(m);
    init_operator_bindings(m);
    init_bc_bindings(m);
    init_mra_config_bindings(m);
    init_adapt_bindings(m);
    init_io_bindings(m);

    // TODO: Add more submodule initializers as they are implemented
    // init_fv_bindings(m);  // Finite volume schemes
    // init_lbm_bindings(m); // Lattice Boltzmann methods

    // Placeholder: Basic test function
    m.def(
        "test_function",
        []()
        {
            return "Samurai Python bindings are working!";
        },
        R"pbdoc(
        Test function to verify bindings are loaded correctly.

        Returns:
            str: Success message
    )pbdoc");

// Python module metadata
#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = SAMURAI_PYTHON_VERSION;
#endif
}
