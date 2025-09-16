=========
MPI support
=========

Samurai uses Boost.MPI to distribute meshes, ghost exchanges and adaptation decisions across ranks. The implementation is fully integrated: once the library is built with ``SAMURAI_WITH_MPI`` the same high-level API works in serial and parallel situations.

Building and initialisation
---------------------------

Enable MPI support at configure time and link against Boost.MPI and an MPI-enabled HDF5 (`CMakeLists.txt:182`, ``conda/mpi-environment.yml``). Typical usage:

.. code-block:: cmake

   set(SAMURAI_WITH_MPI ON)
   find_package(samurai CONFIG REQUIRED)
   target_link_libraries(my_target PRIVATE samurai::samurai)

Executable entry points should start with :cpp:func:`samurai::initialize`, which calls ``MPI_Init``, wires the CLI flag ``--dont-redirect-output`` and redirects non-root stdout to ``/dev/null`` when the flag is not set (`include/samurai/samurai.hpp:63`). ``samurai::finalize`` prints the timers (when ``--timers`` is present) and calls ``MPI_Finalize`` (`include/samurai/samurai.hpp:101`).

Domain partitioning and neighbourhoods
--------------------------------------

When a multiresolution mesh is constructed on several ranks, :cpp:func:`samurai::Mesh_base::partition_mesh` distributes the level-``start_level`` cells. In 1D the global interval is sliced evenly (`include/samurai/mesh.hpp:1090`); in higher dimensions the algorithm walks the list of mesh intervals and assigns contiguous blocks to each rank. The mesh stores the resulting "subdomain" (the local portion of the level-cell list) and a vector of neighbouring ranks (:cpp:func:`samurai::Mesh_base::mpi_neighbourhood`, `include/samurai/mesh.hpp:265`).

Neighbour meshes are synchronised through three helpers:

- :cpp:func:`samurai::Mesh_base::update_mesh_neighbour` sends the full cell array for each mesh id using Boost.MPI packed archives (`include/samurai/mesh.hpp:828`).
- :cpp:func:`samurai::Mesh_base::update_neighbour_subdomain` only exchanges the subdomain part when the partition is unchanged (`include/samurai/mesh.hpp:853`).
- :cpp:func:`samurai::Mesh_base::update_meshid_neighbour` updates a specific mesh id (e.g. ``cells_and_ghosts``) without resending the whole structure (`include/samurai/mesh.hpp:881`).

Ghost layers and communications
-------------------------------

Ghost exchanges are orchestrated by :cpp:func:`samurai::update_ghost_mr` (`include/samurai/algorithm/update.hpp:525`). The sequence for each level is:

1. **Outer ghosts / boundary projection** – :cpp:func:`samurai::update_outer_ghosts` applies user boundary conditions, projects them onto coarser levels and extrapolates outer corners. With MPI enabled, corner projections avoid double ownership by considering the local subdomain (`include/samurai/algorithm/update.hpp:393`).
2. **Periodic copies** – :cpp:func:`samurai::update_ghost_periodic` copies data from the opposite side of the domain. Internally :cpp:func:`samurai::iterate_over_periodic_ghosts` swaps buffers with each neighbouring rank using ``world.isend``/``world.recv`` and reconstructs the ghost values in-place (`include/samurai/algorithm/update.hpp:980`).
3. **Neighbour subdomain exchange** – :cpp:func:`samurai::update_ghost_subdomains` sends slices of the field that overlap another rank’s subdomain and writes back the received values (`include/samurai/algorithm/update.hpp:732`). The helper iterates over ``mesh.mpi_neighbourhood()`` and packs cell data in a contiguous vector for each neighbour.
4. **Projection/prediction** – once outer ghosts are consistent, the algorithm projects coarse values to finer levels and predicts interior ghosts using the configured prediction order (`include/samurai/algorithm/update.hpp:44`).

These steps are applied recursively to all fields passed to ``update_ghost_mr``. The flag ``field.ghosts_updated()`` prevents redundant exchanges; call :cpp:func:`samurai::update_ghost_mr_if_needed` when you are unsure a field is current (`include/samurai/algorithm/update.hpp:509`).

Adaptation and tag synchronisation
----------------------------------

Mesh adaptation relies on Harten’s criterion implemented in :cpp:class:`samurai::Adapt` (`include/samurai/mr/adapt.hpp:47`). During each iteration the adaptor:

- Resets the tag field and calls :cpp:func:`samurai::update_tag_subdomains` to copy tag values on interfaces shared with neighbours (`include/samurai/mr/adapt.hpp:261`).
- Updates ghosts (see above) so detail coefficients are computed with up-to-date halo values (`include/samurai/mr/adapt.hpp:272`).
- Exchanges detail fields via :cpp:func:`samurai::update_ghost_subdomains` before applying refinement/coarsening tests (`include/samurai/mr/adapt.hpp:328`).
- Propagates keep/coarsen decisions with :cpp:func:`samurai::keep_only_one_coarse_tag` and additional ``update_tag_subdomains`` calls to guarantee consistency across ranks (`include/samurai/algorithm/update.hpp:853`).
- Builds the new :cpp:class:`samurai::CellArray`, performs graduation, and swaps the mesh. The equality check uses a global ``all_reduce`` so adaptation stops only when every rank reports no change (`include/samurai/mr/adapt.hpp:382`).

Load balancing hooks (:cpp:func:`samurai::Mesh_base::load_balancing`) estimate workloads using ``mpi::all_gather`` and compute fluxes to neighbours (`include/samurai/mesh.hpp:1136`). The current implementation prints the targeted transfers; custom strategies can use the ``load_fluxes`` vector to migrate cells between ranks.

Periodicity and subdomain masks
-------------------------------

Ghost updates honour periodic flags per direction (`include/samurai/mesh.hpp:68`). Periodic transfers happen before MPI exchanges so that a subdomain bordering a periodic face still receives data from the opposite side. When boundaries are non-periodic, helpers such as :cpp:func:`samurai::keep_boundary_refined` ensure all ranks refine their boundary layer consistently (`include/samurai/mr/adapt.hpp:230`).

Restart and collective I/O
--------------------------

:cpp:func:`samurai::dump` and :cpp:func:`samurai::load` switch to collective HDF5 accessors when MPI is active. Each rank dumps its portion of the ``CellArray`` and the field data into a shared file; the metadata store the communicator size and fail fast on mismatches (`include/samurai/io/restart.hpp:92`, `include/samurai/io/restart.hpp:267`).

Diagnostics and CLI helpers
---------------------------

The CLI flag ``--timers`` aggregates timers across ranks and prints min/avg/max statistics on rank 0 (`include/samurai/timers.hpp:95`). ``--dont-redirect-output`` keeps stdout from all ranks visible, which is useful when debugging MPI runs (`include/samurai/arguments.hpp:27`). The other CLI knobs (``--mr-eps``, ``--mr-reg``) naturally affect the global adaptation thanks to the synchronisation steps described above.

Best practices
--------------

- Construct meshes and fields *after* calling :cpp:func:`samurai::initialize` so that the communicator is ready.
- Always call :cpp:func:`samurai::update_ghost_mr` (or ``_if_needed``) after mesh adaptation and before applying stencil operators.
- Use the provided helpers for tag and ghost exchanges instead of inserting custom MPI messages; the infrastructure already handles projection/prediction and boundary interactions.
- When adding new MPI-aware routines, follow the existing pattern: pack interval data in contiguous buffers, send with ``world.isend`` and receive with ``world.recv`` while keeping the order consistent with the mesh topology.
