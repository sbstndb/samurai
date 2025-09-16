=====================
Finite Volume demos
=====================

The ``demos/FiniteVolume`` directory gathers ready-to-run examples that showcase the finite-volume infrastructure on adaptive meshes. Build the demos with `cmake --build ./build --target burgers` (or ``--target all``) after enabling ``BUILD_DEMOS`` during configuration.

Burgers equation (`burgers.cpp`)
--------------------------------

The :path:`demos/FiniteVolume/burgers.cpp` executable solves Burgers equations in 1D or 2D with a third-order Runge–Kutta integrator and WENO5 fluxes. By default it runs the 2D vector case (``dim = 2``, ``n_comp = 2``) but both values can be changed at the bottom of the file (`demos/FiniteVolume/burgers.cpp:248`).

Key ingredients
^^^^^^^^^^^^^^^

- Mesh: multiresolution mesh built from :cpp:class:`samurai::MRMesh` with configurable min/max levels (`demos/FiniteVolume/burgers.cpp:60`).
- Fields: solution vectors ``u``, intermediate RK stages ``u1``, ``u2`` and ``unp1`` (`demos/FiniteVolume/burgers.cpp:78`).
- Spatial operator: WENO5 convection built by :cpp:func:`samurai::make_convection_weno5` (`demos/FiniteVolume/burgers.cpp:130`).
- Time stepping: classical RK3 sequence with mesh adaptation at each stage (`demos/FiniteVolume/burgers.cpp:169`).
- Output: HDF5 snapshots via `save(...)`, plus restart files in sequential runs (`demos/FiniteVolume/burgers.cpp:14`, `demos/FiniteVolume/burgers.cpp:182`).

Command-line options
^^^^^^^^^^^^^^^^^^^^

The demo uses the Samurai CLI helper to expose the main parameters (`demos/FiniteVolume/burgers.cpp:47`). Examples:

.. code-block:: bash

   ./burgers --min-level 1 --max-level 6 --init-sol hat --Tf 0.5 --nfiles 20
   ./burgers --init-sol linear --dim 1 --path results --filename burgers_linear

Available options include the domain bounds (``--left``, ``--right``), the initial condition (``hat``, ``linear``, ``bands``), final time ``--Tf``, CFL/``--dt``, output folder/name and the multiresolution levels. Pass ``--restart-file <path>`` to resume from a ``.h5`` snapshot written earlier.

Initial and boundary conditions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``linear`` (1D): fills the mesh with the analytical solution ``(ax+b)/(at+1)`` via :func:`exact_solution` and keeps Dirichlet BCs matching the true solution at each time step (`demos/FiniteVolume/burgers.cpp:86`, `demos/FiniteVolume/burgers.cpp:204`).
- ``hat``: radial hat profile with compact support (`demos/FiniteVolume/burgers.cpp:101`).
- ``bands`` (vector, dim>1): each component receives a triangular band along its axis (`demos/FiniteVolume/burgers.cpp:117`).

Boundary conditions are imposed with :cpp:func:`samurai::make_bc`. Scalar cases apply homogeneous Dirichlet values, whereas the ``linear`` test injects the analytical solution through a lambda (`demos/FiniteVolume/burgers.cpp:202`). The intermediate RK fields copy the boundary configuration from ``u`` (`demos/FiniteVolume/burgers.cpp:141`).

Mesh adaptation and error monitoring
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Mesh refinement is handled by :cpp:func:`samurai::make_MRAdapt`; the tolerance is taken from ``samurai::mra_config()`` but can be tuned before calling the functor (`demos/FiniteVolume/burgers.cpp:149`). After each time step the code reports L2 errors against the analytical solution when available, and performs reconstruction on the finest level if the mesh is adaptive (`demos/FiniteVolume/burgers.cpp:213`).

Outputs and visualisation
^^^^^^^^^^^^^^^^^^^^^^^^^

Snapshots go to ``<path>/<filename>_ite_<k>.h5`` together with a restart file in sequential runs. Use the helper script ``python python/read_mesh.py`` to inspect the evolution, e.g.

.. code-block:: bash

   python python/read_mesh.py burgers_ite_ --field u level --start 0 --end <nsave>

Other demos
^^^^^^^^^^^

The same directory contains variants covering linear advection, reaction–diffusion, level-set transport, Stokes flows, etc. They all follow the same structure: configure mesh/fields, set boundary conditions, build finite-volume operators from ``samurai/schemes/fv.hpp``, then march in time. Start with ``advection_1d.cpp`` or ``heat.cpp`` for lighter examples before exploring the AMR-focused cases like ``burgers_mra.cpp`` or ``Stokes_2d.cpp``.

Advection in 2D (`advection_2d.cpp`)
------------------------------------

This example transports a compact patch of scalar field with a constant velocity.

Highlights
^^^^^^^^^^

- Mesh: 2D multiresolution mesh with adjustable min/max refinement (`demos/FiniteVolume/advection_2d.cpp:55`).
- Field: scalar field ``u`` representing the transported quantity (`demos/FiniteVolume/advection_2d.cpp:71`).
- Initial condition: a disk of radius 0.2 centred at (0.3, 0.3) (`demos/FiniteVolume/advection_2d.cpp:16`).
- Flux: upwind discretisation built with :cpp:func:`samurai::upwind` (`demos/FiniteVolume/advection_2d.cpp:120`).
- Time stepping: explicit Euler with adaptive mesh update at each iteration (`demos/FiniteVolume/advection_2d.cpp:103`).
- Output: same ``save`` helper as Burgers, writing HDF5 files and restart snapshots (`demos/FiniteVolume/advection_2d.cpp:36`).

CLI options mirror the Burgers demo (`demos/FiniteVolume/advection_2d.cpp:43`): domain corners, velocity vector, CFL/final time, multiresolution levels, output path, number of files and optional ``--restart-file``.
