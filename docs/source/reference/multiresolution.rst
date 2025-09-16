================
Multiresolution
================

Adaptive meshes in Samurai rely on a multiresolution procedure that predicts fine-level values, measures the local detail and tags cells for refinement or coarsening. The core pieces live in ``samurai/mr``—notably :cpp:func:`samurai::make_MRAdapt`, :cpp:class:`samurai::mra_config`, and the prediction operators in ``samurai/mr/operators.hpp``.

Creating an adaptor
-------------------

.. code-block:: c++

   auto u = samurai::make_scalar_field<double>("u", mesh);
   auto MRadapt = samurai::make_MRAdapt(u);

   auto cfg = samurai::mra_config().epsilon(1e-4).regularity(1.0);
   MRadapt(cfg);

``make_MRAdapt`` deduces a prediction operator from the field type (scalar or vector) and returns a functor that you can reuse at each time step (`demos/FiniteVolume/burgers.cpp:146`, `demos/FiniteVolume/advection_2d.cpp:107`). The adaptor keeps temporary fields for details and tags, and runs the Harten refinement loop until no more cells require updates (`include/samurai/mr/adapt.hpp:64`).

Configuring the tolerance
-------------------------

:cpp:class:`samurai::mra_config` stores the thresholds used by the adaptor:

- ``epsilon``: absolute detail threshold; values below it trigger coarsening, larger values enforce refinement (`include/samurai/mr/config.hpp:18`).
- ``regularity``: exponent applied to detail scaling when comparing levels (`include/samurai/mr/config.hpp:26`).
- ``relative_detail``: switch to relative detail (see ``samurai/mr/rel_detail.hpp``) instead of absolute magnitudes.

Before each adaptation pass ``cfg.parse_args()`` merges command-line overrides (``--mr-eps``, ``--mr-reg``, ``--mr-rel-detail``) provided through the CLI integration (`include/samurai/mr/config.hpp:37`).

What happens during adaptation
------------------------------

For each refinement level (from coarse to fine) the adaptor:

1. Computes predicted values on children cells via the selected prediction operator (see ``samurai/mr/operators.hpp``).
2. Stores detail coefficients in a temporary field ``detail`` and compares them to ``epsilon`` (optionally scaled by ``regularity``) in :cpp:func:`samurai::Adapt::harten` (`include/samurai/mr/adapt.hpp:109`).
3. Tags cells for refinement/coarsening in ``tag``.
4. Applies graduation to guarantee the mesh respects the stencil required by operators (`include/samurai/mr/adapt.hpp:28`, ``samurai/algorithm/graduation.hpp``).
5. Updates the mesh, ghost layers and boundary conditions accordingly.

Additional options
------------------

``make_MRAdapt`` accepts extra fields that must follow the same mesh (e.g. multi-component systems) and optional flags to enlarge the adapted region (`include/samurai/mr/adapt.hpp:47`). The prediction operators can also be customised—see ``samurai/mr/operators.hpp`` or build your own by inheriting from :cpp:class:`samurai::field_operator_base`.

Examples
--------

- Burgers demo adapts the mesh at every Runge–Kutta stage and measures reconstruction error on the finest level (`demos/FiniteVolume/burgers.cpp:149`).
- Advection demo sets a tighter tolerance (``epsilon = 2e-4``) via :cpp:func:`samurai::mra_config` to keep the transported disk sharp (`demos/FiniteVolume/advection_2d.cpp:109`).

When using MPI, adaptation automatically redistributes ghost layers and honours the load-balancing hooks described in :doc:`mpi`.
