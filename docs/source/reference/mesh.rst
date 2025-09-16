====
Mesh
====

Samurai exposes two families of meshes built on the same interval-based storage:

- :cpp:class:`samurai::UniformMesh` for single-level Cartesian grids;
- :cpp:class:`samurai::MRMesh` for multiresolution/adaptive meshes with several refinement levels.

Both derive from the utilities implemented in ``samurai/mesh.hpp`` and share the same query helpers (:cpp:func:`nb_cells`, :cpp:func:`get_interval`, :cpp:func:`get_cell`, etc.).

Choosing a configuration
------------------------

A mesh type is defined by a ``Config`` structure that fixes the dimension, optional ghost width and the interval representation:

.. code-block:: c++

   using UConfig = samurai::UniformConfig<2>;        // 2D uniform mesh
   using MRConf  = samurai::MRConfig<2>;             // default multiresolution parameters

   samurai::UniformMesh<UConfig> uniform;
   samurai::MRMesh<MRConf>       adaptive;

You can override the defaults when you need larger stencils or a different integer interval type. For example, ``samurai::MRConfig<3, 4>`` sets the dimension to 3 and reserves a ghost width of 4 cells around each refined patch.

Constructing a mesh
-------------------

Uniform meshes live on a single refinement level that you pass at construction time. Multiresolution meshes receive both the minimum and maximum levels:

.. code-block:: c++

   constexpr std::size_t dim = 2;
   samurai::Box<double, dim> box({0., 0.}, {1., 1.});

   // one-level grid at level = 5
   samurai::UniformMesh<UConfig> uniform_mesh(box, /*level=*/5);

   // adaptive mesh between levels 2 and 8
   samurai::MRMesh<MRConf> mr_mesh(box, /*min_level=*/2, /*max_level=*/8);

Constructors are also available for ``DomainBuilder`` objects and for pre-existing cell lists/arrays, which is how loaders and mesh adaptation routines rebuild a mesh.

Mesh identifiers and ghost layers
---------------------------------

Meshes expose several views through an enum ``mesh_id_t``. For multiresolution grids the ids are:

``cells``
    Real cells of the current mesh.
``cells_and_ghosts``
    The real cells plus the ghost layer sized according to ``Config::ghost_width``.
``proj_cells``
    Cells participating in projection during multiresolution operations.
``union_cells``
    Union of all refinement levels, useful when applying level-wise operators.
``all_cells``
    A convenience alias equal to ``reference``.

Uniform meshes only define ``cells`` and ``cells_and_ghosts``. Retrieve the desired view with ``mesh[mesh_id_t::cells]``; the result is a :cpp:class:`samurai::CellArray`, ready to feed into iterators such as :cpp:func:`samurai::for_each_interval` or :cpp:func:`samurai::for_each_cell`.

Query helpers
-------------

Typical information is available directly on the mesh:

.. code-block:: c++

   auto cells = mr_mesh[samurai::MRConf::mesh_id_t::cells];
   std::size_t total = mr_mesh.nb_cells();
   std::size_t finest = mr_mesh.max_level();
   double dx = mr_mesh.cell_length(finest);  // cell size at the finest level

   using interval_t = typename decltype(mr_mesh)::interval_t;
   interval_t guess{0, 4};
   auto interval = mr_mesh.get_interval(finest, guess, 0);
   auto cell = mr_mesh.get_cell(finest, 3, 0);

``get_interval`` returns the interval actually stored by the underlying :cpp:class:`samurai::CellArray` (useful when you need the canonical index), while ``get_cell`` exposes the full :cpp:class:`samurai::Cell` with its coordinates, index and level.

Periodicity and topology
------------------------

MR meshes provide constructors that accept an array of booleans to mark periodic directions. The information is retained by :cpp:func:`samurai::Mesh_base::is_periodic` and propagated to the subset and ghost-building routines. Scaling utilities (:cpp:func:`set_origin_point`, :cpp:func:`set_scaling_factor`, :cpp:func:`scale_domain`) let you keep the integer interval representation while working in any physical coordinate system.

MPI-aware features
------------------

When Samurai is compiled with ``SAMURAI_WITH_MPI``, meshes keep track of neighbouring subdomains via :cpp:func:`samurai::Mesh_base::mpi_neighbourhood`. Ghost construction and restart I/O automatically rely on collective communications, so the same application code works in both serial and distributed runs.

Further reading
---------------

- The :doc:`../tutorial/interval` and :doc:`../tutorial/field` tutorials introduce the interval storage and how to attach fields to a mesh.
- :doc:`../tutorial/1d_burgers_amr` shows how :cpp:class:`samurai::MRMesh` integrates with the AMR adaptation loop.
