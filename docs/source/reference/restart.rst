=================
Restart and I/O
=================

The restart helpers let you dump a mesh together with one or several fields to an HDF5 file and later rebuild the exact same state. They work with sequential and MPI builds and reuse the HighFive backend that already ships with |project|.

Getting started
---------------

.. code-block:: c++

   #include <samurai/io/restart.hpp>

   samurai::MRMesh<Config> mesh{box, min_level, max_level};
   auto u = samurai::make_field<double, 1>("u", mesh);

   // write <solution.h5>
   samurai::dump("solution", mesh, u);

   // reload the file (mesh and fields are overwritten in-place)
   samurai::load("solution", mesh, u);

Key points
----------

- The functions :cpp:func:`samurai::dump` and :cpp:func:`samurai::load` work on any mesh deriving from :cpp:class:`samurai::Mesh_base`, including uniform meshes.
- The first argument can be either a file stem (``"solution"``) or an explicit path (``samurai::dump(path, filename, ...)``). The ``.h5`` suffix is appended automatically.
- Every field to be dumped must have a non-empty name (``make_field`` sets it). When loading, a matching field name and number of components are required.
- In MPI builds the I/O uses collective HighFive accessors. The restart file stores the number of ranks and will refuse to load if it does not match the current communicator size.
- The mesh geometry (origin, scaling, min/max level per rank) is stored alongside the cell intervals, so the reconstructed mesh is identical to the saved one.
- :cpp:func:`samurai::dump_fields` and :cpp:func:`samurai::load_fields` expose lower level control when you already manage the :cpp:class:`samurai::CellArray` yourself.

Tips
----

- When restarting long runs, keep a separate timer (see :doc:`timers`) around the I/O calls to understand their weight.
- If you only need the raw arrays, :cpp:func:`samurai::extract_data_as_vector` in ``samurai/io/util.hpp`` gives direct access to the flattened values without writing a file.
- The restart helpers do not create parent directories. Call ``std::filesystem::create_directories`` yourself before dumping to nested paths.
