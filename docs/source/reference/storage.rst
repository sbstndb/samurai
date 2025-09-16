================
Data structures
================

Samurai organises cells and fields around a few key containers. Understanding how they store data makes it easier to write efficient operators and to reason about memory usage.

Cell arrays
-----------

``samurai/level_cell_array.hpp`` implements :cpp:class:`samurai::LevelCellArray`, a per-level container of intervals. At each refinement level it stores an array of one-dimensional :cpp:class:`samurai::Interval` objects plus offset arrays for the transverse directions (`include/samurai/level_cell_array.hpp:67`). Constructors accept
a pre-built :cpp:class:`samurai::LevelCellList`, a :doc:`subset` description or a geometric box, and expose random-access iterators for traversal (`include/samurai/level_cell_array.hpp:314`).

A multilevel :cpp:class:`samurai::CellArray` (`include/samurai/cell_array.hpp:1`) simply stacks `LevelCellArray` instances for every level between ``min_level`` and ``max_level``. Meshes hold one `CellArray` per mesh id (`mesh_id_t::cells`, ``cells_and_ghosts``, …) and the mesh iterators in :cpp:func:`samurai::for_each_cell` or :cpp:func:`samurai::for_each_interval` reuse these containers (`include/samurai/mesh.hpp:56`).

Fields and storage layout
-------------------------

Fields rely on ``samurai/storage/containers.hpp`` to store values. Scalar fields use a flat array of size ``nb_cells`` whereas vector fields can optionally switch to a structure-of-arrays layout by passing ``SOA = true`` to :cpp:func:`samurai::make_vector_field` (`include/samurai/field.hpp:813`). Both layouts track a `static_layout` flag and resize automatically when the mesh changes (`include/samurai/field.hpp:191`).

Each cell carries a contiguous index (``cell.index``) that maps directly to the underlying storage (`include/samurai/cell.hpp:32`). When a `CellArray` changes (due to adaptation), fields call ``resize`` and the mesh recomputes the indices so they remain consistent (`include/samurai/mesh.hpp:216` and `include/samurai/field.hpp:191`).

Derived storage traits (``samurai/storage/containers.hpp``) also provide access to xtensor views for interval-based updates. The call ``field(level, interval, index)`` returns a view spanning the interval in the primary direction and the contiguous components in the transverse directions, honouring the `step` attribute when needed (`include/samurai/field.hpp:213`).

Relations with other modules
----------------------------

- Boundary conditions clone the underlying field storage and work entirely through `cell.index`/interval views (`include/samurai/bc/bc.hpp:232`).
- Restart I/O serialises ``LevelCellArray`` and ``CellArray`` before dumping the field data, so that both mesh geometry and storage layout are restored exactly (`include/samurai/io/restart.hpp:92`, `include/samurai/io/restart.hpp:333`).
- The ``--finer-level-flux`` and ``--save-debug-fields`` CLI flags tap into these structures to compute fluxes on finer meshes and to dump additional coordinate/level arrays (`include/samurai/arguments.hpp:36`).

For memory profiling, :cpp:func:`samurai::memory_usage` sums the contributions of `LevelCellArray` and `CellArray` containers (`include/samurai/memory.hpp:16`). Use it to monitor the effect of tolerance changes or adaptation strategies.
