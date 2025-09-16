=====
Cells
=====

A :cpp:class:`samurai::Cell` represents a single Cartesian element of a mesh. It bundles

- the refinement level (integer, ``level``);
- the integer indices along each axis (``indices``);
- the global storage index used by fields (``index``);
- the geometric data reconstructed from the mesh origin and scaling (``origin_point`` and ``length``).

The class lives in ``samurai/cell.hpp`` and is templated by the mesh dimension and the interval type used internally.

Geometry helpers
----------------

Every cell knows its size and location in physical space. Given the origin ``origin_point`` and the scaling factor of the parent mesh, Samurai computes the edge length as

.. code-block:: c++

   double h = samurai::cell_length(mesh.scaling_factor(), cell.level);

The most common accessors are

``cell.corner()``
    Returns the coordinates of the minimal corner.
``cell.center()``
    Returns the cell centre; a component can be retrieved with ``center(i)``.
``cell.face_center(direction)``
    Returns the centre of the face pointed by a Cartesian unit vector ``direction`` (e.g. ``{1,0}`` in 2D).

Topology and indexing
---------------------

The ``indices`` member stores the integer location in the underlying interval grid. Combined with ``level`` they uniquely identify the cell inside the multiresolution hierarchy. The ``index`` member is the contiguous offset used when fields store values, so iterating over a field and calling ``field[cell]`` is constant-time.

You typically obtain cell objects through mesh iterators:

.. code-block:: c++

   for_each_cell(mesh, [&](const auto& cell)
   {
       auto xc = cell.center();
       auto lv = cell.level;
       // use cell.index to address contiguous storage
   });

   auto one_cell = mesh.get_cell(level, i, j);  // same type

Cells are comparable with ``==``/``!=`` and can be printed via ``operator<<`` for debugging.

Relationship with fields and restart
------------------------------------

Fields use the ``index`` stored in cells to map each cell to an array entry; loading a restart file reconstructs both the mesh (hence all cells) and the field storage so that ``cell.index`` remains valid after ``samurai::load``. Boundary-condition helpers and finite-volume stencils also exchange cells to describe where an operation takes place.

Further reading
---------------

See the :doc:`mesh` section for details on how cells are organised within a mesh, and the :doc:`field` page for manipulating the data attached to each cell.
