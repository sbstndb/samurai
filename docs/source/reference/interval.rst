=========
Intervals
=========

The :cpp:class:`samurai::Interval` type represents the 1D building block used everywhere in |project| to encode sets of consecutive cells. It stores integer coordinates, a stride and an offset into contiguous storage so that manipulating intervals stays cheap even when they shrink or shift.

Structure
---------

.. code-block:: c++

   samurai::Interval<int> i{4, 9};
   // start = 4, end = 9, step = 1, index = 0

   i.contains(6);  // true
   i.size();       // 5 cells
   i.is_valid();   // true (start < end)

Members mean:

``start``
    Left bound (included) in integer coordinates.
``end``
    Right bound (excluded).
``step``
    Increment used when iterating the interval; defaults to 1 but helpers like :cpp:func:`even_elements` switch to 2.
``index``
    Offset in the underlying storage so that the first element resides at ``index + start``. It allows shrinking without relocating the data.

Key operations
--------------

- :cpp:func:`samurai::Interval::contains` tests membership for a given coordinate.
- :cpp:func:`even_elements()` / :cpp:func:`odd_elements()` return views with ``step = 2``.
- Arithmetic operators (`+=`, `-=`, `*=`, `/=`) shift or scale both bounds.
- Bit-shift operators (`>>=`, `<<=`) change the resolution level: ``i >>= 1`` coarsens by a factor 2 while keeping the smallest enclosing interval; ``i <<= 1`` refines.
- Free helpers mirror the in-place variants when you prefer value semantics (``auto j = i >> 1;``).

Intervals appear explicitly when you iterate over meshes or subsets:

.. code-block:: c++

   for_each_interval(mesh[samurai::MRConf::mesh_id_t::cells],
                     [&](std::size_t level, const auto& interval, const auto& index)
   {
       auto span = field(level, interval, index);
       // interval.start/interval.end are integer coordinates at this level
       // span points to the corresponding entries in the field storage
   });

Relation to meshes and subsets
------------------------------

- Mesh queries such as :cpp:func:`samurai::Mesh_base::get_interval` return the canonical interval stored in the :cpp:class:`samurai::CellArray`, ensuring the ``index`` stays consistent with the field layout.
- The algebra of sets (:doc:`subset`) is built on combinations of intervals (intersections, unions, translations), and :cpp:func:`samurai::apply` walks the generated intervals to execute user kernels.
- Restart files serialise intervals as part of the mesh description, so shrinking/expanding a mesh restores identical index assignments after :cpp:func:`samurai::load`.

Tips
----

- Intervals are half-open; remember to stop at ``end`` when writing manual loops.
- :cpp:func:`size()` accounts for ``step`` implicitly when you iterate via :cpp:func:`for_each_interval`; if you manipulate the values yourself adjust the loop increment accordingly.
- When working across levels, rely on the shift operators instead of multiplying/dividing manually—this guarantees you obtain the smallest integer interval enclosing the geometric region.
