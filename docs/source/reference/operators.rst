=========
Operators
=========

The operator framework gives you reusable bricks to implement numerical stencils once and apply them transparently to fields living on Samurai meshes. It revolves around two components located in ``samurai/operators_base.hpp``:

- :cpp:class:`samurai::field_operator_base` stores the iteration context (level, interval and lateral indices);
- :cpp:func:`samurai::make_field_operator_function` lifts your operator to a ``field_expression`` so you can combine it with other expressions and assign it to fields.

Implementing an operator
------------------------

Create a struct inheriting from :cpp:class:`samurai::field_operator_base` and define an ``operator()`` that takes the dimension as a ``samurai::Dim<N>`` tag followed by the fields you want to read or write. The helper macro ``INIT_OPERATOR`` fills in the common aliases and constructors:

.. code-block:: c++

   template <std::size_t dim, class interval_t>
   struct projection_op : samurai::field_operator_base<dim, interval_t>
   {
       INIT_OPERATOR(projection_op);

       template <class Dest, class Src>
       inline void operator()(samurai::Dim<1>, Dest& dest, const Src& src) const
       {
           dest(this->level, this->i) = 0.5 * (src(this->level + 1, 2 * this->i)
                                             + src(this->level + 1, 2 * this->i + 1));
       }

       template <class Dest, class Src>
       inline void operator()(samurai::Dim<2>, Dest& dest, const Src& src) const
       {
           dest(this->level, this->i, this->j) = 0.25
             * (src(this->level + 1, 2 * this->i,     2 * this->j)
              + src(this->level + 1, 2 * this->i + 1, 2 * this->j)
              + src(this->level + 1, 2 * this->i,     2 * this->j + 1)
              + src(this->level + 1, 2 * this->i + 1, 2 * this->j + 1));
       }
   };

   auto projector = samurai::make_field_operator_function<projection_op>(coarse, fine);
   coarse = projector;   // perform level-to-level projection

Inside ``operator()`` you have access to:

- ``this->level``: current refinement level;
- ``this->i``: the interval along the main axis, with integer bounds and storage index;
- ``this->index`` (and ``this->j``, ``this->k`` in higher dimensions): integer indices in the remaining directions.

 You can return values, assign directly to output fields, or build more elaborate expressions by composing the result with other field expressions.

Combining operators and subsets
-------------------------------

``make_field_operator_function`` works equally well on subsets built with :cpp:func:`samurai::intersection` or :cpp:func:`samurai::union_`. Restrict the action of an operator to a given region via :cpp:func:`samurai::Subset::apply_op` or :cpp:func:`samurai::apply`:

.. code-block:: c++

   auto subset = samurai::intersection(mesh[mesh_id_t::cells], ghosts).on(level);
   subset.apply_op(
       [&](auto&& field)
       {
           return samurai::make_field_operator_function<projection_op>(std::forward<decltype(field)>(field));
       });

Ready-made building blocks
--------------------------

The finite-volume module reuses the same foundations:

- Flux-based schemes (:doc:`finite_volume_schemes`) configure a set of directional stencils.
- Local schemes (:doc:`local_schemes`) operate cell per cell but rely on the same index and interval helpers provided by :cpp:class:`samurai::field_operator_base`.
- Explicit operator sums (``samurai/schemes/fv/explicit_operator_sum.hpp``) let you accumulate multiple operators with automatic time stepping.

When writing your own solvers, keep operators modular: expose the core stencil through a small object inheriting from :cpp:class:`samurai::field_operator_base`, then plug it into the rest of the framework (field expressions, subsets, time integrators) as needed.
