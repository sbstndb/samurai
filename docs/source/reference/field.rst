======
Fields
======

A field stores the discrete unknowns attached to each cell of a mesh. Samurai offers two concrete flavours:

- :cpp:class:`samurai::ScalarField` for single-component values;
- :cpp:class:`samurai::VectorField` for multi-component values (``n_comp`` components).

Creation helpers
----------------

Use the factory functions to build the right field type for your mesh:

.. code-block:: c++

   #include <samurai/field.hpp>

   // scalar field of doubles
   auto rho = samurai::make_scalar_field<double>("rho", mesh);

   // vector field with three components (default value type = double)
   auto velocity = samurai::make_vector_field<3>("velocity", mesh);

   // initialise with a constant value
   auto pressure = samurai::make_scalar_field("p", mesh, 1.0);

   // initialise from a lambda evaluated on cell centres
   auto indicator = samurai::make_scalar_field("chi", mesh,
       [](const auto& coords)
       {
           return xt::all(xt::abs(coords - 0.5) < 0.25) ? 1.0 : 0.0;
       });

   // perform Gauss-Legendre projection of a continuous function
   samurai::GaussLegendre<2> quad; // polynomial degree 2
   auto averaged = samurai::make_scalar_field<double>("f", mesh, f, quad);

The templates let you control the storage layout when needed. Passing ``SOA = true`` to :cpp:func:`samurai::make_vector_field` enables structure-of-arrays storage suitable for SIMD-friendly kernels. When omitted, Samurai defaults to an array-of-structures layout which keeps all components of a cell contiguous.

Working with the data
---------------------

A field keeps a reference to the mesh used at construction time. Values can be accessed either via cells or by intervals:

.. code-block:: c++

   for_each_cell(mesh, [&](const auto& cell)
   {
       rho[cell] = initial_density(cell.center());
   });

   // access a block of values on the fly
   auto span = velocity(level, interval, index);
   span += xt::xtensor_fixed<double, xt::xshape<3>>{1.0, 0.0, 0.0};

Use :cpp:func:`samurai::VectorField::fill` or :cpp:func:`samurai::ScalarField::fill` to set every stored value, and :cpp:func:`samurai::VectorField::resize` when the underlying mesh changes size (for example after loading from a restart file). Iterators (:cpp:type:`begin`, :cpp:type:`end`, and their const/reverse variants) are also provided if you prefer STL-style loops.

Boundary conditions
-------------------

Fields hold the boundary conditions attached to them. The helper :cpp:func:`samurai::VectorField::attach_bc` stores the boundary object created by :cpp:func:`samurai::make_bc`, and :cpp:func:`samurai::VectorField::copy_bc_from` lets you duplicate the configuration to another field (same value type and number of components required).

Interoperability
----------------

- Restart files (:doc:`restart`) serialise both the mesh and every field passed to :cpp:func:`samurai::dump`. On reload, :cpp:func:`samurai::load` calls :cpp:func:`resize` for you before repopulating the data.
- Algebra and scheme helpers accept fields transparently. For example, :cpp:func:`samurai::for_each_cell` works with any field built via the factories above, and finite-volume operators expect their arguments to be fields. Tutorials such as :doc:`../tutorial/field` give step-by-step examples.
