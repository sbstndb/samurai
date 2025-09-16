Boundary condition
==================

Samurai exposes a small hierarchy of types to implement boundary conditions. The
classes below are the entry points used by the factories such as
:cpp:func:`samurai::make_bc`.

.. doxygenstruct:: samurai::BcValue
   :project: samurai
   :members:

.. doxygenclass:: samurai::Bc
   :project: samurai
   :members:

.. doxygenclass:: samurai::ConstantBc
   :project: samurai
   :members:

.. doxygenclass:: samurai::FunctionBc
   :project: samurai
   :members:

.. doxygenstruct:: samurai::BcRegion
   :project: samurai
   :members:

.. doxygenclass:: samurai::SetRegion
   :project: samurai
   :members:

.. doxygenclass:: samurai::OnDirection
   :project: samurai
   :members:
