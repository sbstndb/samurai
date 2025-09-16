============
CLI support
============

Samurai embeds a thin wrapper around `CLI11 <https://cliutils.github.io/CLI11/>`_ so that every executable can expose consistent command-line options. Include ``samurai/samurai.hpp`` and call :cpp:func:`samurai::initialize` at the beginning of ``main`` (`demos/FiniteVolume/burgers.cpp:32`, `demos/FiniteVolume/advection_2d.cpp:38`).

Getting started
---------------

.. code-block:: c++

   int main(int argc, char** argv)
   {
       auto& app = samurai::initialize("My demo", argc, argv);

       double Tf = 1.0;
       app.add_option("--Tf", Tf, "Final time")->capture_default_str();

       SAMURAI_PARSE(argc, argv);
       // ... run the code ...
       samurai::finalize();
   }

Key points
----------

- ``samurai::initialize`` sets up CLI11, registers Samurai-specific options (see below) and, when compiled with MPI, initialises the communicator while optionally redirecting stdout to rank 0 (`include/samurai/samurai.hpp:63`).
- ``SAMURAI_PARSE(argc, argv)`` wraps ``app.parse`` and, if PETSc is enabled, allows extra PETSc options before handing control back to the caller (`include/samurai/samurai.hpp:21`).
- ``samurai::finalize`` handles PETSc/MPI teardown and prints the collected timers when ``--timers`` is passed (`include/samurai/samurai.hpp:99`).

Built-in flags
--------------

``--timers``
    Print the timing summary handled by :cpp:class:`samurai::Timers` at exit (`include/samurai/arguments.hpp:32`).
``--finer-level-flux``
    Control flux reconstruction on finer levels (0 disables it; -1 means "use max level"; positive values add levels) (`include/samurai/arguments.hpp:36`).
``--refine-boundary``
    Keep boundary cells refined at ``max_level`` (`include/samurai/arguments.hpp:38`).
``--save-debug-fields``
    Inject additional diagnostic fields when saving HDF5 files (`include/samurai/arguments.hpp:39`).
``--mr-eps`` / ``--mr-reg`` / ``--mr-rel-detail``
    Tune the multiresolution adaptation tolerance, regularity and relative/absolute detail configuration (`include/samurai/arguments.hpp:41`).
``--dont-redirect-output``
    (MPI builds) Disable the default redirection of non-root outputs to ``/dev/null`` (`include/samurai/arguments.hpp:27`).

These defaults are available automatically—just add your own ``app.add_option`` / ``add_flag`` entries as shown in the demos.

Examples in the repo
--------------------

- Burgers demo (`demos/FiniteVolume/burgers.cpp:47`) groups options into "Simulation parameters", "Multiresolution" and "Output" with ``->group(...)`` and uses ``capture_default_str`` so ``--help`` prints the current defaults.
- Advection demo (`demos/FiniteVolume/advection_2d.cpp:43`) exposes vector options (``--velocity``) and relies on restart support by reusing ``samurai::load`` when ``--restart-file`` is set.

When running with PETSc or MPI, ensure you call ``samurai::initialize``/``finalize`` to propagate CLI settings to the underlying libraries.``
