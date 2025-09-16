======
Timers
======

The timing utility bundled with |project| helps you instrument critical sections of your code without pulling in an external profiler. It works in both serial and MPI configurations and prints aggregated statistics when requested.

Quick usage
-----------

.. code-block:: c++

   #include <samurai/timers.hpp>

   auto& timers = samurai::times::timers;

   timers.start("total runtime");

   timers.start("assembly");
   assemble_system();
   timers.stop("assembly");

   timers.start("solve");
   solve();
   timers.stop("solve");

   timers.stop("total runtime");
   timers.print();

What the helper does
--------------------

- :cpp:func:`samurai::Timers::start` creates a new timer on first use and stores the current wall clock (``MPI_Wtime`` when MPI is enabled, ``std::chrono`` otherwise).
- :cpp:func:`samurai::Timers::stop` accumulates the elapsed time and increments the call counter. Forgetting to start a timer triggers an assertion.
- :cpp:func:`samurai::Timers::print` formats the collected data. In serial builds the output is sorted by decreasing duration and reports the relative fraction of the total runtime. In MPI builds rank 0 prints min/avg/max values and the ranks that achieved them.
- A timer named ``"total runtime"`` is treated specially: the report shows the untimed portion next to the instrumented sections, which helps spotting blind spots.

Practical advice
----------------

- Wrap coarse phases (mesh adaptation, assembly, solver, I/O) instead of every inner loop; the printout stays readable and overhead remains negligible.
- Pair timers carefully: nested timers are fine, but always stop them in the reverse order you started them.
- When running under MPI, call :cpp:func:`samurai::Timers::print` only where ``world.rank() == 0`` if you already have your own barrier logic; the helper internally guards the extra ranks but still performs global reductions.
