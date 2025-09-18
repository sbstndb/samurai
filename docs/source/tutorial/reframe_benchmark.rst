ReFrame Benchmark Campaign
==========================

This tutorial explains how to reproduce the ``finite-volume-advection-2d``
benchmark with `ReFrame <https://github.com/reframe-hpc/reframe>`_ on top of a
Spack powered Samurai installation.  The workflow is entirely scripted so that
it can be reused for performance tracking or regression testing.

Prerequisites
-------------

* A working Spack installation that provides ``samurai@0.26.1`` and a compatible
  ``cli11`` release (``cli11@2.3.2`` is known to work with this demo).
* The Samurai sources (this repository) and the ReFrame project cloned side by
  side.  A typical setup is::

      $ git clone https://github.com/reframe-hpc/reframe.git
      $ cd reframe
      $ ./bootstrap.sh

  The ``bootstrap.sh`` step creates ``bin/reframe`` along with the Python
  dependencies that ReFrame requires at runtime.

Repository layout
-----------------

The Samurai tree now ships a minimal ReFrame site configuration and a first
check under ``ci/reframe/``:

* ``ci/reframe/config/local.py`` – single-node configuration that looks for
  ReFrame checks inside ``ci/reframe/checks`` and stores stage/output artefacts
  under ``ci/reframe/``.
* ``ci/reframe/checks/finite_volume_advection.py`` – builds the
  ``finite-volume-advection-2d`` demo with CMake and runs it with a short final
  time to keep executions fast while still producing timing information.

Both files are regular Python modules, so they can be adapted to fit another
system or benchmark.

Running the finite-volume benchmark
-----------------------------------

1. Prepare the Spack environment that the check expects::

      $ . ~/spack/share/spack/setup-env.sh
      $ export SAMURAI_CLI11_SPEC=cli11@2.3.2
      $ export SAMURAI_SPACK_SPEC=samurai@0.26.1

   The ReFrame check loads these specs explicitly and also queries
   ``spack location`` to set ``CLI11_ROOT`` during the CMake configure step.
   Adjust the environment variables if you prefer different build variants.

2. From the Samurai checkout, launch ReFrame against the local configuration::

      $ ../reframe/bin/reframe \
            -C ci/reframe/config/local.py \
            -c ci/reframe/checks \
            -n FiniteVolumeAdvection2DTest \
            -r --performance-report

   The command stages the sources under ``ci/reframe/stage``, configures CMake
   with ``-mtune=native -march=native -O3 -g`` and builds only the
   ``finite-volume-advection-2d`` target.  During the run the executable is
   launched with ``--Tf 0.01 --timers --nfiles 1`` so the timer table is emitted
   to standard output.  ReFrame captures the ``total runtime`` line to populate
   the performance report.  The default reference value (1.8 s with a 50%
   tolerance) can be tightened once several runs establish a baseline on your
   machine.

3. Inspect artefacts if needed:

   * ``ci/reframe/stage/...`` keeps the staged build directory (one per run).
   * ``ci/reframe/output/...`` stores ReFrame reports when ``--performance-report``
     is passed.

   These folders are ignored by Git so they can be safely removed between runs
   if you want a clean workspace.

Extending the campaign
----------------------

* To benchmark another demo, copy
  ``ci/reframe/checks/finite_volume_advection.py`` and change the ``make``
  target plus the executable invocation.  All common helper code lives in the
  same file (e.g. the Spack bootstrap commands) so sharing a base class is
  straightforward if the pipeline grows.
* The check honours the environment variables ``SAMURAI_CLI11_SPEC`` and
  ``SAMURAI_SPACK_SPEC``.  Define them once in a ReFrame mode or in your shell if
  the canonical Spack specs evolve.
* ReFrame’s ``--run`` flag in the example performs both build and run stages.  If
  you only need to compile (for instance to populate caches) use
  ``--build-only``; conversely, ``--skip-build`` will reuse an already compiled
  stagedir.

Troubleshooting
---------------

* If CMake fails with ``cli11`` conversion errors, double check that the Spack
  module for ``cli11`` is at version 2.3.x.  Newer releases change the default
  array serialisation and trigger compilation errors for this demo.
* ReFrame warns about redefining the ``builtin`` environment because the custom
  configuration overrides the default compiler wrappers.  The warning is
  harmless but can be silenced by renaming the environment.
* To start fresh, remove the ``ci/reframe/stage`` directory or run ReFrame with
  ``--clean-stagedir``.
