ReFrame Benchmark Campaign
==========================

This tutorial explains how to reproduce the ``finite-volume-advection-2d``
benchmark with `ReFrame <https://github.com/reframe-hpc/reframe>`_ using a
Conda-based Samurai toolchain. The workflow stays fully scripted so that it can
power reproducible benchmarking or regression runs.

Prerequisites
-------------

* A Conda installation with the latest Samurai package available. A minimal
  environment providing the required build dependencies can be created with::

      $ conda create -n samurai-bench -c conda-forge samurai cli11 cmake ninja openmpi
      $ conda activate samurai-bench

  Adjust the package list to match your local preferences (e.g. replace
  ``openmpi`` with another MPI implementation). The important point is that the
  activated environment exposes Samurai headers/libraries as well as the CLI11
  headers requested by the CMake project.

* The Samurai sources (this repository) and the ReFrame project cloned side by
  side. A typical setup is::

      $ git clone https://github.com/reframe-hpc/reframe.git
      $ cd reframe
      $ ./bootstrap.sh

  The ``bootstrap.sh`` step creates ``bin/reframe`` together with the Python
  dependencies that ReFrame needs at runtime.

Repository layout
-----------------

The Samurai tree ships a minimal ReFrame site configuration and a first check
under ``ci/reframe/``:

* ``ci/reframe/config/local.py`` – single-node configuration that looks for
  ReFrame checks inside ``ci/reframe/checks`` and stores stage/output artefacts
  under ``ci/reframe/``.
* ``ci/reframe/checks/finite_volume_advection.py`` – builds the
  ``finite-volume-advection-2d`` demo with CMake and runs it with a short final
  time to keep executions fast while still producing timing information.

Both files are regular Python modules, so they can be adapted easily for other
systems or demos.

Running the finite-volume benchmark
-----------------------------------

1. Activate the Conda environment that carries Samurai and the build
   dependencies::

      $ conda activate samurai-bench

   The check assumes the compilers, CMake config files and CLI11 headers are
   available through the active environment (``$CONDA_PREFIX`` is added to
   ``CMAKE_PREFIX_PATH`` automatically by Conda).

2. From the Samurai checkout, launch ReFrame against the local configuration::

      $ ../reframe/bin/reframe \
            -C ci/reframe/config/local.py \
            -c ci/reframe/checks \
            -n FiniteVolumeAdvection2DTest \
            -r --performance-report

   The command stages the sources under ``ci/reframe/stage``, configures CMake
   with ``-mtune=native -march=native -O3 -g`` and only builds the
   ``finite-volume-advection-2d`` target. During the run the executable is
   launched with ``--Tf 0.01 --timers --nfiles 1`` so the timer table is emitted
   to standard output. ReFrame captures the ``total runtime`` line to populate
   the performance report. The default reference value (1.8 s with a 50%
   tolerance) can be tightened once several runs establish a baseline on your
   machine.

3. Inspect artefacts if needed:

   * ``ci/reframe/stage/...`` keeps the staged build directory (one per run).
   * ``ci/reframe/output/...`` stores ReFrame reports when ``--performance-report``
     is passed.

   These folders are ignored by Git, therefore they can be removed between runs
   if you want a clean workspace.

Extending the campaign
----------------------

* To benchmark another demo, copy
  ``ci/reframe/checks/finite_volume_advection.py`` and change the ``make``
  target plus the executable invocation. Because the check is now environment
  agnostic, additional setup commands can be added via ReFrame hooks if a
  future system needs custom activation steps.
* Conda makes it easy to try different Samurai releases. Simply
  ``conda install samurai=<version>`` in the same environment and rerun ReFrame
  to capture the new timings.
* ReFrame’s ``--run`` flag in the example performs both build and run stages. If
  you only need to compile (for instance to populate caches) use
  ``--build-only``; conversely, ``--skip-build`` will reuse an already compiled
  stagedir.

Troubleshooting
---------------

* If CMake fails to locate CLI11 or Samurai, verify that the packages are
  present in the active Conda environment (``conda list cli11``) and that
  ``CMAKE_PREFIX_PATH`` includes ``$CONDA_PREFIX``.
* ReFrame warns about redefining the ``builtin`` environment because the custom
  configuration overrides the default compiler wrappers. The warning is
  harmless but can be silenced by renaming the environment.
* To start fresh, remove the ``ci/reframe/stage`` directory or run ReFrame with
  ``--clean-stagedir``.
