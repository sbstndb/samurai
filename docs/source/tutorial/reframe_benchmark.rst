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
  headers requested by the CMake project. The ReFrame configuration uses the
  ``mpirun`` launcher, so the chosen MPI distribution must expose the command in
  the environment.

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
* ``ci/reframe/checks/finite_volume_advection.py`` – parameterised check that
  builds the finite-volume demos and runs them with short final times to keep
  executions fast while still producing timing information.

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
            -n FiniteVolumeDemoTest \
            -r --performance-report

   The command stages the sources under ``ci/reframe/stage``, configures CMake
   with ``-mtune=native -march=native -O3 -g`` and builds the demo required by
   each test instance. Four runs are executed automatically: advection and
   Burgers, both with 1 and 2 MPI ranks. During the advection runs the
   executable is launched with ``--Tf 0.01 --timers --nfiles 1`` so the timer
   table is emitted to standard output. ReFrame captures the ``total runtime``
   line to populate the performance report. Burgers uses ``--Tf 0.05`` and a
   single output file, matching the light-weight profiling use case. The
   default references are currently set to ~3.0 s (1 rank) and ~3.3 s (2 ranks)
   with generous 80% tolerances; refine them once several runs establish a
   baseline on your machine. The test exposes the variables
   ``FiniteVolumeDemoTest.final_time`` and ``FiniteVolumeDemoTest.nfiles`` so
   you can override the command-line arguments without editing the test::

      $ ../reframe/bin/reframe ... \
            -S FiniteVolumeDemoTest.final_time=0.5 \
            -S FiniteVolumeDemoTest.nfiles=1

   The overrides apply to every demo instance of the run.
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

* To benchmark another demo, copy the structure in
  ``ci/reframe/checks/finite_volume_advection.py`` and add an entry in the
  ``DEMO_CONFIGS`` dictionary with the CMake target, runtime options and sanity
  pattern. Because the check is environment agnostic, additional setup commands
  can be added via ReFrame hooks if a future system needs custom activation
  steps.
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
