Long-Run Matrix Runner for Samurai demos

This tool compiles selected Samurai demos and runs a matrix of configurations to stress-test stability across parameters and MPI sizes. It creates per-run directories, captures logs, validates HDF5 outputs for NaNs, and compares MPI runs against a sequential reference using `python/compare.py`.

Quick start
- From repo root: `python3 long_run/run_matrix.py --help`
- Typical: `python3 long_run/run_matrix.py --demos advection_2d,burgers --np 1,2,4 --advection-min 3,4 --advection-max 5,6 --advection-tf 0.1,0.2 --burgers-min 0,1 --burgers-max 2,3 --burgers-tf 0.2`

What it does
- Configures CMake with MPI and demos: `cmake -DWITH_MPI=ON -DBUILD_DEMOS=ON ..`
- Builds targets: `finite-volume-advection-2d`, `finite-volume-burgers`
- Locates built binaries under `build/**/` and runs commands like:
  - `mpirun -np <N> ./finite-volume-advection-2d --min-level <MIN> --max-level <MAX> --Tf <Tf> --nfiles 4 --timers --path <run_dir>`
- For each parameter set:
  - Runs `np=1` first (reference), then each `np>=2`
  - Checks HDF5 files for NaNs
  - Compares MPI vs seq using `python/compare.py`
  - Captures stdout/stderr to `run.log`

Outputs
- Each invocation creates a session directory with timestamp: `long_run/runs/<YYYYMMDD-HHMMSS>/`
- Per-run directories under `long_run/runs/<session>/<demo>/<tag>/np<N>/`
- `run.log` and all generated `.h5` files are kept per run
- Summary table printed at the end and written to `<session>/summary.json`

Notes
- If a run crashes or returns a non-zero exit code, the script records failure and continues.
- Use `--skip-compile` to reuse an existing `build/`.
- Use `--nfiles` to control output files per run (default 4).
- You can override the session timestamp directory with `--session <name>`.
