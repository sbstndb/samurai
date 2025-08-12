### Program structure and CLI

Initialize and finalize Samurai:

```cpp
int main(int argc, char* argv[])
{
  auto& app = samurai::initialize("Description", argc, argv);
  // add CLI options on app (CLI11)
  SAMURAI_PARSE(argc, argv);
  // ... work ...
  samurai::finalize();
}
```

- `samurai::initialize` sets description, reads Samurai arguments, starts timers, and (if enabled) initializes MPI and PETSc.
- `SAMURAI_PARSE(argc, argv)` parses CLI11 options and integrates with PETSc options if present.

Timers and MPI/PETSc
- When `samurai::finalize()` runs and `--timers` is set (default in `args::timers`), it prints accumulated timers.
- With MPI enabled, `initialize` performs `MPI_Init` and redirects stdout to rank 0 unless `--dont-redirect-output` is set.
- With PETSc enabled, `initialize` calls `PetscInitialize` and disables `-options_left` warnings.

Adding options follows the demos (e.g. `advection_1d.cpp`, `heat.cpp`):

```cpp
double left=-1, right=1; std::size_t min_level=3, max_level=6;
app.add_option("--left", left)->capture_default_str()->group("Simulation parameters");
app.add_option("--right", right)->capture_default_str()->group("Simulation parameters");
app.add_option("--min-level", min_level)->capture_default_str()->group("Multiresolution");
app.add_option("--max-level", max_level)->capture_default_str()->group("Multiresolution");
```

Common simulation options used in demos
- Simulation parameters: domain bounds (`--left/--right`), CFL (`--cfl`), time window (`--Ti/--Tf`), restart file
- Multiresolution: `--min-level`, `--max-level`, and MRA config via `--epsilon`, `--regularity`, `--rel-detail`
- Output: `--path`, `--filename`, `--nfiles`, `--save-final-state-only`

