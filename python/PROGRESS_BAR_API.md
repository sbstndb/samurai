# Progress Bar API for Samurai Python

## Overview

The progress bar API provides a simple, intuitive way to track progress in time-dependent Samurai simulations with adaptive mesh refinement. It uses `tqdm` for beautiful progress bars with ETA estimation and supports mesh statistics tracking.

## Installation

The progress bar utilities are included with the Samurai Python package. The only dependency is `tqdm`, which should already be installed in your environment.

```bash
# If tqdm is not installed
pip install tqdm
```

## Quick Start

### Basic Time Stepping

```python
import samurai_python as sam
from samurai_python.utils import progress

# Simple time loop
with progress.time_loop(Tf=1.0, dt=0.01) as pbar:
    while pbar.continue_loop():
        # Simulation step
        MRadaptation(config)
        update_ghost_mr(u)
        pbar.advance_time(dt)
```

### With Mesh Statistics

```python
# Track mesh adaptation in progress bar
with progress.time_loop(Tf=1.0, dt=0.01) as pbar:
    while pbar.continue_loop():
        # Simulation step
        MRadaptation(config)
        update_ghost_mr(u)

        # Update progress bar with mesh statistics
        pbar.advance_time(dt)
        pbar.update_stats(mesh=u.mesh)
```

### Custom Statistics

```python
# Add custom statistics to progress bar
with progress.time_loop(Tf=1.0, dt=0.01) as pbar:
    while pbar.continue_loop():
        # Simulation step
        pbar.advance_time(dt)
        pbar.update_stats(
            mesh=u.mesh,
            residual=compute_residual(),
            max_val=float(max(u))
        )
```

## API Reference

### `time_loop(Tf, dt, desc="Time loop", disable=False, **kwargs)`

Create a time-stepping progress bar context manager.

**Parameters:**
- `Tf` (float): Final simulation time
- `dt` (float): Initial time step (for ETA estimation)
- `desc` (str): Description for progress bar (default: "Time loop")
- `disable` (bool): If True, disable progress bar display
- `**kwargs`: Additional arguments passed to tqdm

**Returns:** `TimeLoop` context manager

**Methods:**
- `continue_loop() -> bool`: Check if simulation should continue
- `advance_time(dt=None) -> None`: Advance simulation time
- `update_stats(mesh=None, **stats) -> None`: Update progress bar statistics

**Example:**

```python
with progress.time_loop(Tf=1.0, dt=0.01, desc="Advection") as pbar:
    while pbar.continue_loop():
        # Variable time step support
        dt = compute_adaptive_dt()
        pbar.advance_time(dt)
        pbar.update_stats(mesh=u.mesh)
```

### `iteration(total, desc="Iterations", disable=False, **kwargs)`

Create a fixed-count iteration progress bar.

**Parameters:**
- `total` (int): Total number of iterations
- `desc` (str): Description for progress bar (default: "Iterations")
- `disable` (bool): If True, disable progress bar display
- `**kwargs`: Additional arguments passed to tqdm

**Returns:** `IterationLoop` context manager

**Methods:**
- `update(n=1) -> None`: Update progress by n iterations
- `set_postfix(**kwargs) -> None`: Update postfix statistics

**Example:**

```python
with progress.iteration(total=100, desc="Optimization") as pbar:
    for i in range(100):
        # Do work
        pbar.update()
        pbar.set_postfix(loss=compute_loss())
```

### `mesh_adaptation(mesh, desc="Mesh adaptation", disable=False)`

Context manager for mesh adaptation operations.

**Parameters:**
- `mesh`: Mesh being adapted
- `desc` (str): Description for progress (default: "Mesh adaptation")
- `disable` (bool): If True, disable progress output

**Example:**

```python
with progress.mesh_adaptation(mesh) as stats:
    MRadaptation(config)
    # Automatically shows: "Adapting mesh... done (0.123s) | 15234 -> 14876 cells (-358)"
```

### `MeshStatistics(enable_level_breakdown=False)`

Track and compute mesh statistics efficiently.

**Parameters:**
- `enable_level_breakdown` (bool): Track cell counts per level

**Methods:**
- `update(mesh) -> None`: Update statistics from current mesh
- `get_summary() -> str`: Get formatted summary string
- `get_level_breakdown() -> str`: Get cells per level

**Properties:**
- `n_cells` (int): Total number of cells
- `min_level` (int): Minimum refinement level
- `max_level` (int): Maximum refinement level

**Example:**

```python
stats = MeshStatistics(enable_level_breakdown=True)
stats.update(mesh)
print(stats.get_summary())  # "15234 cells [4-10]"
print(stats.get_level_breakdown())  # "L4: 1024, L5: 2048, ..."
```

### `compute_mesh_stats(mesh) -> Dict[str, int]`

Convenience function to compute mesh statistics in one call.

**Returns:** Dictionary with keys: 'n_cells', 'min_level', 'max_level'

**Example:**

```python
stats = compute_mesh_stats(mesh)
print(f"Simulation has {stats['n_cells']} cells")
```

## Complete Example

Here's a complete example showing all features:

```python
#!/usr/bin/env python3
import samurai_python as sam
from samurai_python.utils import progress
from pathlib import Path

def main():
    # Setup
    box = sam.Box2D([0.0, 0.0], [1.0, 1.0])
    config = sam.MeshConfig2D()
    config.min_level = 4
    config.max_level = 10
    config.disable_minimal_ghost_width()

    mesh = sam.MRMesh2D(box, config)
    u = sam.field.zeros(mesh, "u")
    unp1 = sam.field.zeros(mesh, "unp1")

    # Initialize
    init_circular(u, center=(0.3, 0.3), radius=0.2)
    sam.make_dirichlet_bc(u, 0.0)

    # Initial adaptation
    MRadaptation = sam.make_MRAdapt(u)
    mra_config = sam.MRAConfig()
    mra_config.epsilon = 2e-4
    mra_config.regularity = 1

    print("Initial adaptation...")
    MRadaptation(mra_config)

    # Time stepping with progress bar
    Tf = 0.1
    dt = 0.001
    save_interval = 10

    output_path = Path("./results")
    output_path.mkdir(parents=True, exist_ok=True)

    with progress.time_loop(Tf=Tf, dt=dt, desc="Advection 2D") as pbar:
        it = 0
        while pbar.continue_loop():
            # Mesh adaptation
            MRadaptation(mra_config)
            unp1.resize()
            sam.update_ghost_mr(u)

            # Time step
            pbar.advance_time(dt)
            it += 1

            # Compute flux and update
            velocity = [1.0, 1.0]
            upwind_result = sam.operators.upwind(velocity, u)
            unp1.assign(u - dt * upwind_result)
            sam.swap_field_arrays_2d(u, unp1)

            # Update progress bar with mesh statistics
            if it % save_interval == 0:
                pbar.update_stats(
                    mesh=u.mesh,
                    max_val=float(max(u))
                )
                sam.save(str(output_path), f"sol_{it:05d}", u)

    print("Simulation complete!")

if __name__ == "__main__":
    main()
```

## Advanced Features

### Variable Time Steps

The progress bar handles variable time steps automatically:

```python
with progress.time_loop(Tf=1.0, dt=0.01) as pbar:
    while pbar.continue_loop():
        # Compute adaptive time step
        dt = compute_cfl_based_dt()
        pbar.advance_time(dt)
```

### Custom tqdm Arguments

Pass any tqdm arguments:

```python
with progress.time_loop(
    Tf=1.0,
    dt=0.01,
    desc="Simulation",
    colour='green',
    ncols=120,
    ascii=True
) as pbar:
    ...
```

### Disable Progress Bar

For automated testing or logging:

```python
with progress.time_loop(Tf=1.0, dt=0.01, disable=True) as pbar:
    ...
```

### Matplotlib Integration

Works with matplotlib interactive mode:

```python
import matplotlib.pyplot as plt

plt.ion()  # Interactive mode
fig, ax = plt.subplots()

with progress.time_loop(Tf=1.0, dt=0.01) as pbar:
    while pbar.continue_loop():
        # Simulation step
        pbar.advance_time(dt)

        # Update plot
        ax.clear()
        plot_field(u, ax)
        plt.pause(0.001)

plt.ioff()
plt.show()
```

## Performance Notes

1. **Mesh Statistics Caching**: `MeshStatistics` caches results to avoid recomputation
2. **Minimal Overhead**: Progress bars add < 1% overhead to simulation time
3. **tqdm Efficiency**: tqdm is highly optimized and handles large iteration counts well

## Troubleshooting

### tqdm not available

If tqdm is not installed, the progress bar automatically disables itself and shows a warning:

```
Warning: tqdm not available, progress bar disabled
```

### Progress bar not showing

Make sure you're not redirecting stdout. tqdm requires access to the terminal.

### In Jupyter notebooks

tqdm automatically detects Jupyter and uses a notebook-compatible progress bar. No changes needed.

## Files

- `/home/sbstndbs/sbstndbs/samurai/python/src/samurai_python/utils/__init__.py`
- `/home/sbstndbs/sbstndbs/samurai/python/src/samurai_python/utils/progress/__init__.py`
- `/home/sbstndbs/sbstndbs/samurai/python/src/samurai_python/utils/progress/core.py`
- `/home/sbstndbs/sbstndbs/samurai/python/src/samurai_python/utils/progress/stats.py`
