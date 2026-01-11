# Progress Bar Examples for Samurai Python

This directory contains example scripts demonstrating how to use progress bars in Samurai Python simulations.

## Overview

Progress bars provide visual feedback during long-running simulations, showing:
- Current progress (percentage and step count)
- Elapsed time
- Live metrics (time step, cell count, mesh levels, etc.)
- Estimated time remaining (with tqdm)

## Example Scripts

### 1. `demo_progress.py` - Progress Bar API Demonstration

**Purpose**: Standalone demonstration of the progress bar API without running a full simulation.

**Features**:
- Basic progress bar usage
- Time stepping simulation style
- Mesh adaptation tracking
- Custom iteration (nonlinear solver)
- Nested progress tracking
- TQDM integration with automatic fallback

**Usage**:
```bash
python demo_progress.py
```

**What you'll learn**:
- How to create a progress bar
- How to update progress with custom metrics
- How to use context managers for automatic cleanup
- How to nest progress bars for multi-loop scenarios
- How to use tqdm for richer progress display

**No samurai_python required** - This is a pure Python demonstration!

---

### 2. `advection_2d_progress.py` - Full Simulation with Progress Bars

**Purpose**: Complete 2D advection simulation with progress bars integrated.

**Features**:
- 2D adaptive mesh refinement (AMR)
- Upwind operator for advection equation
- Progress bar during time stepping
- Live metrics: time, cell count, min/max mesh level
- Real-time matplotlib visualization
- HDF5 output for Paraview

**Equation**: `∂u/∂t + a·∇u = 0` with velocity `a = (1, 1)`

**Usage**:
```bash
python advection_2d_progress.py
```

**What you'll see**:
```
Time stepping: [████████████░░░░░░░░░░░░░░░░░░░░░░░░]  40.0% (200/500)
               elapsed: 12.3s | time=0.040, cells=1523, min_lvl=4, max_lvl=8
```

**Output**:
- HDF5 files in `./results/FV_advection_2d_progress_*.h5`
- XDMF files for Paraview visualization

---

### 3. `burgers_2d_progress.py` - RK3 Time Stepping with Progress

**Purpose**: Burgers equation simulation with RK3 time stepping and variable dt.

**Features**:
- 2D Burgers equation: `∂u/∂t + u·∇u = 0`
- RK3 (Runge-Kutta 3rd order) time integration
- WENO5 spatial discretization
- Variable time stepping with automatic adjustment
- Progress bar with live metrics
- Vector field operations

**Usage**:
```bash
python burgers_2d_progress.py
```

**What you'll see**:
```
Burgers RK3: [████████████████░░░░░░░░░░░░░░░░░░░░░]  50.0% (250/500)
             elapsed: 45.2s | time=0.500, dt=0.002, cells=2145, max_vel=1.234
```

**Output**:
- HDF5 files in `./results/burgers_2d_progress_*.h5`
- XDMF files for Paraview visualization

---

## Using Progress Bars in Your Own Code

### Basic Usage

```python
from demo_progress import ProgressBar

# Create a progress bar
with ProgressBar(total=100, desc="Processing") as pbar:
    for i in range(100):
        # Your computation here
        result = do_work(i)

        # Update progress with custom metrics
        pbar.update(1, metrics={"value": result, "iteration": i})
```

### Time Stepping Example

```python
from demo_progress import ProgressBar
import samurai_python as sam

# Simulation parameters
Tf = 1.0
dt = 0.01
nt = int(Tf / dt)

with ProgressBar(total=nt, desc="Time stepping") as pbar:
    t = 0.0
    for it in range(nt):
        # Physics update
        u = compute_next_step(u, dt)
        t += dt

        # Update progress every 10 steps
        if (it + 1) % 10 == 0:
            pbar.update(10, metrics={
                "time": f"{t:.3f}",
                "cells": mesh.nb_cells
            })
```

### Mesh Adaptation Example

```python
from demo_progress import ProgressBar
import samurai_python as sam

MRadaptation = sam.adaptation.make_MRAdapt(u)
mra_config = sam.config.MRAConfig()

# Adapt for multiple cycles
n_cycles = 10
with ProgressBar(total=n_cycles, desc="Mesh adaptation") as pbar:
    for i in range(n_cycles):
        MRadaptation(mra_config)

        # Get mesh statistics
        mesh_stats = get_mesh_stats(mesh)

        # Update progress
        pbar.update(1, metrics={
            "cells": mesh_stats['cells'],
            "min_level": mesh_stats['min_level'],
            "max_level": mesh_stats['max_level']
        })
```

### Using TQDM (if installed)

```python
from demo_progress import TQDMProgressBar

with TQDMProgressBar(total=1000, desc="Simulation") as pbar:
    for i in range(1000):
        # Your work here
        result = compute(i)

        # Update with keyword arguments
        pbar.update(1, cells=1000, dt=0.01, residual=1e-5)
```

## Progress Bar API Reference

### `ProgressBar(total, desc, metrics)`

Basic progress bar implementation (no external dependencies).

**Parameters**:
- `total` (int): Total number of iterations/steps
- `desc` (str): Description text to display
- `metrics` (dict, optional): Initial metric values

**Methods**:
- `update(n=1, metrics=None)`: Advance progress by n steps
- `set_description(desc)`: Update description text
- `close()`: Close the progress bar

**Context Manager**:
```python
with ProgressBar(total=100, desc="Processing") as pbar:
    # Automatically closed when exiting context
    for i in range(100):
        pbar.update(1)
```

### `TQDMProgressBar(total, desc)`

Rich progress bar using tqdm if available, falls back to basic ProgressBar.

**Parameters**:
- `total` (int): Total number of iterations/steps
- `desc` (str): Description text to display

**Methods**:
- `update(n=1, **metrics)`: Advance progress with keyword metric arguments
- `set_description(desc)`: Update description text
- `close()`: Close the progress bar

## Features

### Custom Metrics

Display live statistics alongside progress:
```python
pbar.update(1, metrics={
    "time": f"{t:.3f}",
    "cells": 1523,
    "dt": 0.002,
    "residual": f"{res:.2e}"
})
```

### Progress Bar Display

**Basic ProgressBar**:
```
Time stepping: [████████████░░░░░░░░░░░░░░░░░░░░░░░░]  40.0% (200/500)
               elapsed: 12.3s | time=0.040, cells=1523, min_lvl=4, max_lvl=8
```

**TQDM ProgressBar** (if tqdm installed):
```
Burgers RK3:  40%|████████████▌     | 200/500 [00:12<00:18, 16.5iter/s,
             cells=1523, dt=0.002, max_vel=1.23]
```

### Best Practices

1. **Update frequency**: Don't update every single iteration. Update every N iterations to reduce overhead.
   ```python
   if (it + 1) % 10 == 0:
       pbar.update(10)
   ```

2. **Metric formatting**: Format metrics for consistent display width
   ```python
   metrics={"time": f"{t:.3f}", "cells": n_cells}  # Good
   metrics={"time": t, "cells": n_cells}           # Inconsistent
   ```

3. **Context managers**: Always use context managers for automatic cleanup
   ```python
   with ProgressBar(total=100, desc="Processing") as pbar:
       # Your code here
       pass  # Automatically closed
   ```

4. **Error handling**: Progress bars handle exceptions gracefully
   ```python
   try:
       with ProgressBar(total=100, desc="Processing") as pbar:
           for i in range(100):
               if error_condition:
                   raise ValueError("Error!")
               pbar.update(1)
   except ValueError:
       print("Error occurred, progress bar closed automatically")
   ```

## Requirements

### For `demo_progress.py`
- Python 3.6+
- No external dependencies (uses built-in modules only)
- Optional: `tqdm` for richer progress display

### For `advection_2d_progress.py` and `burgers_2d_progress.py`
- Python 3.6+
- `samurai_python` module (built from source)
- `matplotlib` (for real-time visualization, optional)
- `tqdm` (optional, for richer progress display)

## Installation

### Install tqdm (optional but recommended)
```bash
pip install tqdm
```

### Build samurai_python
```bash
cd /path/to/samurai
cmake . -B build -DBUILD_DEMOS=ON
cmake --build build --target samurai_python
```

## Running the Examples

### 1. Run the standalone demo
```bash
cd python/examples
python demo_progress.py
```

### 2. Run advection with progress
```bash
cd python/examples
python advection_2d_progress.py
```

### 3. Run Burgers with progress
```bash
cd python/examples
python burgers_2d_progress.py
```

## Output Examples

### Console Output

**Time stepping progress**:
```
Time stepping: [████████████████░░░░░░░░░░░░░░░░░░░]  50.0% (250/500)
               elapsed: 30.2s | time=0.250, cells=1834, min_lvl=4, max_lvl=8
```

**Mesh adaptation progress**:
```
Mesh adaptation: [████████████████████░░░░░░░░░░░░░]  60.0% (6/10)
                 elapsed: 0.6s | cells=1500, min_level=5, max_level=9
```

### Real-time Visualization

The examples support real-time matplotlib visualization showing:
- Field values (color plot)
- Mesh structure (cell boundaries)
- Progress information in title

Disable by setting:
```python
enable_realtime_viz = False
```

## Troubleshooting

### Progress bar not displaying
- Ensure you're using a terminal that supports carriage returns (`\r`)
- Try running with `python -u` for unbuffered output
- Check that `sys.stdout.flush()` is being called

### Metrics not updating
- Ensure you're calling `update()` with `metrics` parameter
- Check that metric names are strings and values are printable
- Use formatted strings for consistent display: `f"{value:.3f}"`

### TQDM not working
- Install tqdm: `pip install tqdm`
- Falls back automatically to basic ProgressBar if not available
- Check tqdm version: `python -c "import tqdm; print(tqdm.__version__)"`

## Additional Resources

- Samurai documentation: https://hpc-math-samurai.readthedocs.io
- TQDM documentation: https://tqdm.github.io
- Python C++ bindings: `/python/src/bindings/`

## Contributing

To add progress bars to other examples:

1. Import from demo_progress:
   ```python
   from demo_progress import ProgressBar
   ```

2. Create progress bar before main loop:
   ```python
   pbar = ProgressBar(total=nt, desc="Time stepping")
   ```

3. Update progress in loop:
   ```python
   pbar.update(1, metrics={"time": f"{t:.3f}"})
   ```

4. Close progress bar after loop:
   ```python
   pbar.close()
   ```

Or use context manager for automatic cleanup:
```python
with ProgressBar(total=nt, desc="Time stepping") as pbar:
    for it in range(nt):
        # Your code here
        pbar.update(1, metrics={"time": f"{t:.3f}"})
```
