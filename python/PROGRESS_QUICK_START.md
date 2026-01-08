# Progress Bar API - Quick Start Guide

## Installation

The progress bar is included with Samurai Python. Just ensure tqdm is installed:

```bash
pip install tqdm
```

## Basic Usage

### 1. Simple Time Loop

```python
from samurai_python.utils import progress

with progress.time_loop(Tf=1.0, dt=0.01) as pbar:
    while pbar.continue_loop():
        # Your simulation code here
        pbar.advance_time(dt)
```

### 2. With Mesh Statistics

```python
with progress.time_loop(Tf=1.0, dt=0.01) as pbar:
    while pbar.continue_loop():
        # Simulation step
        pbar.advance_time(dt)
        pbar.update_stats(mesh=u.mesh)  # Shows cells and levels
```

### 3. With Custom Statistics

```python
with progress.time_loop(Tf=1.0, dt=0.01) as pbar:
    while pbar.continue_loop():
        pbar.advance_time(dt)
        pbar.update_stats(
            mesh=u.mesh,
            residual=1e-6,
            max_val=float(max(u))
        )
```

### 4. Iteration Loop (Fixed Count)

```python
with progress.iteration(total=100, desc="Processing") as pbar:
    for i in range(100):
        # Your code here
        pbar.update()
```

### 5. Mesh Adaptation Tracking

```python
with progress.mesh_adaptation(mesh) as stats:
    MRadaptation(config)
# Output: "Adapting mesh... done (0.123s) | 15234 -> 14876 cells (-358)"
```

## Integration in Existing Code

Replace your time loop:

```python
# Before
t = 0.0
while t < Tf:
    # simulation
    t += dt
    print(f"t = {t:.3f}")

# After
with progress.time_loop(Tf=Tf, dt=dt) as pbar:
    while pbar.continue_loop():
        # simulation
        pbar.advance_time(dt)
        pbar.update_stats(mesh=u.mesh)
```

## Common Patterns

### With Save Interval

```python
save_interval = 10
with progress.time_loop(Tf=1.0, dt=0.01) as pbar:
    it = 0
    while pbar.continue_loop():
        # simulation
        pbar.advance_time(dt)
        it += 1
        
        if it % save_interval == 0:
            pbar.update_stats(mesh=u.mesh)
            save(u, it)
```

### With Matplotlib

```python
plt.ion()
with progress.time_loop(Tf=1.0, dt=0.01) as pbar:
    while pbar.continue_loop():
        # simulation
        pbar.advance_time(dt)
        
        # Update plot
        ax.clear()
        plot(u, ax)
        plt.pause(0.001)

plt.ioff()
plt.show()
```

### Disable for Testing

```python
# No progress bar displayed
with progress.time_loop(Tf=1.0, dt=0.01, disable=True) as pbar:
    while pbar.continue_loop():
        # simulation
        pbar.advance_time(dt)
```

## API at a Glance

| Function | Purpose | Returns |
|----------|---------|---------|
| `time_loop(Tf, dt)` | Time stepping progress | `TimeLoop` context |
| `iteration(total)` | Fixed iteration count | `IterationLoop` context |
| `mesh_adaptation(mesh)` | Track adaptation | Context manager |
| `compute_mesh_stats(mesh)` | Get mesh dict | `{'n_cells', 'min_level', 'max_level'}` |

## TimeLoop Methods

| Method | Purpose |
|--------|---------|
| `continue_loop()` | Check if t < Tf |
| `advance_time(dt)` | Increment time |
| `update_stats(mesh, **kw)` | Update display |

## IterationLoop Methods

| Method | Purpose |
|--------|---------|
| `update(n=1)` | Increment counter |
| `set_postfix(**kw)` | Update display |

## Full Example

```python
#!/usr/bin/env python3
import samurai_python as sam
from samurai_python.utils import progress

# Setup
box = sam.Box2D([0.0, 0.0], [1.0, 1.0])
config = sam.MeshConfig2D()
config.min_level = 4
config.max_level = 10

mesh = sam.MRMesh2D(box, config)
u = sam.field.zeros(mesh, "u")
unp1 = sam.field.zeros(mesh, "unp1")

# Initialize
init_field(u)
sam.make_dirichlet_bc(u, 0.0)

# Adaptation
MRadaptation = sam.make_MRAdapt(u)
mra_config = sam.MRAConfig()
mra_config.epsilon = 2e-4
mra_config.regularity = 1

# Time loop with progress
Tf = 0.1
dt = 0.001

with progress.time_loop(Tf=Tf, dt=dt, desc="Advection 2D") as pbar:
    while pbar.continue_loop():
        # Adapt mesh
        MRadaptation(mra_config)
        unp1.resize()
        sam.update_ghost_mr(u)

        # Time step
        pbar.advance_time(dt)

        # Compute
        velocity = [1.0, 1.0]
        upwind = sam.operators.upwind(velocity, u)
        unp1.assign(u - dt * upwind)
        sam.swap_field_arrays_2d(u, unp1)

        # Update progress
        pbar.update_stats(mesh=u.mesh)

print("Done!")
```

## Need More?

See `PROGRESS_BAR_API.md` for complete documentation.
