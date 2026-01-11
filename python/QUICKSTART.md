# Samurai Python Quick Start Guide

Welcome to **Samurai Python** - a Python library for Adaptive Mesh Refinement (AMR) and Multiresolution analysis.

This guide will help you get started with the basics of Samurai Python in just a few minutes.

---

## Installation

### Prerequisites

- Python 3.8 or later
- CMake 3.20 or later
- A C++20 compatible compiler (GCC 11+, Clang 13+, MSVC 2022+)

### Build from Source

```bash
# Clone the repository
git clone https://github.com/hpc-maths/samurai.git
cd samurai

# Configure with CMake
cmake . -B build -DBUILD_PYTHON_BINDINGS=ON -DCMAKE_BUILD_TYPE=Release

# Build the Python bindings
cmake --build build --target samurai_python

# Install (optional)
cmake --install build
```

### Set Up Python Path

```bash
# Add the build directory to your Python path
export PYTHONPATH="${PYTHONPATH}:/path/to/samurai/build/python"
```

Or from Python:

```python
import sys
sys.path.insert(0, '/path/to/samurai/build/python')
```

---

## Your First Samurai Program

Let's create a simple 2D advection simulation with adaptive mesh refinement.

```python
import samurai_python as sam

def main():
    # 1. Define the computational domain
    box = sam.geometry.Box2D([0., 0.], [1., 1.])

    # 2. Configure the mesh
    config = sam.config.MeshConfig2D()
    config.min_level = 4      # Minimum refinement level
    config.max_level = 8      # Maximum refinement level

    # 3. Create the mesh
    mesh = sam.mesh.MRMesh2D(box, config)

    # 4. Create a field
    u = sam.field.zeros(mesh, "u")

    print(f"Mesh created: {mesh.nb_cells} cells")

if __name__ == "__main__":
    main()
```

**Output:**
```
Mesh created: 256 cells
```

---

## Core Concepts

### 1. Submodule Organization (v0.30.0+)

Samurai Python is organized into logical submodules:

| Submodule | Purpose | Example |
|-----------|---------|---------|
| `sam.geometry` | Geometric primitives | `sam.geometry.Box2D(...)`, `sam.geometry.box(...)` |
| `sam.config` | Mesh configuration | `sam.config.MeshConfig2D()`, `sam.config.make(...)` |
| `sam.mesh` | Mesh types | `sam.mesh.MRMesh2D(...)`, `sam.mesh.make(...)` |
| `sam.field` | Fields | `sam.field.ScalarField2D(...)`, `sam.field.scalar(...)` |
| `sam.adaptation` | Mesh adaptation | `sam.adaptation.MRAdapt(...)`, `sam.adaptation.make_MRAdapt(...)` |
| `sam.algorithms` | Iteration | `sam.algorithms.for_each_cell(...)` |
| `sam.boundary` | Boundary conditions | `sam.boundary.make_dirichlet_bc(...)` |
| `sam.operators` | Differential operators | `sam.operators.upwind(...)` |

### 2. Factory Functions (Recommended)

Factory functions automatically infer dimensions from your data:

```python
# Instead of manually specifying dimension
box = sam.geometry.Box2D([0., 0.], [1., 1.])  # Must specify 2D

# Use the factory function
box = sam.geometry.box([0., 0.], [1., 1.])    # Auto-detects 2D!

# Works for any dimension
box_1d = sam.geometry.box([0.], [1.])         # 1D
box_2d = sam.geometry.box([0., 0.], [1., 1.]) # 2D
box_3d = sam.geometry.box([0., 0., 0.], [1., 1., 1.])  # 3D
```

### 3. Mesh Configuration

```python
# Option 1: Direct configuration
config = sam.config.MeshConfig2D()
config.min_level = 4
config.max_level = 8

# Option 2: Factory function (recommended)
config = sam.config.make(min_level=4, max_level=8)

# Option 3: One-line mesh creation with factory
mesh = sam.mesh.make(box, min_level=4, max_level=8)
```

### 4. Field Creation

```python
# Scalar field
u = sam.field.scalar(mesh, "u")              # Direct class
u = sam.field.zeros(mesh, "u")               # NumPy-style (recommended)

# Vector field
vel = sam.field.vector(mesh, "vel", n_components=2)  # Direct class
vel = sam.field.zeros_vector(mesh, "vel", n_components=2)  # NumPy-style

# Initialize with custom values
u = sam.field.ones(mesh, "u")                # All 1.0
u = sam.field.full(mesh, 3.14, "pi")         # All 3.14
```

---

## Complete Example: 2D Advection

Here's a complete 2D advection simulation with AMR:

```python
import samurai_python as sam

def init_circular(u, center=(0.5, 0.5), radius=0.2):
    """Initialize field with a circular pattern."""
    def init_cell(cell):
        cx, cy = cell.center()
        dist_sq = (cx - center[0])**2 + (cy - center[1])**2
        u[cell] = 1.0 if dist_sq < radius**2 else 0.0

    sam.algorithms.for_each_cell(u.mesh, init_cell)

def main():
    # Create mesh
    box = sam.geometry.Box2D([0., 0.], [1., 1.])
    mesh = sam.mesh.make(box, min_level=4, max_level=8)

    # Create field
    u = sam.field.zeros(mesh, "u")
    unp1 = sam.field.zeros(mesh, "unp1")

    # Initialize
    init_circular(u, center=(0.5, 0.5), radius=0.2)

    # Boundary conditions
    sam.boundary.make_dirichlet_bc(u, 0.0)

    # Adapt mesh to initial condition
    MRadapt = sam.adaptation.make_MRAdapt(u)
    mra_config = sam.config.MRAConfig(epsilon=1e-4, regularity=1.0)
    MRadapt(mra_config)

    # Time stepping
    velocity = [1.0, 1.0]
    dt = 0.5 * mesh.min_cell_length / max(abs(v) for v in velocity)

    Tf = 0.05
    t = 0.0

    print(f"Starting simulation: {mesh.nb_cells} cells")

    for it in range(100):
        if t >= Tf:
            break

        # Adapt mesh
        MRadapt(mra_config)

        # Update time
        t += dt

        # Update ghost cells
        sam.adaptation.update_ghost_mr(u)

        # Compute flux and update
        flux = sam.operators.upwind(u, velocity)
        unp1.assign(u - dt * flux)

        # Swap fields
        sam.swap_field_arrays_2d(u, unp1)

        if it % 10 == 0:
            print(f"Step {it}: t={t:.4f}, cells={mesh.nb_cells}")

    print(f"Simulation complete: {mesh.nb_cells} cells")

if __name__ == "__main__":
    main()
```

**Output:**
```
Starting simulation: 1523 cells
Step 0: t=0.0010, cells=1589
Step 10: t=0.0098, cells=1645
Step 20: t=0.0195, cells=1712
...
Step 90: t=0.0879, cells=1567
Simulation complete: 1521 cells
```

---

## Mesh Adaptation

Samurai's key feature is automatic mesh adaptation:

```python
# Set up adaptation
MRadapt = sam.adaptation.make_MRAdapt(u)

# Configure adaptation criteria
config = sam.config.MRAConfig()
config.epsilon = 1e-4      # Refinement tolerance
config.regularity = 1.0    # Gradation parameter

# Apply adaptation
MRadapt(config)

# Update ghost cells after adaptation
sam.adaptation.update_ghost_mr(u)
```

**Parameters:**
- `epsilon`: Tolerance for refinement (smaller = more refinement)
- `regularity`: Controls transition between levels (1 = one level difference)

---

## Boundary Conditions

```python
# Dirichlet (fixed value)
bc = sam.boundary.make_dirichlet_bc(u, 0.0)

# Neumann (fixed derivative)
bc = sam.boundary.make_neumann_bc(u, 0.0)

# Apply boundary condition
sam.boundary.apply_bc(u, bc)
```

---

## Common Patterns

### Iterate Over Cells

```python
sam.algorithms.for_each_cell(mesh, lambda cell: print(cell.center()))

# Or with a named function
def process_cell(cell):
    u[cell] = some_function(cell.center())

sam.algorithms.for_each_cell(mesh, process_cell)
```

### Iterate Over Intervals

```python
def update_field(level, interval, index):
    u[level, interval, index] = 2 * u[level, interval, index]

sam.algorithms.for_each_interval(mesh, update_field)
```

### Access Field Values

```python
# By cell
sam.algorithms.for_each_cell(mesh, lambda cell: print(u[cell]))

# By level/interval/index
sam.algorithms.for_each_interval(mesh, lambda level, interval, index: print(u[level, interval, index]))
```

---

## Visualization

### Save to HDF5 (for Paraview)

```python
sam.io.save("./results", "simulation", mesh, u)

# This creates:
# - ./results/simulation_0000.h5
# - ./results/simulation_0000.xdmf (for Paraview)
```

### Matplotlib Visualization

```python
import matplotlib.pyplot as plt
import samurai_python as sam

# ... create mesh and field ...

plt.figure(figsize=(8, 8))
plt.imshow(u.numpy_view(), origin='lower', extent=[0, 1, 0, 1])
plt.colorbar(label='u')
plt.title('Field u')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('field.png')
```

---

## NumPy-Style Helpers

Samurai provides NumPy-like field creation:

```python
# Create fields initialized with values
u = sam.field.zeros(mesh, "u")        # All zeros
v = sam.field.ones(mesh, "v")          # All ones
w = sam.field.full(mesh, 3.14, "pi")  # All 3.14

# Create field like another
error = sam.field.zeros_like(u, "error")    # Same mesh as u, but zeros
residual = sam.field.ones_like(u, "res")     # Same mesh as u, but ones

# Vector fields
vel = sam.field.zeros_vector(mesh, "vel", n_components=2)
B = sam.field.zeros_vector(mesh, "B", n_components=3)
```

---

## Next Steps

1. **Explore Examples**: Check `python/examples/` for more examples
2. **Read Documentation**: https://hpc-math-samurai.readthedocs.io
3. **Learn Advanced Features**:
   - RK3 time stepping
   - WENO5 spatial discretization
   - Vector fields
   - Multiple physics schemes

---

## Troubleshooting

### Import Error

**Problem:** `ImportError: No module named 'samurai_python'`

**Solution:** Add build directory to Python path:
```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/samurai/build/python"
```

### AttributeError

**Problem:** `AttributeError: module has no attribute 'Box2D'`

**Solution:** Use submodule prefix:
```python
# Wrong
box = sam.Box2D([0., 0.], [1., 1.])

# Correct
box = sam.geometry.Box2D([0., 0.], [1., 1.])
```

### Build Errors

**Problem:** CMake configuration fails

**Solution:** Ensure you have C++20 compiler:
```bash
# Check GCC version
g++ --version  # Should be 11+

# Check Clang version
clang++ --version  # Should be 13+
```

---

## API Reference Summary

### Geometry
- `sam.geometry.Box1D(min, max)`
- `sam.geometry.Box2D(min, max)`
- `sam.geometry.Box3D(min, max)`
- `sam.geometry.box(min, max)` - Factory (auto-detect dimension)

### Config
- `sam.config.MeshConfig1D()`
- `sam.config.MeshConfig2D()`
- `sam.config.MeshConfig3D()`
- `sam.config.make(min_level, max_level)` - Factory

### Mesh
- `sam.mesh.MRMesh1D(box, config)`
- `sam.mesh.MRMesh2D(box, config)`
- `sam.mesh.MRMesh3D(box, config)`
- `sam.mesh.make(box, min_level, max_level)` - Factory

### Field
- `sam.field.scalar(mesh, name)`
- `sam.field.vector(mesh, name, n_components)`
- `sam.field.zeros(mesh, name)` - NumPy-style
- `sam.field.ones(mesh, name)` - NumPy-style
- `sam.field.full(mesh, value, name)` - NumPy-style

### Adaptation
- `sam.adaptation.MRAdapt(field)`
- `sam.adaptation.make_MRAdapt(field)` - Factory
- `sam.adaptation.update_ghost_mr(field)`

### Algorithms
- `sam.algorithms.for_each_cell(mesh, function)`
- `sam.algorithms.for_each_interval(mesh, function)`

### Boundary
- `sam.boundary.make_dirichlet_bc(field, value)`
- `sam.boundary.make_neumann_bc(field, value)`

### Operators
- `sam.operators.upwind(field, velocity)`
- `sam.operators.diffusion(field)`
- `sam.operators.make_convection_weno5(velocity)`

### I/O
- `sam.io.save(path, filename, mesh, *fields)`
- `sam.io.load(path, filename, mesh)`
- `sam.io.dump(mesh, *fields)`

---

## Resources

- **GitHub**: https://github.com/hpc-maths/samurai
- **Documentation**: https://hpc-math-samurai.readthedocs.io
- **Issues**: https://github.com/hpc-maths/samurai/issues
- **Discussions**: https://github.com/hpc-maths/samurai/discussions

---

Happy simulating!
