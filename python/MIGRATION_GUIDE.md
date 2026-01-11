# Samurai Python Migration Guide: v0.29.x → v0.30.0+

## Breaking Change: Submodule-Only API

**Version 0.30.0** introduces a major API reorganization to solve the "type proliferation" problem. Previously, users had to manually select from 29 dimension-specific types (e.g., `sam.ScalarField2D`, `sam.VectorField2D_2`, `sam.MeshConfig2D`, etc.).

**Now**, all types are organized in logical submodules, and factory functions automatically infer dimensions.

---

## Quick Reference: Old API → New API

| Before (v0.29.x) | After (v0.30.0+) |
|-------------------|-------------------|
| `sam.Box1D([0.], [1.])` | `sam.geometry.Box1D([0.], [1.])` **or** `sam.geometry.box([0.], [1.])` |
| `sam.MeshConfig2D()` | `sam.config.MeshConfig2D()` **or** `sam.config.make(min_level=4, max_level=8)` |
| `sam.MRMesh2D(box, config)` | `sam.mesh.MRMesh2D(box, config)` **or** `sam.mesh.make(box, min_level=4, max_level=8)` |
| `sam.ScalarField2D(mesh, "u")` | `sam.field.ScalarField2D(mesh, "u")` **or** `sam.field.scalar(mesh, "u")` |
| `sam.VectorField2D_2(mesh, "vel")` | `sam.field.VectorField2D_2(mesh, "vel")` **or** `sam.field.vector(mesh, "vel", n_components=2)` |
| `sam.make_scalar_field(...)` | `sam.field.scalar(...)` |
| `sam.make_vector_field(...)` | `sam.field.vector(...)` |
| `sam.MRAdapt(...)` | `sam.adaptation.MRAdapt(...)` **or** `sam.adaptation.make_MRAdapt(...)` |
| `sam.MRAConfig(...)` | `sam.config.MRAConfig(...)` |
| `sam.Interval1D(...)` | `sam.interval.Interval1D(...)` |
| `sam.for_each_cell(...)` | `sam.algorithms.for_each_cell(...)` |
| `sam.make_dirichlet_bc(...)` | `sam.boundary.make_dirichlet_bc(...)` |

---

## Step-by-Step Migration

### Step 1: Update Imports

**Before:**
```python
import samurai_python as sam
```

**After:**
```python
import samurai_python as sam
# All functionality still available, just organized in submodules
```

### Step 2: Update Box Creation

**Before:**
```python
import samurai_python as sam
box = sam.Box2D([0., 0.], [1., 1.])
```

**After (Option 1 - Direct class):**
```python
import samurai_python as sam
box = sam.geometry.Box2D([0., 0.], [1., 1.])
```

**After (Option 2 - Factory with auto-dimension):**
```python
import samurai_python as sam
box = sam.geometry.box([0., 0.], [1., 1.])  # Automatically returns Box2D
```

### Step 3: Update Mesh Configuration

**Before:**
```python
import samurai_python as sam
config = sam.MeshConfig2D()
config.min_level = 4
config.max_level = 8
```

**After (Option 1 - Direct class):**
```python
import samurai_python as sam
config = sam.config.MeshConfig2D()
config.min_level = 4
config.max_level = 8
```

**After (Option 2 - Factory with auto-dimension):**
```python
import samurai_python as sam
config = sam.config.make(min_level=4, max_level=8)
```

### Step 4: Update Mesh Creation

**Before:**
```python
import samurai_python as sam
mesh = sam.MRMesh2D(box, config)
```

**After (Option 1 - Direct class):**
```python
import samurai_python as sam
mesh = sam.mesh.MRMesh2D(box, config)
```

**After (Option 2 - Factory with auto-dimension):**
```python
import samurai_python as sam
# Auto-detects dimension from box or config
mesh = sam.mesh.make(box, min_level=4, max_level=8)
```

### Step 5: Update Field Creation

**Before:**
```python
import samurai_python as sam
u = sam.ScalarField2D(mesh, "u")
vel = sam.VectorField2D_2(mesh, "vel")
```

**After (Option 1 - Direct class):**
```python
import samurai_python as sam
u = sam.field.ScalarField2D(mesh, "u")
vel = sam.field.VectorField2D_2(mesh, "vel")
```

**After (Option 2 - Factory with auto-dimension):**
```python
import samurai_python as sam
# Factory functions infer dimension from mesh
u = sam.field.scalar(mesh, "u")
vel = sam.field.vector(mesh, "vel", n_components=2)
```

### Step 6: Update Adaptation

**Before:**
```python
import samurai_python as sam
MRadapt = sam.MRAdapt(u)
MRadapt(sam.MRAConfig(epsilon=1e-4))
```

**After:**
```python
import samurai_python as sam
MRadapt = sam.adaptation.MRAdapt(u)  # or sam.adaptation.make_MRAdapt(u)
MRadapt(sam.config.MRAConfig(epsilon=1e-4))
```

### Step 7: Update Boundary Conditions

**Before:**
```python
import samurai_python as sam
bc = sam.make_dirichlet_bc(u, 0.0)
```

**After:**
```python
import samurai_python as sam
bc = sam.boundary.make_dirichlet_bc(u, 0.0)
# Or use the convenience alias at module level (still available):
bc = sam.make_dirichlet_bc(u, 0.0)
```

### Step 8: Update Algorithms

**Before:**
```python
import samurai_python as sam
sam.for_each_cell(mesh, lambda cell: ...)
```

**After:**
```python
import samurai_python as sam
sam.algorithms.for_each_cell(mesh, lambda cell: ...)
```

---

## New Submodule Structure

| Submodule | Contains |
|-----------|----------|
| `sam.geometry` | `Box1D`, `Box2D`, `Box3D`, `DomainBuilder1D`, etc. |
| `sam.config` | `MeshConfig1D`, `MeshConfig2D`, `MeshConfig3D`, `MRAConfig` |
| `sam.mesh` | `MRMesh1D`, `MRMesh2D`, `MRMesh3D`, `UniformMesh1D`, etc. |
| `sam.field` | `ScalarField1D`, `VectorField2D_2`, etc. + factory functions |
| `sam.interval` | `Interval1D`, `Interval2D`, `Interval3D` |
| `sam.adaptation` | `MRAdapt`, `update_ghost_mr` |
| `sam.algorithms` | `for_each_cell`, `for_each_interval` |
| `sam.boundary` | `make_dirichlet_bc`, `make_neumann_bc` |
| `sam.operators` | `upwind`, `make_convection_weno5`, etc. |
| `sam.io` | `save`, `load`, `dump`, `open_h5py` |

---

## Complete Example Migration

### Before (v0.29.x)

```python
import samurai_python as sam

# Create box
box = sam.Box2D([0., 0.], [1., 1.])

# Configure mesh
config = sam.MeshConfig2D()
config.min_level = 4
config.max_level = 8

# Create mesh
mesh = sam.MRMesh2D(box, config)

# Create fields
u = sam.ScalarField2D(mesh, "u")
vel = sam.VectorField2D_2(mesh, "velocity")

# Set up boundary condition
bc = sam.make_dirichlet_bc(u, 0.0)

# Set up adaptation
MRadapt = sam.MRAdapt(u)
MRadapt(sam.MRAConfig(epsilon=1e-4))

# Iterate
sam.for_each_cell(mesh, lambda cell: u[cell] = 0.0)
```

### After (v0.30.0+) - Direct Class Style

```python
import samurai_python as sam

# Create box
box = sam.geometry.Box2D([0., 0.], [1., 1.])

# Configure mesh
config = sam.config.MeshConfig2D()
config.min_level = 4
config.max_level = 8

# Create mesh
mesh = sam.mesh.MRMesh2D(box, config)

# Create fields
u = sam.field.ScalarField2D(mesh, "u")
vel = sam.field.VectorField2D_2(mesh, "velocity")

# Set up boundary condition
bc = sam.boundary.make_dirichlet_bc(u, 0.0)

# Set up adaptation
MRadapt = sam.adaptation.MRAdapt(u)
MRadapt(sam.config.MRAConfig(epsilon=1e-4))

# Iterate
sam.algorithms.for_each_cell(mesh, lambda cell: u[cell] = 0.0)
```

### After (v0.30.0+) - Factory Style (Recommended)

```python
import samurai_python as sam

# Create box (auto-detects 2D from array length)
box = sam.geometry.box([0., 0.], [1., 1.])

# Create mesh (auto-detects dimension from box)
mesh = sam.mesh.make(box, min_level=4, max_level=8)

# Create fields (auto-detects dimension from mesh)
u = sam.field.scalar(mesh, "u")
vel = sam.field.vector(mesh, "vel", n_components=2)

# Set up boundary condition
bc = sam.boundary.make_dirichlet_bc(u, 0.0)

# Set up adaptation
MRadapt = sam.adaptation.make_MRAdapt(u)
MRadapt(sam.config.MRAConfig(epsilon=1e-4))

# Iterate
sam.algorithms.for_each_cell(mesh, lambda cell: u[cell] = 0.0)
```

---

## NumPy-Style Field Helpers (New in v0.30.0)

Version 0.30.0 also adds NumPy-style field creation helpers:

```python
import samurai_python as sam
import numpy as np

# Before
u = sam.field.scalar(mesh, "u", init=0.0)

# After (NumPy-style)
u = sam.field.zeros(mesh, "u")
u2 = sam.field.ones(mesh, "u2")
u3 = sam.field.full(mesh, 3.14, "u3")

# Create field like another
error = sam.field.zeros_like(u, "error")
```

---

## Common Migration Patterns

### Pattern 1: Switching Between 1D, 2D, 3D

**Before (needed different code for each dimension):**
```python
# 2D version
box = sam.Box2D([0., 0.], [1., 1.])
mesh = sam.MRMesh2D(box, config)

# To switch to 3D, had to change to Box3D, MRMesh3D, etc.
```

**After (factory functions auto-detect):**
```python
# Works for any dimension
box = sam.geometry.box([0., 0.], [1., 1.])  # 2D
box = sam.geometry.box([0., 0., 0.], [1., 1., 1.])  # 3D

mesh = sam.mesh.make(box, min_level=4, max_level=8)
```

### Pattern 2: RK3 Time Stepping with Multiple Fields

**Before:**
```python
u = sam.ScalarField2D(mesh, "u")
u1 = sam.ScalarField2D(mesh, "u1")
u2 = sam.ScalarField2D(mesh, "u2")
```

**After:**
```python
u = sam.field.zeros(mesh, "u")
u1 = sam.field.zeros(mesh, "u1")
u2 = sam.field.zeros(mesh, "u2")
```

---

## Troubleshooting

### "AttributeError: module has no attribute 'Box2D'"

**Problem:** You're trying to use the old API without the submodule prefix.

**Solution:** Update to use the submodule:
```python
# Wrong
box = sam.Box2D([0., 0.], [1., 1.])

# Correct
box = sam.geometry.Box2D([0., 0.], [1., 1.])
```

### "AttributeError: module 'samurai_python.geometry' has no attribute 'mesh'"

**Problem:** Mixing submodule and module-level access incorrectly.

**Solution:** Use the correct submodule:
```python
# Wrong
mesh = sam.geometry.mesh.MRMesh2D(...)

# Correct
mesh = sam.mesh.MRMesh2D(...)
```

---

## Summary of Benefits

1. **No more type selection**: Factory functions infer dimension automatically
2. **Logical organization**: Related types grouped in submodules
3. **Better IDE support**: Auto-completion within logical submodules
4. **NumPy-style helpers**: `zeros()`, `ones()`, `full()`, `*_like()` for fields
5. **Cleaner namespace**: Reduces top-level type clutter

---

## Need Help?

- GitHub Issues: https://github.com/hpc-maths/samurai/issues
- Documentation: https://hpc-math-samurai.readthedocs.io
