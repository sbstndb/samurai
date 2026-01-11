# Field Namespace API Documentation (v0.30.0)

## Overview

**BREAKING CHANGE in v0.30.0**: The old module-level factory functions (`make_scalar_field`, `make_vector_field`, `swap_field_arrays_*`) have been **removed**. Only the `sam.field.scalar()` and `sam.field.vector()` namespace API remains.

The `sam.field` namespace provides a cleaner, more Pythonic API for creating scalar and vector fields in Samurai Python bindings.

## Quick Start

### Creating Scalar Fields

```python
import samurai_python as sam

# 1D scalar field with default init (0.0)
u = sam.field.scalar(mesh_1d, "u")

# 1D scalar field with explicit init value
u = sam.field.scalar(mesh_1d, "u", init=1.5)

# 2D scalar field
u = sam.field.scalar(mesh_2d, "temperature", init=20.0)

# 3D scalar field
u = sam.field.scalar(mesh_3d, "pressure", init=101325.0)
```

### Creating Vector Fields

```python
import samurai_python as sam

# 2D vector field with 2 components (default)
velocity = sam.field.vector(mesh_2d, "velocity")

# 2D vector field with 2 components and explicit init
velocity = sam.field.vector(mesh_2d, "velocity", n_components=2, init=1.0)

# 2D vector field with 3 components
magnetic_field = sam.field.vector(mesh_2d, "B", n_components=3, init=0.0)

# 3D vector field with 3 components
v = sam.field.vector(mesh_3d, "v", n_components=3, init=0.0)
```

## API Reference

### sam.field.scalar()

Create a scalar field on an adaptive mesh.

**Signature**:
```python
sam.field.scalar(mesh, name, init=0.0) -> ScalarField
```

**Parameters**:
- `mesh` (MRMesh): The mesh to define the field on (1D, 2D, or 3D)
- `name` (str): Field identifier
- `init` (float, optional): Initial value for all cells (default: 0.0)

**Returns**:
- `ScalarField1D`, `ScalarField2D`, or `ScalarField3D` depending on mesh dimension

**Example**:
```python
u = sam.field.scalar(mesh_2d, "u", init=1.0)
print(u.name)  # "u"
print(u.dim)   # 2
```

### sam.field.vector()

Create a vector field on an adaptive mesh.

**Signature**:
```python
sam.field.vector(mesh, name, n_components=2, init=0.0) -> VectorField
```

**Parameters**:
- `mesh` (MRMesh): The mesh to define the field on (1D, 2D, or 3D)
- `name` (str): Field identifier
- `n_components` (int, optional): Number of components (2 or 3, default: 2)
- `init` (float, optional): Initial value for all components and cells (default: 0.0)

**Returns**:
- `VectorField{1D,2D,3D}_{2,3}` depending on mesh dimension and component count

**Example**:
```python
vel = sam.field.vector(mesh_2d, "vel", n_components=2, init=0.0)
print(vel.name)        # "vel"
print(vel.dim)         # 2
print(vel.n_components) # 2
```

## Common Usage Patterns

### Time Stepping (RK3)

```python
# Create fields for TVD-RK3 time stepping
u = sam.field.scalar(mesh, "u", init=0.0)
u1 = sam.field.scalar(mesh, "u1", init=0.0)
u2 = sam.field.scalar(mesh, "u2", init=0.0)
unp1 = sam.field.scalar(mesh, "unp1", init=0.0)

# RK3 stages
u1.assign(u - dt * sam.upwind(velocity, u))
u2.assign(3./4 * u + 1./4 * (u1 - dt * sam.upwind(velocity, u1)))
unp1.assign(1./3 * u + 2./3 * (u2 - dt * sam.upwind(velocity, u2)))

# Efficient swap (pure Python, no C++ helper needed)
u.array, unp1.array = unp1.array, u.array
u_ghost, unp1_ghost = unp1.ghosts_updated(), u.ghosts_updated()
u.ghosts_updated = u_ghost
unp1.ghosts_updated = unp1_ghost
```

### Burgers Equation

```python
# Create vector field for Burgers equation
u = sam.field.vector(mesh_2d, "u", n_components=2, init=0.0)
u1 = sam.field.vector(mesh_2d, "u1", n_components=2, init=0.0)
u2 = sam.field.vector(mesh_2d, "u2", n_components=2, init=0.0)
unp1 = sam.field.vector(mesh_2d, "unp1", n_components=2, init=0.0)

# Time stepping with WENO5 convection
flux1 = sam.make_convection_weno5(u)
u1.assign(u - dt * flux1)

flux2 = sam.make_convection_weno5(u1)
u2.assign(3./4 * u + 1./4 * (u1 - dt * flux2))

flux3 = sam.make_convection_weno5(u2)
unp1.assign(1./3 * u + 2./3 * (u2 - dt * flux3))

# Swap
u.array, unp1.array = unp1.array, u.array
u_ghost, unp1_ghost = unp1.ghosts_updated(), u.ghosts_updated()
u.ghosts_updated = u_ghost
unp1.ghosts_updated = unp1_ghost
```

## Migration from v0.29.x to v0.30.0

### Removed Functions

The following functions have been **removed** in v0.30.0:

```python
# REMOVED - use sam.field.scalar() instead
sam.make_scalar_field(mesh, "u", 0.0)  # ❌ No longer exists

# REMOVED - use sam.field.vector() instead
sam.make_vector_field(mesh, "v", 2, 0.0)  # ❌ No longer exists

# REMOVED - use pure Python swap instead
sam.swap_field_arrays_2d(u, unp1)  # ❌ No longer exists
sam.swap_field_arrays_1d(u, unp1)  # ❌ No longer exists
sam.swap_field_arrays_3d(u, unp1)  # ❌ No longer exists
```

### Migration Guide

**Old API** (v0.29.x and earlier):
```python
u = sam.make_scalar_field(mesh, "u", 0.0)
vel = sam.make_vector_field(mesh, "vel", 2, 0.0)
```

**New API** (v0.30.0+):
```python
u = sam.field.scalar(mesh, "u", init=0.0)
vel = sam.field.vector(mesh, "vel", n_components=2, init=0.0)
```

**Old swap helper** (v0.29.x):
```python
sam.swap_field_arrays_2d(u, unp1)
```

**New pure Python swap** (v0.30.0+):
```python
u.array, unp1.array = unp1.array, u.array
u_ghost, unp1_ghost = unp1.ghosts_updated(), u.ghosts_updated()
u.ghosts_updated = u_ghost
unp1.ghosts_updated = unp1_ghost
```

### Class Constructors Still Available

Direct class constructors still work (for backward compatibility with existing code):

```python
# These still work
u = sam.ScalarField2D("u", mesh, 0.0)
vel = sam.VectorField2D_2("vel", mesh, 0.0)
```

However, the namespace API is **recommended** for new code:
- More Pythonic
- Keyword arguments for clarity
- Dimension automatically inferred from mesh type

## Benefits of the Namespace API

1. **Dimension Inference**: No need to specify 1D/2D/3D in function names
2. **Keyword Arguments**: `init=` and `n_components=` are more readable
3. **Consistent**: Same pattern for both scalar and vector fields
4. **Organized**: All field creation functions in `sam.field` namespace

## Version History

- **v0.30.0** (2026-01-07): Breaking change - removed `make_scalar_field`, `make_vector_field`, and `swap_field_arrays_*` functions
- **v0.29.0** (2026-01-06): Introduced `sam.field.scalar()` and `sam.field.vector()` namespace API alongside old functions
- **v0.28.0** and earlier: Only module-level factory functions available
