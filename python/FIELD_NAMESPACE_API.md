# Field Namespace API Documentation

## Overview

The `sam.field` namespace provides a cleaner, more Pythonic API for creating scalar and vector fields in Samurai Python bindings.

## New API (Recommended)

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

## Old API (Still Supported)

The following APIs continue to work for backward compatibility:

```python
# Direct constructor (old way)
u = sam.ScalarField2D("u", mesh, 0.0)
u = sam.ScalarField3D("u", mesh, 0.0)

# Factory functions (old way)
u = sam.make_scalar_field(mesh, "u", 0.0)
v = sam.make_vector_field(mesh, "v", 2, 0.0)
```

## Comparison

### Before (Old API)

```python
# Verbose, dimension-specific
u = sam.ScalarField2D("u", mesh, 0.0)
v = sam.VectorField2D_2("velocity", mesh, 0.0)
```

### After (New API)

```python
# Cleaner, dimension inferred from mesh
u = sam.field.scalar(mesh, "u", init=0.0)
v = sam.field.vector(mesh, "velocity", n_components=2, init=0.0)
```

## Complete Example

```python
#!/usr/bin/env python3
"""Example using the new sam.field namespace API."""

import samurai_python as sam

# Create mesh
box = sam.Box2D([0.0, 0.0], [1.0, 1.0])
config = sam.MeshConfig2D(min_level=2, max_level=4)
mesh = sam.MRMesh2D(box, config)

# Create scalar field using new API
u = sam.field.scalar(mesh, "u", init=1.0)
print(f"Created {type(u).__name__} with name '{u.name}'")

# Create vector field using new API
velocity = sam.field.vector(mesh, "velocity", n_components=2, init=0.0)
print(f"Created {type(velocity).__name__} with {velocity.n_components} components")

# Access the field submodule
print("\nAvailable in sam.field:")
print(f"  - ScalarField2D: {hasattr(sam.field, 'ScalarField2D')}")
print(f"  - VectorField2D_2: {hasattr(sam.field, 'VectorField2D_2')}")
print(f"  - scalar(): {hasattr(sam.field, 'scalar')}")
print(f"  - vector(): {hasattr(sam.field, 'vector')}")
```

## Common Patterns

### TVD-RK3 Time Stepping

```python
# Create 3 fields for RK3 time stepping
u = sam.field.scalar(mesh, "u", init=1.0)
u1 = sam.field.scalar(mesh, "u1", init=1.0)
u2 = sam.field.scalar(mesh, "u2", init=1.0)

# Time stepping loop
for n in range(steps):
    # Stage 1
    du1 = compute_rhs(u)
    u1 = u - dt * du1

    # Stage 2
    du2 = compute_rhs(u1)
    u2 = (3.0/4.0) * u + (1.0/4.0) * (u1 - dt * du2)

    # Stage 3
    du3 = compute_rhs(u2)
    u = (1.0/3.0) * u + (2.0/3.0) * (u2 - dt * du3)
```

### Burgers Equation

```python
# 2D Burgers uses a 2-component vector field
u = sam.field.vector(mesh, "u", n_components=2, init=0.0)

# Apply initial condition
for cell in mesh:
    u[cell] = [initial_condition_x(cell.center),
               initial_condition_y(cell.center)]
```

## API Reference

### `sam.field.scalar(mesh, name, init=0.0)`

Create a scalar field. Dimension is inferred from the mesh type.

**Parameters:**
- `mesh` (MRMesh1D/2D/3D): The mesh to define the field on
- `name` (str): Field identifier
- `init` (float, optional): Initial value for all cells (default: 0.0)

**Returns:**
- `ScalarField1D` / `ScalarField2D` / `ScalarField3D`

### `sam.field.vector(mesh, name, n_components=2, init=0.0)`

Create a vector field. Dimension is inferred from the mesh type.

**Parameters:**
- `mesh` (MRMesh1D/2D/3D): The mesh to define the field on
- `name` (str): Field identifier
- `n_components` (int, optional): Number of components (2 or 3, default: 2)
- `init` (float, optional): Initial value for all cells (default: 0.0)

**Returns:**
- `VectorField1D_2` / `VectorField1D_3`
- `VectorField2D_2` / `VectorField2D_3`
- `VectorField3D_2` / `VectorField3D_3`

## Migration Guide

### From Old API to New API

| Old API | New API |
|---------|---------|
| `sam.ScalarField1D("u", mesh, 0.0)` | `sam.field.scalar(mesh, "u", init=0.0)` |
| `sam.ScalarField2D("u", mesh, 0.0)` | `sam.field.scalar(mesh, "u", init=0.0)` |
| `sam.ScalarField3D("u", mesh, 0.0)` | `sam.field.scalar(mesh, "u", init=0.0)` |
| `sam.make_scalar_field(mesh, "u", 0.0)` | `sam.field.scalar(mesh, "u", init=0.0)` |
| `sam.VectorField2D_2("v", mesh, 0.0)` | `sam.field.vector(mesh, "v", n_components=2, init=0.0)` |
| `sam.make_vector_field(mesh, "v", 2, 0.0)` | `sam.field.vector(mesh, "v", n_components=2, init=0.0)` |

## Benefits

1. **Dimension inference**: No need to specify dimension in function name
2. **Consistent API**: Same pattern for all dimensions
3. **Keyword arguments**: Self-documenting code with `init=` and `n_components=`
4. **Namespace organization**: All field-related functionality in `sam.field`
5. **100% backward compatible**: Old APIs still work

## Implementation Details

- The `sam.field` namespace also provides access to field classes:
  - `sam.field.ScalarField1D`
  - `sam.field.ScalarField2D`
  - `sam.field.ScalarField3D`
  - `sam.field.VectorField1D_2`, `sam.field.VectorField1D_3`
  - `sam.field.VectorField2D_2`, `sam.field.VectorField2D_3`
  - `sam.field.VectorField3D_2`, `sam.field.VectorField3D_3`

## Notes

- The new API was introduced in version 0.28.0
- No breaking changes - old code continues to work
- The old APIs are not deprecated and will continue to be supported
