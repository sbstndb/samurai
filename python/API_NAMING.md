# API Naming Conventions

## Preferred API (Submodule-based)

The Samurai Python API is organized into logical submodules. Use these submodule functions for clarity and consistency.

### Boundary Conditions
```python
# PREFERRED
sam.boundary.dirichlet(field, value)
sam.boundary.neumann(field, value)

# DEPRECATED (still works but avoid)
sam.make_dirichlet_bc(field, value)
sam.make_neumann_bc(field, value)
```

### Operators
```python
# PREFERRED
sam.operators.upwind(field, velocity)
sam.operators.convection_weno5(field, velocity)
sam.operators.apply_upwind(input, output, velocity)

# Note: dimension-specific functions (apply_upwind_1d, etc.)
# are being phased out in favor of dimension-agnostic versions
```

### Adaptation
```python
# PREFERRED - use the class
MRadapt = sam.adaptation.MRAdapt(field)
MRadapt(config)

# DEPRECATED
sam.make_MRAdapt(field)
```

## Migration Guide

### Old API → New API

| Old | New |
|-----|-----|
| `sam.make_dirichlet_bc(u, 0.0)` | `sam.boundary.dirichlet(u, 0.0)` |
| `sam.make_neumann_bc(u, 0.0)` | `sam.boundary.neumann(u, 0.0)` |
| `sam.make_convection_weno5(u, vel)` | `sam.operators.convection_weno5(u, vel)` |
| `sam.make_MRAdapt(u)` | `sam.adaptation.MRAdapt(u)` |

## Rationale

- Submodule organization provides better namespace clarity
- Consistent with Python best practices
- Autocomplete works better with submodules
- Follows pattern: `sam.{category}.{action}()`
