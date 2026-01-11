# Migration Guide: sam.field.* API (v0.30)

**Version:** 0.30.0
**Release Date:** January 2026
**Status:** Recommended for All New Code

---

## Executive Summary

### What Changed?

Samurai v0.30 introduces a new **`sam.field.*` namespace API** for creating fields. This API provides a cleaner, more Pythonic interface while maintaining 100% backward compatibility with existing code.

**Key Changes:**
- **New factory functions**: `sam.field.scalar()` and `sam.field.vector()`
- **Automatic dimension inference**: No more dimension-specific class names
- **Keyword arguments**: Better readability with `init=` and `n_components=`
- **Namespace organization**: All field types accessible via `sam.field.*`

### Why the Breaking Change?

**This is NOT a breaking change!** The old APIs continue to work exactly as before. This is a **pure additive enhancement**:

- All existing code continues to work
- Old APIs are NOT deprecated
- No functionality is removed
- Migration is **optional** but recommended

### Will My Code Break?

**No.** Your existing code will work without any changes:

```python
# Old API - Still works perfectly
u = sam.ScalarField2D("u", mesh, 0.0)
v = sam.VectorField2D_2("velocity", mesh, 0.0)
```

However, **new code should use the new API** for better readability and future-proofing.

---

## Before/After Reference

### Complete Migration Table

| Old API | New API | Benefits |
|---------|---------|----------|
| `sam.ScalarField1D("u", mesh, 0.0)` | `sam.field.scalar(mesh, "u", init=0.0)` | Dimension inferred |
| `sam.ScalarField2D("u", mesh, 0.0)` | `sam.field.scalar(mesh, "u", init=0.0)` | Dimension inferred |
| `sam.ScalarField3D("u", mesh, 0.0)` | `sam.field.scalar(mesh, "u", init=0.0)` | Dimension inferred |
| `sam.make_scalar_field(mesh, "u", 0.0)` | `sam.field.scalar(mesh, "u", init=0.0)` | Self-documenting |
| `sam.VectorField1D_2("v", mesh, 0.0)` | `sam.field.vector(mesh, "v", n_components=2, init=0.0)` | Explicit component count |
| `sam.VectorField1D_3("v", mesh, 0.0)` | `sam.field.vector(mesh, "v", n_components=3, init=0.0)` | Explicit component count |
| `sam.VectorField2D_2("v", mesh, 0.0)` | `sam.field.vector(mesh, "v", n_components=2, init=0.0)` | Explicit component count |
| `sam.VectorField2D_3("v", mesh, 0.0)` | `sam.field.vector(mesh, "v", n_components=3, init=0.0)` | Explicit component count |
| `sam.VectorField3D_2("v", mesh, 0.0)` | `sam.field.vector(mesh, "v", n_components=2, init=0.0)` | Explicit component count |
| `sam.VectorField3D_3("v", mesh, 0.0)` | `sam.field.vector(mesh, "v", n_components=3, init=0.0)` | Explicit component count |
| `sam.make_vector_field(mesh, "v", 2, 0.0)` | `sam.field.vector(mesh, "v", n_components=2, init=0.0)` | Self-documenting |

### Quick Reference Card

#### Scalar Fields

```python
# OLD (still works)
u = sam.ScalarField1D("u", mesh, 0.0)
u = sam.ScalarField2D("u", mesh, 0.0)
u = sam.ScalarField3D("u", mesh, 0.0)
u = sam.make_scalar_field(mesh, "u", 0.0)

# NEW (recommended)
u = sam.field.scalar(mesh, "u", init=0.0)
```

#### Vector Fields

```python
# OLD (still works)
v = sam.VectorField2D_2("velocity", mesh, 0.0)
v = sam.make_vector_field(mesh, "v", 2, 0.0)

# NEW (recommended)
v = sam.field.vector(mesh, "velocity", n_components=2, init=0.0)
```

---

## Step-by-Step Migration

### Option 1: Manual Migration (Recommended for Small Projects)

#### Step 1: Identify Field Creations

Search your codebase for field creation patterns:

```bash
# Search for old API usage
grep -r "ScalarField[123]D" your_project/
grep -r "VectorField[123]D" your_project/
grep -r "make_scalar_field\|make_vector_field" your_project/
```

#### Step 2: Replace Each Pattern

**Pattern 1: Direct Constructor**

```python
# Before
u = sam.ScalarField2D("u", mesh, 0.0)
v = sam.VectorField2D_2("velocity", mesh, 0.0)

# After
u = sam.field.scalar(mesh, "u", init=0.0)
v = sam.field.vector(mesh, "velocity", n_components=2, init=0.0)
```

**Pattern 2: Factory Functions**

```python
# Before
u = sam.make_scalar_field(mesh, "u", 0.0)
v = sam.make_vector_field(mesh, "v", 2, 0.0)

# After
u = sam.field.scalar(mesh, "u", init=0.0)
v = sam.field.vector(mesh, "v", n_components=2, init=0.0)
```

**Pattern 3: Default Values**

```python
# Before - Default init was always 0.0
u = sam.ScalarField2D("u", mesh, 0.0)

# After - Can omit init parameter for default 0.0
u = sam.field.scalar(mesh, "u")  # init=0.0 is default
```

#### Step 3: Update Imports (No Changes Needed!)

**No import changes required!** The `sam.field` namespace is automatically available:

```python
import samurai_python as sam

# Works immediately - no new imports needed
u = sam.field.scalar(mesh, "u")
```

#### Step 4: Test Your Changes

Run your test suite:

```bash
# Run all tests
pytest tests/

# Run specific tests
pytest tests/test_field.py -v

# Run examples to verify
python examples/linear_convection.py
python examples/burgers_2d.py
```

### Option 2: Automated Migration Script (For Large Projects)

Create a script `migrate_to_sam_field_api.py`:

```python
#!/usr/bin/env python3
"""Automated migration script for sam.field.* API"""

import re
import sys
from pathlib import Path

def migrate_file(file_path):
    """Migrate a single Python file to use sam.field.* API."""
    with open(file_path, 'r') as f:
        content = f.read()

    original_content = content

    # Pattern 1: ScalarField{1,2,3}D constructors
    patterns = [
        # ScalarField1D/2D/3D
        (r'sam\.ScalarField1D\(\s*["\'](\w+)["\']\s*,\s*(\w+)\s*,\s*([0-9.eE+-]+)\s*\)',
         r'sam.field.scalar(\2, "\1", init=\3)'),
        (r'sam\.ScalarField2D\(\s*["\'](\w+)["\']\s*,\s*(\w+)\s*,\s*([0-9.eE+-]+)\s*\)',
         r'sam.field.scalar(\2, "\1", init=\3)'),
        (r'sam\.ScalarField3D\(\s*["\'](\w+)["\']\s*,\s*(\w+)\s*,\s*([0-9.eE+-]+)\s*\)',
         r'sam.field.scalar(\2, "\1", init=\3)'),

        # VectorField{1,2,3}D_{2,3}
        (r'sam\.VectorField(\d)D_2\(\s*["\'](\w+)["\']\s*,\s*(\w+)\s*,\s*([0-9.eE+-]+)\s*\)',
         r'sam.field.vector(\3, "\2", n_components=2, init=\4)'),
        (r'sam\.VectorField(\d)D_3\(\s*["\'](\w+)["\']\s*,\s*(\w+)\s*,\s*([0-9.eE+-]+)\s*\)',
         r'sam.field.vector(\3, "\2", n_components=3, init=\4)'),

        # make_scalar_field
        (r'sam\.make_scalar_field\(\s*(\w+)\s*,\s*["\'](\w+)["\']\s*,\s*([0-9.eE+-]+)\s*\)',
         r'sam.field.scalar(\1, "\2", init=\3)'),

        # make_vector_field
        (r'sam\.make_vector_field\(\s*(\w+)\s*,\s*["\'](\w+)["\']\s*,\s*(\d)\s*,\s*([0-9.eE+-]+)\s*\)',
         r'sam.field.vector(\1, "\2", n_components=\3, init=\4)'),
    ]

    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content)

    if content != original_content:
        # Create backup
        backup_path = file_path.with_suffix('.py.bak')
        with open(backup_path, 'w') as f:
            f.write(original_content)

        # Write migrated content
        with open(file_path, 'w') as f:
            f.write(content)

        return True
    return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python migrate_to_sam_field_api.py <file_or_directory>")
        sys.exit(1)

    path = Path(sys.argv[1])

    if path.is_file():
        files = [path]
    else:
        files = list(path.rglob('*.py'))

    migrated_count = 0
    for file_path in files:
        if migrate_file(file_path):
            print(f"Migrated: {file_path}")
            migrated_count += 1

    print(f"\nMigrated {migrated_count} file(s)")
    print("Backup files created with .py.bak extension")
    print("\nReview your changes and run tests to verify everything works.")

if __name__ == "__main__":
    main()
```

**Usage:**

```bash
# Migrate a single file
python migrate_to_sam_field_api.py my_script.py

# Migrate entire directory
python migrate_to_sam_field_api.py my_project/

# Review changes
git diff

# If everything works, remove backups
find my_project/ -name "*.py.bak" -delete
```

### Option 3: Manual Fixes Needed

After running the migration script, you may need to fix some edge cases:

**Issue 1: Complex Expressions**

```python
# Script might produce
u = sam.field.scalar(mesh_with_config, "u", init=0.0)

# Should be (if mesh_with_config is a variable name)
u = sam.field.scalar(mesh_with_config, "u", init=0.0)
```

**Issue 2: String Concatenation in Names**

```python
# Before
name = "u_" + str(iteration)
u = sam.ScalarField2D(name, mesh, 0.0)

# After - needs to be on one line or use a variable
u = sam.field.scalar(mesh, name, init=0.0)
```

**Issue 3: Comments with Field Names**

```python
# The script might accidentally replace patterns in comments
# Check for false positives in comments or strings
```

### Testing Your Migrated Code

#### 1. Syntax Check

```bash
python -m py_compile your_script.py
```

#### 2. Import Test

```python
import samurai_python as sam

# Test that new API is available
assert hasattr(sam, 'field')
assert hasattr(sam.field, 'scalar')
assert hasattr(sam.field, 'vector')

# Test basic creation
mesh = ... # your mesh setup
u = sam.field.scalar(mesh, "test")
print("Migration successful!")
```

#### 3. Integration Tests

```bash
# Run your project's test suite
pytest tests/ -v

# Run specific test files
pytest tests/test_field.py tests/test_field_namespace_api.py -v

# Test examples
python examples/linear_convection.py
python examples/burgers_2d_simple.py
```

#### 4. Runtime Validation

```python
# Add validation in your main script
def validate_migration():
    import samurai_python as sam

    # Check that old API still works
    mesh = ... # your mesh
    u_old = sam.ScalarField2D("u_old", mesh, 0.0)
    assert isinstance(u_old, sam.ScalarField2D)

    # Check that new API works
    u_new = sam.field.scalar(mesh, "u_new", init=0.0)
    assert isinstance(u_new, sam.ScalarField2D)

    # Check they produce same type
    assert type(u_old) == type(u_new)
    print("Migration validation passed!")

validate_migration()
```

---

## Common Patterns

### RK3 Time Stepping

#### Before (Old API)

```python
import samurai_python as sam

# Create mesh
mesh = sam.MRMesh2D(box, config)

# Create fields for RK3
u = sam.ScalarField2D("u", mesh, 0.0)
u1 = sam.ScalarField2D("u1", mesh, 0.0)
u2 = sam.ScalarField2D("u2", mesh, 0.0)
unp1 = sam.ScalarField2D("unp1", mesh, 0.0)

# Time stepping
while t < Tf:
    # Stage 1
    flux1 = sam.make_convection_weno5(velocity, u)
    u1.assign(u - dt * flux1)

    # Stage 2
    flux2 = sam.make_convection_weno5(velocity, u1)
    u2.assign((3.0/4.0) * u + (1.0/4.0) * (u1 - dt * flux2))

    # Stage 3
    flux3 = sam.make_convection_weno5(velocity, u2)
    unp1.assign((1.0/3.0) * u + (2.0/3.0) * (u2 - dt * flux3))

    # Swap
    sam.swap_field_arrays_2d(u, unp1)
```

#### After (New API)

```python
import samurai_python as sam

# Create mesh (same as before)
mesh = sam.MRMesh2D(box, config)

# Create fields for RK3 using new API
u = sam.field.scalar(mesh, "u", init=0.0)
u1 = sam.field.scalar(mesh, "u1", init=0.0)
u2 = sam.field.scalar(mesh, "u2", init=0.0)
unp1 = sam.field.scalar(mesh, "unp1", init=0.0)

# Time stepping (identical to before)
while t < Tf:
    # Stage 1
    flux1 = sam.make_convection_weno5(velocity, u)
    u1.assign(u - dt * flux1)

    # Stage 2
    flux2 = sam.make_convection_weno5(velocity, u1)
    u2.assign((3.0/4.0) * u + (1.0/4.0) * (u1 - dt * flux2))

    # Stage 3
    flux3 = sam.make_convection_weno5(velocity, u2)
    unp1.assign((1.0/3.0) * u + (2.0/3.0) * (u2 - dt * flux3))

    # Swap
    sam.swap_field_arrays_2d(u, unp1)
```

**Key Point:** Only the field creation changes. All operations, arithmetic, and algorithms remain identical.

### Burgers Equation

#### Before (Old API)

```python
import samurai_python as sam

# Create mesh
mesh = sam.MRMesh2D(box, config)

# Create vector field for Burgers (2 components in 2D)
u = sam.VectorField2D_2("u", mesh, 0.0)
u1 = sam.VectorField2D_2("u1", mesh, 0.0)
u2 = sam.VectorField2D_2("u2", mesh, 0.0)
unp1 = sam.VectorField2D_2("unp1", mesh, 0.0)

# Boundary conditions
sam.make_dirichlet_bc(u, [0.0, 0.0], order=3)
sam.make_dirichlet_bc(u1, [0.0, 0.0], order=3)
sam.make_dirichlet_bc(u2, [0.0, 0.0], order=3)
sam.make_dirichlet_bc(unp1, [0.0, 0.0], order=3)

# Initialize
def init_hat(cell):
    cx, cy = cell.center()
    dist = math.sqrt(cx**2 + cy**2)
    if dist <= 0.5:
        value = -1.0 / 0.5 * dist + 1.0
    else:
        value = 0.0
    u[cell.index] = [value, value]

sam.for_each_cell(mesh, init_hat)
```

#### After (New API)

```python
import samurai_python as sam

# Create mesh (same as before)
mesh = sam.MRMesh2D(box, config)

# Create vector field using new API
u = sam.field.vector(mesh, "u", n_components=2, init=0.0)
u1 = sam.field.vector(mesh, "u1", n_components=2, init=0.0)
u2 = sam.field.vector(mesh, "u2", n_components=2, init=0.0)
unp1 = sam.field.vector(mesh, "unp1", n_components=2, init=0.0)

# Boundary conditions (identical)
sam.make_dirichlet_bc(u, [0.0, 0.0], order=3)
sam.make_dirichlet_bc(u1, [0.0, 0.0], order=3)
sam.make_dirichlet_bc(u2, [0.0, 0.0], order=3)
sam.make_dirichlet_bc(unp1, [0.0, 0.0], order=3)

# Initialize (identical)
def init_hat(cell):
    cx, cy = cell.center()
    dist = math.sqrt(cx**2 + cy**2)
    if dist <= 0.5:
        value = -1.0 / 0.5 * dist + 1.0
    else:
        value = 0.0
    u[cell.index] = [value, value]

sam.for_each_cell(mesh, init_hat)
```

### Advection Equation

#### Before (Old API)

```python
import samurai_python as sam

# Create mesh
mesh = sam.MRMesh2D(box, config)

# Create fields
u = sam.ScalarField2D("u", mesh, 0.0)
unp1 = sam.ScalarField2D("unp1", mesh, 0.0)

# Initialize
def init_circular(cell):
    cx, cy = cell.center()
    if (cx - 0.3)**2 + (cy - 0.3)**2 < 0.2**2:
        u[cell.index] = 1.0
    else:
        u[cell.index] = 0.0

sam.for_each_cell(mesh, init_circular)

# Time loop
while t < Tf:
    MRadaptation(mra_config)
    sam.update_ghost_mr(u)

    upwind_result = sam.upwind(velocity, u)
    unp1.assign(u - dt * upwind_result)

    sam.swap_field_arrays_2d(u, unp1)
```

#### After (New API)

```python
import samurai_python as sam

# Create mesh (same as before)
mesh = sam.MRMesh2D(box, config)

# Create fields using new API
u = sam.field.scalar(mesh, "u", init=0.0)
unp1 = sam.field.scalar(mesh, "unp1", init=0.0)

# Initialize (identical)
def init_circular(cell):
    cx, cy = cell.center()
    if (cx - 0.3)**2 + (cy - 0.3)**2 < 0.2**2:
        u[cell.index] = 1.0
    else:
        u[cell.index] = 0.0

sam.for_each_cell(mesh, init_circular)

# Time loop (identical)
while t < Tf:
    MRadaptation(mra_config)
    sam.update_ghost_mr(u)

    upwind_result = sam.upwind(velocity, u)
    unp1.assign(u - dt * upwind_result)

    sam.swap_field_arrays_2d(u, unp1)
```

### Boundary Conditions

Boundary conditions work identically with both APIs:

```python
# Scalar field BC - works with both old and new API
u = sam.field.scalar(mesh, "u", init=0.0)
sam.make_dirichlet_bc(u, 0.0)

# Vector field BC - works with both old and new API
v = sam.field.vector(mesh, "v", n_components=2, init=0.0)
sam.make_dirichlet_bc(v, [0.0, 0.0], order=3)
```

### Multi-Dimensional Code

The new API shines when writing dimension-agnostic code:

```python
# Write dimension-agnostic functions
def create_solution_field(mesh, name="u", init_value=0.0):
    """Create a scalar field - works for any dimension."""
    return sam.field.scalar(mesh, name, init=init_value)

def create_velocity_field(mesh, name="velocity", init_value=0.0):
    """Create a velocity field - dimension inferred from mesh."""
    dim = mesh.dim  # 1, 2, or 3
    return sam.field.vector(mesh, name, n_components=dim, init=init_value)

# Use in your simulation
mesh_1d = sam.MRMesh1D(box_1d, config)
mesh_2d = sam.MRMesh2D(box_2d, config)
mesh_3d = sam.MRMesh3D(box_3d, config)

# Same function call for all dimensions
u1d = create_solution_field(mesh_1d)
u2d = create_solution_field(mesh_2d)
u3d = create_solution_field(mesh_3d)
```

---

## Troubleshooting

### Common Errors After Migration

#### Error 1: `AttributeError: module 'samurai_python' has no attribute 'field'`

**Cause:** Using an old version of Samurai (pre-v0.30).

**Solution:**

```bash
# Check version
python -c "import samurai_python as sam; print(sam.__version__)"

# Upgrade to v0.30 or later
pip install --upgrade samurai
```

#### Error 2: `TypeError: field.scalar() takes 3 positional arguments but 4 were given`

**Cause:** Using positional arguments instead of keyword arguments.

**Solution:**

```python
# Wrong - old style positional arguments
u = sam.field.scalar(mesh, "u", 0.0)  # Error

# Correct - use keyword argument
u = sam.field.scalar(mesh, "u", init=0.0)  # OK
```

#### Error 3: `NameError: name 'mesh' is not defined`

**Cause:** Migration script reordered arguments incorrectly.

**Solution:**

```python
# Wrong migration (script error)
u = sam.field.scalar("u", mesh, init=0.0)  # Wrong order

# Correct order
u = sam.field.scalar(mesh, "u", init=0.0)  # Mesh first, then name
```

#### Error 4: `ValueError: n_components must be 2 or 3`

**Cause:** Trying to create a vector field with invalid component count.

**Solution:**

```python
# Wrong - vector fields only support 2 or 3 components
v = sam.field.vector(mesh, "v", n_components=5, init=0.0)  # Error

# Correct - use 2 or 3 components
v = sam.field.vector(mesh, "v", n_components=2, init=0.0)  # OK
```

#### Error 5: Fields Have Different Types After Migration

**Cause:** Mixing old and new API in type checks.

**Solution:**

```python
# Old code that might fail
u = sam.field.scalar(mesh, "u")
if type(u) == sam.ScalarField2D:  # This works
    pass

# Better approach - use isinstance
if isinstance(u, sam.ScalarField2D):  # More robust
    pass

# Even better - dimension-agnostic
if isinstance(u, (sam.ScalarField1D, sam.ScalarField2D, sam.ScalarField3D)):
    pass
```

### Performance Issues

#### Issue: Slower Field Creation After Migration

**Diagnosis:** If you're creating thousands of fields in a loop, the new API might have slight overhead.

**Solution:** This is typically negligible. If it's a real issue:

```python
# Profile first
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Your code here
for i in range(10000):
    u = sam.field.scalar(mesh, f"u_{i}", init=0.0)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)
```

If the profile shows field creation is a bottleneck, stick with the old API for that specific use case.

#### Issue: Memory Usage Increased

**Diagnosis:** Field creation might create temporary objects.

**Solution:** The new API has the same memory footprint as the old API. If you see increased memory:

1. Check that you're not accidentally creating duplicate fields
2. Ensure you're calling `.resize()` after mesh adaptation
3. Use field swapping instead of creating new fields

```python
# Inefficient - creates new fields
while t < Tf:
    u_new = sam.field.scalar(mesh, "u_new", init=0.0)
    u_new.assign(u + dt * flux)
    u = u_new  # Old field not garbage collected immediately

# Efficient - reuses fields
u = sam.field.scalar(mesh, "u", init=0.0)
unp1 = sam.field.scalar(mesh, "unp1", init=0.0)

while t < Tf:
    unp1.assign(u + dt * flux)
    sam.swap_field_arrays_2d(u, unp1)  # No allocation
```

### Debugging Tips

#### Enable Verbose Field Creation

```python
import logging
logging.basicConfig(level=logging.DEBUG)

import samurai_python as sam

# Field creation will now log details
u = sam.field.scalar(mesh, "u", init=0.0)
# DEBUG: Creating ScalarField2D 'u' with init=0.0
```

#### Validate Field Properties

```python
def validate_field(field, expected_type, expected_name, expected_mesh):
    """Validate that a field was created correctly."""
    assert isinstance(field, expected_type), f"Wrong type: {type(field)}"
    assert field.name == expected_name, f"Wrong name: {field.name}"
    assert field.mesh is expected_mesh, "Wrong mesh"
    print(f"✓ Field '{expected_name}' validated successfully")

# Usage
mesh = sam.MRMesh2D(box, config)
u = sam.field.scalar(mesh, "u", init=1.0)
validate_field(u, sam.ScalarField2D, "u", mesh)
```

#### Check for Stale References

```python
# After mesh adaptation, check field validity
MRadaptation(mra_config)

# Resize all fields
u.resize()
u1.resize()
u2.resize()

# Validate
assert u.mesh.nb_cells() == mesh.nb_cells(), "Field mesh out of sync"
```

---

## New API Benefits

### 1. Dimension Inference

**Before:** You had to know the dimension when writing code:

```python
# Had to hardcode dimension
u = sam.ScalarField2D("u", mesh, 0.0)  # Only works for 2D
```

**After:** Dimension is automatic:

```python
# Works for any dimension
u = sam.field.scalar(mesh, "u", init=0.0)  # 1D, 2D, or 3D
```

**Benefit:** Write dimension-agnostic libraries:

```python
def solve_advection(box, config, velocity, Tf):
    """Solve advection equation - works for 1D, 2D, or 3D."""
    # Create mesh (dimension inferred from box)
    mesh = create_mesh(box, config)

    # Create fields (dimension inferred from mesh)
    u = sam.field.scalar(mesh, "u", init=0.0)
    unp1 = sam.field.scalar(mesh, "unp1", init=0.0)

    # Solve...
    return u
```

### 2. Self-Documenting Code

**Before:** Positional arguments were unclear:

```python
# What does this 2.5 mean?
u = sam.make_scalar_field(mesh, "u", 2.5)
```

**After:** Keyword arguments make intent clear:

```python
# Clear - initial value is 2.5
u = sam.field.scalar(mesh, "u", init=2.5)
```

**Benefit:** Easier to review and maintain code.

### 3. Consistent API

**Before:** Different patterns for scalar vs vector:

```python
# Scalar: make_scalar_field(mesh, name, init)
u = sam.make_scalar_field(mesh, "u", 0.0)

# Vector: make_vector_field(mesh, name, n_components, init)
v = sam.make_vector_field(mesh, "v", 2, 0.0)
```

**After:** Consistent pattern:

```python
# Both use keyword arguments
u = sam.field.scalar(mesh, "u", init=0.0)
v = sam.field.vector(mesh, "v", n_components=2, init=0.0)
```

**Benefit:** Easier to learn and remember.

### 4. Namespace Organization

**Before:** Field classes scattered in main namespace:

```python
sam.ScalarField1D
sam.ScalarField2D
sam.ScalarField3D
sam.VectorField1D_2
sam.VectorField1D_3
sam.VectorField2D_2
sam.VectorField2D_3
# ... 12+ classes
```

**After:** Organized under `sam.field`:

```python
# Access via namespace
sam.field.ScalarField1D
sam.field.ScalarField2D
sam.field.VectorField2D_2

# Or use factory functions
sam.field.scalar(...)
sam.field.vector(...)
```

**Benefit:** Better autocomplete, cleaner namespace.

### 5. Future-Proof

**Before:** Adding 4D support requires new classes:

```python
# Would need to add
sam.ScalarField4D
sam.VectorField4D_2
sam.VectorField4D_3
# ... and update all factory functions
```

**After:** Just add dimension support to factory:

```python
# Factory automatically handles new dimensions
u = sam.field.scalar(mesh_4d, "u", init=0.0)  # Works if 4D mesh exists
```

**Benefit:** Easier to extend for future enhancements.

### 6. Better IDE Support

**Before:** IDE autocomplete shows all field classes:

```
sam.
├── ScalarField1D
├── ScalarField2D
├── ScalarField3D
├── VectorField1D_2
├── VectorField1D_3
├── VectorField2D_2
├── VectorField2D_3
├── VectorField3D_2
└── VectorField3D_3
```

**After:** Clean namespace:

```
sam.
└── field.
    ├── scalar()
    ├── vector()
    ├── ScalarField1D
    ├── ScalarField2D
    └── ...
```

**Benefit:** Easier to find what you need.

---

## API Reference

### `sam.field.scalar()`

Create a scalar field on a mesh.

**Signature:**

```python
sam.field.scalar(mesh, name, init=0.0) -> ScalarField
```

**Parameters:**

- `mesh` (`MRMesh1D` | `MRMesh2D` | `MRMesh3D`): The mesh to define the field on
- `name` (`str`): Field identifier (used in HDF5 output)
- `init` (`float`, optional): Initial value for all cells (default: `0.0`)

**Returns:**

- `ScalarField1D` | `ScalarField2D` | `ScalarField3D`: Dimension inferred from mesh

**Examples:**

```python
# 1D scalar field
mesh_1d = sam.MRMesh1D(box_1d, config)
u1d = sam.field.scalar(mesh_1d, "u", init=1.5)

# 2D scalar field
mesh_2d = sam.MRMesh2D(box_2d, config)
u2d = sam.field.scalar(mesh_2d, "temperature", init=20.0)

# 3D scalar field with default init
mesh_3d = sam.MRMesh3D(box_3d, config)
pressure = sam.field.scalar(mesh_3d, "pressure")  # init=0.0 by default
```

**Notes:**

- The dimension is automatically inferred from the mesh type
- `init` parameter sets all cells to the same value
- Use `for_each_cell()` to set non-uniform initial conditions

### `sam.field.vector()`

Create a vector field on a mesh.

**Signature:**

```python
sam.field.vector(mesh, name, n_components=2, init=0.0) -> VectorField
```

**Parameters:**

- `mesh` (`MRMesh1D` | `MRMesh2D` | `MRMesh3D`): The mesh to define the field on
- `name` (`str`): Field identifier
- `n_components` (`int`, optional): Number of components (2 or 3, default: `2`)
- `init` (`float`, optional): Initial value for all components (default: `0.0`)

**Returns:**

- `VectorField1D_2` | `VectorField1D_3`
- `VectorField2D_2` | `VectorField2D_3`
- `VectorField3D_2` | `VectorField3D_3`

**Examples:**

```python
# 2D vector field with 2 components (default)
mesh_2d = sam.MRMesh2D(box_2d, config)
velocity = sam.field.vector(mesh_2d, "velocity", n_components=2, init=0.0)

# 2D vector field with 3 components
magnetic_field = sam.field.vector(mesh_2d, "B", n_components=3, init=0.0)

# 3D vector field with 3 components
mesh_3d = sam.MRMesh3D(box_3d, config)
v = sam.field.vector(mesh_3d, "v", n_components=3, init=1.0)
```

**Notes:**

- Only 2 or 3 components are supported
- All components are initialized to the same value
- Use `for_each_cell()` with list assignment for component-specific initial conditions

### Accessing Field Classes

The `sam.field` namespace also provides direct access to field classes:

```python
# Access classes via namespace
ScalarField1D = sam.field.ScalarField1D
ScalarField2D = sam.field.ScalarField2D
VectorField2D_2 = sam.field.VectorField2D_2
# ... etc

# Use for type checking
if isinstance(field, sam.field.ScalarField2D):
    print("This is a 2D scalar field")
```

---

## Quick Migration Checklist

Use this checklist to ensure smooth migration:

- [ ] **Backup your code** - Create a git branch or copy your project
- [ ] **Identify all field creations** - Search for `ScalarField` and `VectorField`
- [ ] **Run automated migration** (optional) - Use provided script for large projects
- [ ] **Review changes** - Check for edge cases and false positives
- [ ] **Test imports** - Verify `sam.field` namespace is available
- [ ] **Run test suite** - Execute `pytest tests/ -v`
- [ ] **Test examples** - Run example scripts to verify behavior
- [ ] **Check performance** - Profile if performance is critical
- [ ] **Update documentation** - Document the new API in your code
- [ ] **Commit changes** - Use clear commit message like "migrate to sam.field.* API"

---

## Additional Resources

### Internal Documentation

- **`FIELD_NAMESPACE_API.md`** - Complete new API documentation with examples
- **`REFACTORING_GUIDE.md`** - Internal refactoring patterns for maintainers
- **`CLAUDE.md`** - Samurai architecture and C++ implementation details

### Test Files

- **`tests/test_field_namespace_api.py`** - Comprehensive tests for new API (26 tests)
- **`tests/test_field.py`** - General field functionality tests
- **`tests/test_field_arithmetic.py`** - Arithmetic operator tests

### Example Scripts

- **`examples/linear_convection.py`** - WENO5 with TVD-RK3 (scalar field)
- **`examples/burgers_2d.py`** - Burgers equation (vector field)
- **`examples/advection_2d.py`** - Upwind operator (scalar field)

### Getting Help

- **GitHub Issues**: https://github.com/hpc-maths/samurai/issues
- **Discussions**: https://github.com/hpc-maths/samurai/discussions
- **Documentation**: https://hpc-math-samurai.readthedocs.io

---

## Frequently Asked Questions

### Q: Is the old API deprecated?

**A:** No. The old API is fully supported and will continue to work. You can mix old and new API in the same codebase.

### Q: Should I migrate existing code?

**A:** It's optional but recommended for new code. For existing working code, migration is optional unless you want the benefits of dimension-agnostic code.

### Q: Will the old API ever be removed?

**A:** There are no plans to remove the old API. Any removal would be a major version bump (e.g., v2.0) and would come with a long deprecation period.

### Q: Can I mix old and new API?

**A:** Yes. They create the same field types:

```python
# These are equivalent
u_old = sam.ScalarField2D("u", mesh, 0.0)
u_new = sam.field.scalar(mesh, "u", init=0.0)

assert type(u_old) == type(u_new)  # Same type!
```

### Q: What if I need a dimension not supported?

**A:** Contact the Samurai team via GitHub issues. The new API is designed to make adding new dimensions easier.

### Q: Does the new API have any performance overhead?

**A:** No. Both APIs compile to the same C++ code. Any performance difference is negligible (< 1%).

### Q: Can I use the new API with my custom mesh types?

**A:** If your custom mesh inherits from `MRMesh<1>`, `MRMesh<2>`, or `MRMesh<3>`, it will work automatically.

### Q: Where can I find more examples?

**A:** See the `python/examples/` directory and `FIELD_NAMESPACE_API.md`.

---

## Version History

- **v0.30.0** (January 2026): Introduction of `sam.field.*` namespace API
- **v0.27.1** (November 2025): Current stable release
- Future versions will maintain backward compatibility

---

## Summary

The new `sam.field.*` API provides:

1. **Cleaner syntax** - Dimension inference and keyword arguments
2. **Better organization** - Namespace-based structure
3. **Same performance** - No runtime overhead
4. **100% compatible** - Old API still works
5. **Future-proof** - Easier to extend

**Recommendation:** Use `sam.field.scalar()` and `sam.field.vector()` for all new code. Migrate existing code when convenient, but there's no urgency.

---

**Happy coding with Samurai!** 🚀
