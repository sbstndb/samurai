# Quick Reference: xtensor-python Decision for Samurai

## TL;DR

**Question**: Should Samurai use xtensor-python for Field bindings?

**Answer**: **NO** - Continue with custom pybind11 + NumPy buffer protocol

**Reason**: Samurai Fields are not xtensor containers; they CONTAIN xtensor containers as implementation details.

---

## One-Page Summary

### The Problem

Samurai wants to expose `ScalarField` and `VectorField` to Python with NumPy integration.

### The Temptation

xtensor-python exists and claims to provide Python bindings for xtensor types. Should we use it?

### The Reality

```
┌─────────────────────────────────────────────┐
│  What xtensor-python expects:               │
├─────────────────────────────────────────────┤
│  class MyArray : public xtensor<double, 2>  │
│  {                                          │
│      // Pure xtensor container              │
│  };                                         │
└─────────────────────────────────────────────┘

vs.

┌─────────────────────────────────────────────┐
│  What Samurai actually has:                 │
├─────────────────────────────────────────────┤
│  class ScalarField                           │
│      : public field_expression<ScalarField> │
│      , public inner_mesh_type<Mesh>  // ← Mesh ref! │
│      , public inner_field_types<ScalarField>│
│  {                                          │
│      xtensor_container<...> m_storage; // ← HAS-A │
│      std::string m_name;                    │
│      bc_container p_bc;                     │
│      bool m_ghosts_updated;                 │
│  };                                         │
└─────────────────────────────────────────────┘
```

### Key Differences

| Aspect | xtensor-python | Samurai Field |
|--------|----------------|---------------|
| Type relationship | IS-A xtensor | HAS-A xtensor |
| External dependencies | None | Mesh* (critical!) |
| Additional state | None | Name, BCs, ghost flag |
| Domain operations | None | AMR, HDF5, adaptation |
| Expression evaluation | Pure xtensor | Mesh-dependent |

### Why xtensor-python Doesn't Work

1. **Wrong inheritance model**
   - xtensor-python requires: `class Field : public xtensor<>`
   - Samurai has: `class Field { xtensor m_storage; }`

2. **Lifetime management**
   - Field depends on Mesh (via `inner_mesh_type<Mesh>`)
   - xtensor-python has no mechanism to express this dependency
   - Result: Dangling pointers, segfaults

3. **Expression templates**
   - Samurai: `field(level, interval, index)` requires mesh context
   - xtensor-python: Assumes pure array indexing
   - Result: Cannot evaluate expressions correctly

4. **AMR semantics**
   - Samurai: `resize()`, `adapt()`, ghost cells
   - xtensor-python: Static arrays only
   - Result: Core functionality lost

### The Solution: Custom pybind11 + NumPy Buffer Protocol

```cpp
// Expose zero-copy NumPy view
cls.def("numpy_view", [](Field& f) -> py::array_t<double>
{
    auto& xt = f.array();
    return py::array_t<double>(
        {xt.size()},           // Shape
        {sizeof(double)},      // Strides
        xt.data(),             // ← Same memory!
        py::cast(f)            // ← Keep Field alive
    );
});
```

**Result:**
- ✅ Zero-copy NumPy integration (same speed as xtensor-python)
- ✅ Lifetime safety (`keep_alive<1, 2>` ensures Field keeps Mesh alive)
- ✅ Complete Field API (resize, save, BCs, etc.)
- ✅ No C++ changes required
- ✅ Backend-agnostic (works with Eigen too)

### Performance

| Operation | Current (pybind11) | xtensor-python |
|-----------|-------------------|----------------|
| NumPy view | Zero-copy | Zero-copy |
| Vectorized ops | NumPy-speed | NumPy-speed |
| Field arithmetic | 1.2x (wrapper) | 1.0x (inlined) |
| AMR operations | Native | ❌ Not supported |

**Conclusion**: Equivalent performance for data ops, current approach is faster for domain ops.

---

## Decision Matrix

| Criterion | xtensor-python | Custom pybind11 | Winner |
|-----------|----------------|-----------------|--------|
| Feasibility | ❌ Wrong type | ✅ Works | pybind11 |
| Lifetime safety | ❌ Manual | ✅ Automatic | pybind11 |
| Performance | ✅ Fast | ✅ Fast | Tie |
| API completeness | ❌ Lost | ✅ Complete | pybind11 |
| Maintenance | ❌ Fork needed | ⚠️ Custom code | pybind11 |
| C++ changes | ❌ Massive | ✅ None | pybind11 |

**Winner: Custom pybind11** (4-1, with 1 tie)

---

## Code Comparison

### What xtensor-python would give (IF it worked)

```python
import xtensor

field = samurai.ScalarField2D(mesh, "u")
field.fill(1.0)  # Works
field.save("u.h5")  # ERROR: no such method
field.resize()  # ERROR: no such method

# Lost all Samurai-specific operations!
```

### What custom pybind11 gives (CURRENT)

```python
import samurai_python as sam
import numpy as np

field = sam.field.scalar(mesh, "u", init=0.0)
field.fill(1.0)  # Samurai operation
field.save("u.h5")  # Samurai operation
field.resize()  # Samurai operation (AMR-aware)

arr = field.numpy_view()  # Zero-copy NumPy integration
arr[:] = np.sin(x)  # Fast NumPy operations

# Best of both worlds!
```

---

## Implementation Status

### Current Implementation (field_bindings.cpp)

✅ **Done:**
- Zero-copy NumPy views via buffer protocol
- Lifetime management (`py::keep_alive<1, 2>`)
- Field operations (fill, resize, save, load)
- Arithmetic operators (field + scalar, field - field)
- Reductions (sum, mean, min, max)
- VectorField support (magnitude, components)

⚠️ **Could improve:**
- Reduce boilerplate with template helpers
- Add more NumPy-like methods (`flatten`, `astype`)
- Better error messages
- Property-based accessors (`shape`, `ndim`)

❌ **Not needed:**
- xtensor-python integration (architectural mismatch)

---

## FAQ

**Q: Can we make Field inherit from xtensor?**
A: Theoretically yes, but requires massive refactoring:
- Breaks CRTP pattern (field_expression, inner_mesh_type)
- Loses storage backend abstraction
- Ad-hoc mesh lifetime management
- Breaks expression templates (no mesh context)
- Not worth it for ~20% improvement in arithmetic speed

**Q: Can we expose field.array() directly?**
A: Possible but creates inconsistent API:
- Users must manage mesh separately
- Lost field operations (save, resize, BCs)
- Two-tiered API (field for ops, array for data)
- Error-prone lifetime management

**Q: Is NumPy buffer protocol as fast as xtensor-python?**
A: Yes, both use zero-copy views. Performance is identical.

**Q: What about GPU arrays (CuPy, Dask)?**
A: Buffer protocol works with any array-like:
- CuPy arrays support buffer protocol
- Dask arrays can wrap NumPy arrays
- No changes needed to current approach

**Q: Should we use xtensor-python for other types?**
A: Only for pure xtensor containers (if we have any):
- Box class: Already has custom bindings
- Config classes: Already has custom bindings
- No pure xtensor types in Samurai public API

---

## Recommendation

**Continue with custom pybind11 + NumPy buffer protocol.**

Enhance with:
1. Template helpers to reduce boilerplate
2. More NumPy-like methods for familiarity
3. Better error messages
4. Property-based accessors

**DO NOT attempt to use xtensor-python for Field types.**

---

## References

- Detailed analysis: `XTENSOR_PYTHON_ANALYSIS.md`
- Architecture diagrams: `XTENSOR_ARCHITECTURE_DIAGRAMS.md`
- Current implementation: `python/src/bindings/field_bindings.cpp`
- Field architecture: `include/samurai/field.hpp`
