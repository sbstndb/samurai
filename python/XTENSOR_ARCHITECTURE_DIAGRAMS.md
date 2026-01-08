# xtensor-python vs Custom pybind11: Visual Architecture Analysis

## Architecture Comparison

### Current Samurai Field Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      samurai::ScalarField                        │
├─────────────────────────────────────────────────────────────────┤
│  Inherits:                                                       │
│  ├── field_expression<ScalarField>  ──► Expression templates    │
│  ├── inner_mesh_type<Mesh>         ──► Mesh* p_mesh            │
│  └── inner_field_types<ScalarField> ──► Storage wrapper         │
│                                                                   │
│  Members:                                                         │
│  ├── std::string m_name                                           │
│  ├── data_type m_storage ────────► xtensor_container<double,1>   │
│  │                                  └── xt::xtensor<double,1>   │
│  ├── bc_container p_bc          ──► Boundary conditions         │
│  └── bool m_ghosts_updated                                         │
│                                                                   │
│  Methods:                                                         │
│  ├── fill(value)                  ──► Fill all cells            │
│  ├── resize()                     ──► Update after AMR          │
│  ├── attach_bc(bc)                ──► Set boundary conditions   │
│  ├── save(filepath)               ──► HDF5 I/O                  │
│  └── operator[](cell)             ──► Access by cell            │
└─────────────────────────────────────────────────────────────────┘
           │                                    │
           │ keeps alive                        │ contains
           ▼                                    ▼
┌─────────────────────┐            ┌──────────────────────────┐
│     samurai::Mesh   │            │  xt::xtensor<double, 1>   │
│  ┌───────────────┐  │            │  ┌──────────────────────┐ │
│  │ CellArray*    │  │            │  │ double* m_data        │ │
│  │ mesh_id::cells│  │            │  │ std::size_t m_size    │ │
│  │ mesh_id::ghosts│  │            │  │ layout_type m_layout  │ │
│  └───────────────┘  │            │  └──────────────────────┘ │
└─────────────────────┘            └──────────────────────────┘
```

### xtensor-python Assumptions (WHAT IT EXPECTS)

```
┌─────────────────────────────────────────────────────────────────┐
│                 Type inheriting from xcontainer<>               │
├─────────────────────────────────────────────────────────────────┤
│  Inherits:                                                       │
│  ├── xcontainer<Derived>                                        │
│  ├── xsemantic_base<Derived>                                    │
│  └── xiterable<Derived>                                         │
│                                                                   │
│  Members:                                                         │
│  └── storage_type m_storage  ──► PURE xtensor (no mesh ref)     │
│                                                                   │
│  Methods:                                                         │
│  ├── operator[](i)              ──► Index access                │
│  ├── shape()                    ──► Array shape                │
│  ├── size()                     ──► Total elements            │
│  └── reshape(new_shape)         ──► Change shape               │
└─────────────────────────────────────────────────────────────────┘
           │
           │ NO EXTERNAL DEPENDENCIES
           ▼
      (standalone)
```

---

## Why xtensor-python Doesn't Work

### Mismatch #1: Inheritance vs Composition

```
XTENSOR-PYTHON EXPECTS:                    SAMURAI REALITY:

┌──────────────────────┐                  ┌──────────────────────┐
│ Field extends xtensor│                  │  Field HAS-A xtensor │
├──────────────────────┤                  ├──────────────────────┤
│ class Field          │                  │ class Field          │
│   : public xtensor   │                  │ {                    │
│ {                    │                  │   xtensor m_storage; │
│   // pure xtensor    │                  │   Mesh* p_mesh;      │
│ }                    │                  │ }                    │
└──────────────────────┘                  └──────────────────────┘
       │                                            │
       │ Field IS-A xtensor                         │ Field HAS-A xtensor
       │                                           │ (and HAS-A Mesh)
       ▼                                           ▼
  Compatible with xtensor-python           Incompatible with xtensor-python
```

### Mismatch #2: Lifetime Dependencies

```
XTENSOR-PYTHON MODEL:                     SAMURAI MODEL:

┌──────────┐                               ┌──────┐
│ Field    │                               │ Field│
│ (self    │                               │      │
│  contained)                              │      │
└──────────┘                               │      │
       │                                  │      │ keeps alive
       ▼                                  │      ▼
  No external                      ┌──────┴─────────────┐
  dependencies                    │       Mesh          │
                                   │  ┌──────────────┐ │
                                   │  │ CellArrays   │ │
                                   │  └──────────────┘ │
                                   └────────────────────┘
                                          │ keeps alive
                                          ▼
                                   ┌──────────────────┐
                                   │ Domain/Config    │
                                   └──────────────────┘

xtensor-python:                        Samurai:
- Doesn't know about Mesh              - Requires ownership chain
- Can't express dependency             - pybind11: keep_alive<1,2>
- Breaks if Mesh deleted               - Safe with lifetime mgmt
```

### Mismatch #3: Expression Templates

```
XTENSOR-PYTHON:                        SAMURAI:

Field + Field ──► xtensor             Field + Field ──► field_expression<>
│                                      │
│                                      │ for_each_interval(mesh,
│                                      │   [&](level, interval, index)
│                                      │   {
│                                      │     result(level,i,index) =
│                                      │       lhs(level,i,index) +
│                                      │       rhs(level,i,index);
│                                      │   });
│                                      ▼
                                   Requires MESH context for
                                   interval-based evaluation
```

---

## Current Solution: Custom pybind11 + NumPy Buffer Protocol

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Python Layer                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  import samurai_python as sam                                    │
│  import numpy as np                                              │
│                                                                   │
│  mesh = sam.mesh.make(box, min_level=4, max_level=8)            │
│  field = sam.field.scalar(mesh, "u", init=0.0)                  │
│                                                                   │
│  # Field operations (Samurai API)                                │
│  field.fill(1.0)                                                 │
│  field.resize()                                                  │
│  field.save("solution.h5")                                       │
│                                                                   │
│  # NumPy integration (zero-copy)                                 │
│  arr = field.numpy_view()  ────┐                                 │
│  arr[:] = np.sin(x)           │  Buffer Protocol                 │
│                               │  (same memory, zero-copy)        │
└───────────────────────────────┼─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                   pybind11 Binding Layer                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  py::class_<ScalarField<dim>>(m, "ScalarField")                  │
│      .def(py::init<Mesh&, string, double>(),                     │
│           py::keep_alive<1, 2>())  // Field keeps Mesh alive     │
│      .def("fill", &Field::fill)                                  │
│      .def("resize", &Field::resize)                              │
│      .def("numpy_view", [](Field& f) {                           │
│          return py::array_t<double>(                             │
│              {f.array().size()},                                 │
│              {sizeof(double)},                                   │
│              f.array().data(),    // ← Same memory!              │
│              py::cast(f)         // ← Keep Field alive           │
│          );                                                      │
│      })                                                          │
│      .def("__add__", &field_add_scalar<dim>)                     │
│      .def("__sub__", &field_sub_scalar<dim>)                     │
│      // ...                                                      │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
           │
           │ binds to
           ▼
┌─────────────────────────────────────────────────────────────────┐
│                     C++ Samurai Layer                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  template <class mesh_t, class value_t>                          │
│  class ScalarField : public field_expression<ScalarField<...>>, │
│                      public inner_mesh_type<mesh_t>,             │
│                      public inner_field_types<ScalarField<...>>  │
│  {                                                                │
│      mesh_t* p_mesh;              // ← Mesh reference            │
│      xtensor_container<...> m_storage;  // ← xtensor wrapper     │
│      std::string m_name;                                        │
│      bc_container p_bc;                                          │
│      bool m_ghosts_updated;                                      │
│  };                                                              │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
           │                                    │
           │ contains                           │ wraps
           ▼                                    ▼
┌─────────────────────┐            ┌──────────────────────────┐
│  samurai::Mesh      │            │  xt::xtensor<double, 1>   │
│  (interval-based    │            │  (actual storage)          │
│   AMR data structure)│            └──────────────────────────┘
└─────────────────────┘                       ▲
                                              │
                                          Buffer Protocol
                                              │
                       ┌──────────────────────┴──────────────────┐
                       │         NumPy Array (Python)            │
                       │  ┌───────────────────────────────────┐ │
                       │  │ py::array_t<double>               │ │
                       │  │ - Same memory as xtensor          │ │
                       │  │ - Zero-copy view                  │ │
                       │  │ - Keeps Field alive               │ │
                       │  └───────────────────────────────────┘ │
                       └─────────────────────────────────────────┘
```

### Data Flow

```
Python Operation          →  C++ Call                  →  Result
─────────────────────────────────────────────────────────────────

field.fill(1.0)          →  f.fill(1.0)              →  m_data.fill(1.0)
                                                     →  (xtensor operation)

arr = field.numpy_view() →  py::array_t<double>(     →  NumPy array sharing
                           f.array().data(),            same memory
                           py::cast(f))
                                                     →  Zero-copy!

arr[:] = np.sin(x)       →  (Direct NumPy operation) →  Writes to
                                                           field's data
                                                           (no copy)

field.resize()           →  f.resize()               →  m_storage.resize(
                                                           mesh.nb_cells())
                                                     →  Updates to new mesh

field.save("file.h5")    →  save(path, file,         →  HDF5 output
                           mesh, field)

field1 + field2          →  field_add_field(f1, f2)  →  Creates new field
                                                     →  Expression template
                                                     →  Lazy evaluation
```

---

## Design Space Comparison

### Option Matrix

| Aspect | Option A: Inheritance | Option B: Direct Exposure | Option C: Adapter (✓) | Option D: Custom xtensor-python |
|--------|---------------------|-------------------------|---------------------|------------------------------|
| **Architecture** | | | | |
| C++ changes required | Massive | Medium | None | Extreme |
| Preserves CRTP | ❌ No | ⚠️ Partial | ✓ Yes | ❌ No |
| Backend flexibility | ❌ No | ❌ No | ✓ Yes | ❌ No |
| **Python API** | | | | |
| Field operations | ⚠️ Ad-hoc | ❌ Lost | ✓ Complete | ⚠️ Partial |
| NumPy integration | ✓ Native | ✓ Native | ✓ Zero-copy | ✓ Native |
| API consistency | ❌ Mixed | ❌ Two-tier | ✓ Unified | ⚠️ Mixed |
| **Safety** | | | | |
| Lifetime management | ❌ Manual | ❌ Broken | ✓ Automatic | ❌ Manual |
| Mesh reference | ⚠️ Raw pointer | ⚠️ Lost | ✓ Managed | ❌ Broken |
| AMR safety | ❌ Unsafe | ❌ Unsafe | ✓ Safe | ❌ Unsafe |
| **Performance** | | | | |
| Data access | ✓ Direct | ✓ Direct | ✓ Zero-copy | ✓ Direct |
| Expression eval | ✓ Lazy | ✓ Lazy | ✓ Lazy | ⚠️ Complex |
| Arithmetic overhead | ✓ Inline | ✓ Inline | ⚠️ Function call | ✓ Inline |
| **Maintenance** | | | | |
| Implementation effort | ❌ Very High | ⚠️ Medium | ⚠️ Medium | ❌ Very High |
| xtensor dependency | ❌ Tight | ⚠️ Loose | ✓ Minimal | ❌ Very Tight |
| Upstream compat | ❌ Fragile | ⚠️ OK | ✓ Stable | ❌ Fragile |

### Code Examples

#### ❌ Option A: Inheritance (NOT FEASIBLE)

```cpp
// HYPOTHETICAL - Requires massive refactoring
template <class mesh_t, class value_t>
class ScalarField : public xt::xtensor<value_t, 1>  // Direct inheritance
{
    // Lose: field_expression, inner_mesh_type, inner_field_types
    // Must re-implement ALL Samurai functionality on top of xtensor

    mesh_t* m_mesh;  // Ad-hoc mesh reference (no lifetime management!)
    std::string m_name;
    // ... lose all SAMURAI architecture benefits ...

    // xtensor-python would work, BUT:
    // 1. Breaks expression templates (no mesh context)
    // 2. Loses storage backend abstraction
    // 3. Lifetime management broken
    // 4. AMR operations don't fit xtensor model
};
```

#### ❌ Option B: Direct Exposure (NOT RECOMMENDED)

```python
# PROBLEMATIC API
field = sam.ScalarField2D(mesh, "u")
arr = field.array()  # Returns xtensor container

# User must manage mesh separately (error-prone!)
mesh_ref = mesh  # User must keep mesh alive
del mesh  # Oops! Field now has dangling pointer

# Lost Samurai semantics
arr.fill(1.0)  # Works (xtensor method)
arr.resize()   # Wrong! Doesn't update mesh structure
arr.save()     # Doesn't exist (Field method)

# Inconsistent API
field.save("file.h5")  # Field method
arr.fill(1.0)          # xtensor method
# User must know which operations go where!
```

#### ✓ Option C: Adapter (RECOMMENDED)

```python
# CLEAN, CONSISTENT API
import samurai_python as sam
import numpy as np

# Create mesh and field (lifetime managed automatically)
mesh = sam.mesh.make(box, min_level=4, max_level=8)
field = sam.field.scalar(mesh, "u", init=0.0)
# ^^^ Field keeps Mesh alive, Mesh keeps Domain alive

# Field operations (Samurai semantics)
field.fill(1.0)      # Samurai method
field.resize()       # Samurai method (AMR-aware)
field.save("u.h5")   # Samurai method (HDF5)

# NumPy integration (when needed)
arr = field.numpy_view()  # Zero-copy view
arr[:] = np.sin(x)        # Fast NumPy ops
# Field and arr share same memory!

# Method chaining
field.fill(0.0).save("initial.h5")

# AMR adaptation (safe!)
sam.adapt(mesh, config)
field.resize()  # Updates to new mesh structure
```

---

## Performance Characteristics

### Memory Layout

```
Current Implementation (Option C):

┌─────────────────────────────────────────────────────────────┐
│ Python: field = sam.field.scalar(mesh, "u")                 │
└─────────────────────────────────────────────────────────────┘
                        │
                        │ 1. C++ allocates xtensor storage
                        ▼
┌─────────────────────────────────────────────────────────────┐
│ C++: Field::m_storage.m_data (xt::xtensor<double,1>)        │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ [0.0][0.0][0.0][0.0]...  (contiguous memory)            │ │
│ └─────────────────────────────────────────────────────────┘ │
│   Address: 0x7f1234000000                                   │
└─────────────────────────────────────────────────────────────┘
                        │
                        │ 2. NumPy view (ZERO-COPY)
                        ▼
┌─────────────────────────────────────────────────────────────┐
│ Python: arr = field.numpy_view()                            │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ numpy.ndarray (shares memory with field)                │ │
│ │  Address: 0x7f1234000000  (SAME AS ABOVE!)              │ │
│ │  [0.0][0.0][0.0][0.0]...                                │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                              │
│  arr[0] = 1.0  →  Writes directly to field's memory        │
└─────────────────────────────────────────────────────────────┘
                        │
                        │ 3. No copies, full speed
                        ▼
                      Performance:
                      - arr[:] = np.sin(x)  →  NumPy-optimized
                      - field.fill(1.0)      →  xtensor-optimized
                      - Both operate on SAME memory
```

### Operation Timings (Approximate)

```
Operation               | Option C (Current) | xtensor-python (hypothetical)
------------------------|-------------------|------------------------------
Field creation          | 1.0x (baseline)   | 1.0x (same)
Fill with scalar        | 1.0x (xtensor)    | 1.0x (xtensor)
NumPy element access    | 1.0x (zero-copy)  | 1.0x (zero-copy)
NumPy vectorized ops    | 1.0x (NumPy)      | 1.0x (NumPy)
Field arithmetic        | 1.2x (wrapper)    | 1.0x (inlined)
AMR adaptation          | 1.0x (native)     | ❌ Not supported
HDF5 I/O                | 1.0x (native)     | ❌ Not supported

Overall:  Equivalent performance for data operations
         Current approach is FASTER for domain-specific ops
```

---

## Recommendation Summary

### ✅ USE: Custom pybind11 with NumPy Buffer Protocol (Option C)

**Why:**
1. **Architectural fit**: No C++ changes required
2. **Lifetime safety**: Automatic ownership management
3. **Performance**: Zero-copy buffer protocol
4. **Semantic completeness**: Field operations + NumPy integration
5. **Maintainability**: Localized, testable code
6. **Flexibility**: Backend-agnostic (xtensor/Eigen)

### ❌ DO NOT USE: xtensor-python

**Why:**
1. **Wrong type model**: Field doesn't inherit from xcontainer
2. **Lifetime issues**: Can't express mesh dependency
3. **Semantic mismatch**: AMR operations don't map to xtensor
4. **Expression templates**: Require mesh context
5. **Maintenance burden**: Forking/modifying xtensor-python required

---

## Implementation Roadmap

### Phase 1: Enhance Current Bindings (WEEK 1-2)

- [ ] Add template helpers to reduce boilerplate
- [ ] Implement missing NumPy-like methods (`flatten`, `astype`)
- [ ] Add property-based accessors (`shape`, `ndim`, `dtype`)
- [ ] Improve error messages with type checking

### Phase 2: Expand NumPy Integration (WEEK 3-4)

- [ ] Support NumPy array protocol (not just buffer)
- [ ] Add `__array__` method for implicit conversion
- [ ] Implement NumPy ufunc support
- [ ] Add Dask compatibility layer

### Phase 3: Pythonic Enhancements (WEEK 5-6)

- [ ] Context manager for file I/O
- [ ] Progress bars for long operations
- [ ] Jupyter notebook integration
- [ ] Visualization helpers

---

## Conclusion

The current custom pybind11 implementation with NumPy buffer protocol is **architecturally sound**, **performant**, and **maintainable**. xtensor-python is **not suitable** for Samurai Fields due to fundamental architectural mismatches.

**Recommendation**: Continue with Option C, enhance with better helpers and NumPy-like methods.
