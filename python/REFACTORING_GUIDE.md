# Field Bindings Refactoring Guide

## Executive Summary

This document provides guidance for future refactoring of `field_bindings.cpp` to reduce code duplication while maintaining API stability and code quality.

**Current Status:**
- File size: 1,242 lines
- Duplication: ~41% (510 lines)
- Recommendation: **DO NOT refactor yet** - read below

---

## Why NOT to Refactor Right Now

### 1. Insufficient Test Coverage

**Problem:** Arithmetic operators (`__add__`, `__sub__`, `__mul__`, `__truediv__`) are heavily used in examples but have NO dedicated tests.

**Risk:** Any change to operator behavior could introduce subtle bugs in numerical schemes that would go undetected.

**Solution First:** Run `pytest tests/test_field_arithmetic.py` (newly added) to establish baseline.

### 2. API Stability

**Current Usage:**
- 5 example scripts use field operations
- 15 test files with 260+ test methods
- Unknown external user code

**Requirement:** 100% backward compatibility is mandatory. Any breaking change requires a major version bump.

### 3. Cost-Benefit Analysis

| Factor | Current | After Refactoring |
|--------|---------|-------------------|
| Lines of code | 1242 | ~800 (-35%) |
| Compile time | Baseline | +10% (worse) |
| Onboarding difficulty | Medium | Hard |
| Maintenance | Higher (3× bug fix cost) | Lower (1×) |
| Time investment | 0 | 15-21 hours |

**ROI Timeline:**
- Short-term (3 months): Negative
- Medium-term (6-12 months): Neutral
- Long-term (1+ year): Positive

---

## Current Duplication Patterns

### Pattern 1: Arithmetic Operator Helpers (Lines 40-102)

```cpp
// Current: Separate functions for each dimension
template <std::size_t dim>
ScalarField<dim> field_sub_scalar(const ScalarField<dim>& field, double scalar)
{
    auto& mesh = const_cast<typename ScalarField<dim>::mesh_t&>(field.mesh());
    auto result = samurai::make_scalar_field<double>(field.name() + "_sub", mesh);
    result = field - scalar;
    return result;
}

// Duplicated for: add, mul, div, field-field operations
```

**Future refactoring approach:** Template with `if constexpr` for ScalarField vs VectorField

### Pattern 2: Time-Stepping Functions (Lines 1087-1241)

```cpp
// Current: 9 identical functions (3 dims × 3 operations)
m.def("euler_update_1d", [](ScalarField<1>& unp1, ...) { unp1 = u - dt * du; });
m.def("euler_update_2d", [](ScalarField<2>& unp1, ...) { unp1 = u - dt * du; });
m.def("euler_update_3d", [](ScalarField<3>& unp1, ...) { unp1 = u - dt * du; });
// ... same for rk3_stage2, rk3_stage3, swap_field_arrays
```

**Future refactoring approach:** Single template function bound with overload resolution

### Pattern 3: Factory Functions (Lines 868-1066)

```cpp
// Current: 6 make_scalar_field overloads with identical bodies
m.def("make_scalar_field", [](MRMesh<1>& mesh, const std::string& name, double init) {
    return samurai::make_scalar_field<double>(name, mesh, init);
});
m.def("make_scalar_field", [](MRMesh<2>& mesh, const std::string& name, double init) {
    return samurai::make_scalar_field<double>(name, mesh, init);
});
// ... 3D version
```

**Future refactoring approach:** Variadic template with `std::index_sequence`

---

## When TO Refactor

Refactoring is appropriate when:

1. ✅ **Tests pass:** All arithmetic operator tests pass (test_field_arithmetic.py)
2. ✅ **Need arises:** Adding 4D support or new field types
3. ✅ **Team capacity:** 2+ days available for testing and validation
4. ✅ **Stable API:** No pending API changes expected soon

---

## How TO Refactor (When Ready)

### Phase 1: Create Arithmetic Helper Template

```cpp
// NEW: Generic helper for field-scalar operations
template <typename Field, typename Mesh, typename ResultFactory>
auto bind_field_scalar_op(py::class_<Field>& cls,
                         const char* op_name,
                         const char* suffix,
                         ResultFactory&& factory)
{
    cls.def(op_name,
        [&factory](Field& f, double scalar) {
            auto& mesh = const_cast<Mesh&>(f.mesh());
            auto result = factory(f, suffix, mesh);
            apply_op(result, f, scalar, op_name);
            return result;
        },
        py::arg("scalar"),
        std::string(op_name) + " scalar operation");
}

// Usage in bind_scalar_field and bind_vector_field
bind_field_scalar_op<Field, Mesh>(cls, "__sub__", "sub", make_field_like<Field>);
bind_field_scalar_op<Field, Mesh>(cls, "__add__", "add", make_field_like<Field>);
// ... etc
```

**Benefits:**
- Eliminates ~150 lines of duplication
- Single source of truth for arithmetic logic
- Consistent behavior across ScalarField and VectorField

### Phase 2: Template-ify Time-Stepping Functions

```cpp
// NEW: Single template implementation
template <std::size_t dim>
void euler_update_impl(ScalarField<dim>& unp1,
                       const ScalarField<dim>& u,
                       double dt,
                       const ScalarField<dim>& du)
{
    unp1 = u - dt * du;
}

// Binding with overload resolution
m.def("euler_update", &euler_update_impl<1>, py::arg("unp1"), ...);
m.def("euler_update", &euler_update_impl<2>, py::arg("unp1"), ...);
m.def("euler_update", &euler_update_impl<3>, py::arg("unp1"), ...);
```

**Benefits:**
- Eliminates ~100 lines
- Bug fixes applied once, not 3×
- Easier to add new time-stepping schemes

### Phase 3: Consolidate Factory Functions

```cpp
// NEW: Generic factory with dimension loop
template <std::size_t... Dims>
void bind_make_scalar_field_all(py::module_& m, std::index_sequence<Dims...>)
{
    (m.def("make_scalar_field",
        [](MRMesh<Dims>& mesh, const std::string& name, double init) {
            return samurai::make_scalar_field<double>(name, mesh, init);
        },
        py::arg("mesh"), py::arg("name"), py::arg("init_value") = 0.0,
        ("Create " + std::to_string(Dims) + "D scalar field").c_str()
    ), ...);
}

// Usage
bind_make_scalar_field_all(m, std::make_index_sequence<3>{});
```

**Benefits:**
- Eliminates ~150 lines
- Adding 4D support requires one line

---

## Testing Strategy (Required Before Refactoring)

### 1. Baseline Tests

```bash
# MUST PASS before any refactoring
pytest tests/test_field_arithmetic.py -v
pytest tests/test_field.py -v
pytest tests/test_vectorfield_setitem.py -v
```

### 2. Integration Tests

```bash
# Verify examples still work after refactoring
python examples/linear_convection.py
python examples/advection_2d.py
python examples/burgers_2d_simple.py
```

### 3. Regression Tests

After each refactoring phase:

```bash
# Run full test suite
pytest tests/ -v

# Check for memory leaks
valgrind --leak-check=full python tests/test_field_arithmetic.py
```

---

## Common Pitfalls to Avoid

### ❌ Don't: Change Python API Names

```cpp
// WRONG: Breaks user code
py::class_<ScalarField<2>>(m, "ScalarField<2>")  // Don't use template syntax

// CORRECT: Keep existing names
py::class_<ScalarField<2>>(m, "ScalarField2D")
```

### ❌ Don't: Remove Existing Methods

```cpp
// WRONG: Removes backward compatibility
// Don't remove: field_sub_scalar, field_add_scalar, etc.

// CORRECT: Keep old functions, add new templated ones alongside
```

### ❌ Don't: Use Macro-Based Approaches

```cpp
// AVOID: Macros make debugging harder
#define BIND_FIELD_OP(op) \
    cls.def(#op, [](Field& f, double s) { return f op s; })

// PREFER: Templates with better type safety
template <typename Field>
void bind_field_ops(py::class_<Field>& cls) { ... }
```

### ✅ Do: Maintain Exact Python-Level Behavior

```cpp
// Ensure result fields have correct naming
result.name() = field.name() + "_sub";  // Keep this pattern

// Ensure arithmetic operators create NEW fields
auto result = make_field_like<Field>(...);  // Don't modify in-place
```

---

## Template Best Practices for This Codebase

### 1. Follow Existing Patterns

The codebase already uses template helpers successfully:

```cpp
// From box_bindings.cpp - GOOD EXAMPLE
template <std::size_t dim>
void bind_box(py::module_& m, const std::string& name)
{
    py::class_<Box<double, dim>>(m, name.c_str())
        .def(py::init<>(...))
        .def("length", &Box<double, dim>::length);
}
```

### 2. Use `if constexpr` for Type-Specific Behavior

```cpp
template <typename Field>
void bind_field_specific_methods(py::class_<Field>& cls)
{
    // Common methods
    cls.def("size", [](const Field& f) { return f.mesh().nb_cells(); });

    // Type-specific behavior
    if constexpr (Field::n_comp == 1) {
        // ScalarField only
        cls.def("is_scalar", []() { return true; });
    } else {
        // VectorField only
        cls.def("n_components", [](const Field& f) { return Field::n_comp; });
    }
}
```

### 3. Use `std::index_sequence` for Unrolling

```cpp
template <std::size_t... Dims>
void bind_all_dimensions(py::module_& m, std::index_sequence<Dims...>)
{
    // Expands to: bind_dim<1>(m), bind_dim<2>(m), bind_dim<3>(m)
    (bind_dim<Dims>(m), ...);
}

// Usage
bind_all_dimensions(m, std::make_index_sequence<3>{});
```

---

## Success Criteria

Refactoring is successful when:

1. ✅ All existing tests pass (100% pass rate)
2. ✅ New arithmetic tests pass (test_field_arithmetic.py)
3. ✅ All example scripts produce identical output
4. ✅ No new compiler warnings
5. ✅ Code reduction > 30%
6. ✅ Added documentation for new patterns
7. ✅ Code review approval from 2+ maintainers

---

## Quick Reference: File Structure

```
python/src/bindings/
├── field_bindings.cpp (1242 lines) ← THIS FILE
│   ├── Lines 40-102:    Arithmetic helper functions (scalar only)
│   ├── Lines 120-236:   bind_field_common_methods (used by both)
│   ├── Lines 240-388:   bind_scalar_field (×3 dims)
│   ├── Lines 392-716:   bind_vectorfield_methods (vector-specific)
│   ├── Lines 720-786:   bind_vector_field (×6 variants)
│   ├── Lines 868-1066:  Factory functions
│   └── Lines 1087-1241: Time-stepping helpers
```

---

## Resources

### Internal References
- `box_bindings.cpp` - Example of good template-based bindings
- `mesh_config_bindings.cpp` - Fluent interface pattern
- `operator_bindings.cpp` - Overload resolution patterns

### External References
- [pybind11 documentation](https://pybind11.readthedocs.io/) - Advanced topics
- [pybind11 #3026](https://github.com/pybind/pybind11/issues/3026) - CRTP discussions
- [CppCon 2018: "Cpp17 Features in pybind11"](https://www.youtube.com/watch?v=mCqY5TcaKdY) - Video talk

---

## Contact & Decision Log

**Decision Date:** 2026-01-07
**Decision:** DEFER refactoring pending test coverage
**Next Review:** After test_field_arithmetic.py passes all tests
**Owner:** Python bindings maintainers

**Change Log:**
- 2026-01-07: Initial guide created, decision to defer refactoring
- Future: Update this document when refactoring begins
