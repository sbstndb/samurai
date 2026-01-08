# POC In-Place Operator Test Suite Guide

## Overview

This test suite validates the **Proof of Concept (POC)** for in-place arithmetic operators (`+=`, `-=`, `*=`, `/=`) on ScalarField objects.

**Test File**: `/home/sbstndbs/sbstndbs/samurai/python/tests/test_poc_inplace_ops.py`

## Purpose

The POC aims to prove that in-place operators:

1. **Modify fields in-place** (same object `id()`)
2. **Return self for chaining** (`(u += 1) *= 2`)
3. **Work after mesh adaptation** (no stale references - CRITICAL!)
4. **Provide performance benefits** over copy-based operators

## Test Structure

### 1. TestScalarFieldInPlaceOps (13 tests)
**Basic functionality tests**

- `test_iadd_scalar` - `field += scalar` modifies in-place
- `test_isub_scalar` - `field -= scalar` modifies in-place
- `test_imul_scalar` - `field *= scalar` modifies in-place
- `test_itruediv_scalar` - `field /= scalar` modifies in-place
- `test_iadd_field` - `field += other_field` adds element-wise
- `test_isub_field` - `field -= other_field` subtracts element-wise
- `test_returns_self` - In-place ops return self for chaining
- `test_modifies_in_place` - Original object is modified (not copied)
- `test_2d_inplace_ops` - Operations work in 2D
- `test_zero_initialization` - Works with zero-initialized fields
- `test_negative_values` - Handles negative values correctly
- `test_fractional_values` - Handles fractional values correctly
- `test_chained_field_ops` - Complex chained operations work

### 2. TestInPlaceOpsAfterMeshAdaptation (5 tests)
**CRITICAL POC validation - solves the stale reference problem**

- `test_iadd_after_adaptation` - In-place ops work after mesh adaptation
- `test_no_stale_reference_after_adaptation` - No crash after adaptation (KEY TEST!)
- `test_multiple_ops_after_multiple_adaptations` - Multiple ops over multiple adaptations
- `test_comparison_with_copy_operator` - Demonstrates difference from copy-based approach
- `test_field_field_ops_after_adaptation` - Field-to-field ops after adaptation

**Why this is critical:**
- Current problem: `u = u - dt*flux` creates new field with stale mesh reference
- After `MRadaptation()`, this stale reference causes SEGFAULT
- Solution: `u -= dt*flux` modifies in-place, no stale reference

### 3. TestPerformance (2 tests)
**Performance validation**

- `test_inplace_faster_than_copy` - Benchmarks in-place vs copy-based
- `test_memory_efficiency` - Verifies no memory allocation

### 4. TestEdgeCases (4 tests)
**Boundary conditions and edge cases**

- `test_divide_by_zero_raises` - Division by zero handling
- `test_very_large_values` - Large value handling (1e100)
- `test_very_small_values` - Small value handling (1e-100)
- `test_chained_field_ops` - Complex field-to-field chaining

### 5. TestVectorFieldInPlaceOps (2 tests)
**Future work placeholder**

- `test_vector_iadd_scalar` - Skipped (not yet implemented)
- `test_vector_iadd_vector` - Skipped (not yet implemented)

## Running the Tests

### Run all tests:
```bash
cd /home/sbstndbs/sbstndbs/samurai
pytest python/tests/test_poc_inplace_ops.py -v
```

### Run specific test class:
```bash
pytest python/tests/test_poc_inplace_ops.py::TestScalarFieldInPlaceOps -v
```

### Run specific test:
```bash
pytest python/tests/test_poc_inplace_ops.py::TestInPlaceOpsAfterMeshAdaptation::test_no_stale_reference_after_adaptation -v
```

### Run with performance output:
```bash
pytest python/tests/test_poc_inplace_ops.py::TestPerformance -v -s
```

## Test Coverage Summary

| Category | Tests | Focus |
|----------|-------|-------|
| Basic Ops | 13 | `+=`, `-=`, `*=`, `/=` with scalars and fields |
| Adaptation Safety | 5 | **CRITICAL**: No stale references after mesh changes |
| Performance | 2 | Speed and memory efficiency |
| Edge Cases | 4 | Boundary conditions and extreme values |
| Future Work | 2 | VectorField support (skipped) |

**Total: 26 tests** (24 active, 2 skipped)

## Key Validation Points

### 1. Object Identity
```python
original_id = id(field)
field += 1.0
assert id(field) == original_id  # Same object!
```

### 2. Name Preservation
```python
original_name = field.name  # e.g., "u"
field += 1.0
assert field.name == original_name  # Still "u", not "u_add"
```

### 3. Mesh Adaptation Safety
```python
MRadaptation(config)
field += 1.0  # Should NOT crash (unlike field = field + 1.0)
```

### 4. Method Chaining
```python
(field += 1.0) *= 2.0 -= 1.0  # All operations on same object
```

## Integration with Existing Code

### Current Workaround (advection_2d.py line 181):
```python
# WRONG: Creates new field with stale reference
# unp1 = u - dt * upwind_result

# CORRECT: Uses .assign() workaround
unp1.assign(u - dt * upwind_result)
```

### POC Solution (future):
```python
# BETTER: Direct in-place operation
unp1.assign(u)
unp1 -= dt * upwind_result

# OR EVEN BETTER (if implemented):
unp1 = u - dt * upwind_result  # With operator overloading
```

## Implementation Requirements

For tests to pass, the C++ bindings need to implement:

```cpp
// In field_bindings.cpp
cls.def("__iadd__", [](Field& f, double scalar) -> Field& {
    f.array() += scalar;  // xtensor in-place addition
    return f;
}, py::arg("scalar"), "Add scalar in-place");

cls.def("__isub__", [](Field& f, double scalar) -> Field& {
    f.array() -= scalar;
    return f;
}, py::arg("scalar"), "Subtract scalar in-place");

cls.def("__imul__", [](Field& f, double scalar) -> Field& {
    f.array() *= scalar;
    return f;
}, py::arg("scalar"), "Multiply by scalar in-place");

cls.def("__itruediv__", [](Field& f, double scalar) -> Field& {
    f.array() /= scalar;
    return f;
}, py::arg("scalar"), "Divide by scalar in-place");

// Field-to-field versions
cls.def("__iadd__", [](Field& f, const Field& other) -> Field& {
    f.array() = f.array() + other.array();  // Element-wise
    return f;
}, py::arg("other"), "Add field in-place");
```

## Success Criteria

The POC is successful if:

1. ✅ All basic operation tests pass (13 tests)
2. ✅ **All adaptation safety tests pass (5 tests) - MOST CRITICAL**
3. ✅ Performance tests show speedup (or at least no regression)
4. ✅ Edge cases are handled correctly

## Next Steps After POC Validation

1. Implement in-place operators in C++ bindings
2. Run this test suite to validate
3. Update examples (advection_2d.py) to use in-place ops
4. Add VectorField in-place ops
5. Benchmark performance improvements in real simulations

## References

- Current bindings: `/home/sbstndbs/sbstndbs/samurai/python/src/bindings/field_bindings.cpp`
- Example usage: `/home/sbstndbs/sbstndbs/samurai/python/examples/advection_2d.py`
- Current workaround: Line 181 uses `.assign()` method
