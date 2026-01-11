# POC Results: In-Place Operators

**Branch**: `poc/mutation-semantics`
**Date**: [FILL IN DATE]
**Commit**: [FILL IN COMMIT HASH]

## Overview

This document captures the results of the proof-of-concept (POC) implementation of in-place mutation operators for Samurai Python fields. The POC aims to solve the "stale reference" problem where expression results can become invalid after mesh adaptation.

## Performance Results

### Test Environment

- **Branch**: `poc/mutation-semantics`
- **Date**: [FILL IN]
- **System**: [FILL IN - CPU, RAM, OS]
- **Build**: Release mode, [WITH_MPI status]
- **Mesh sizes**:
  - 2D: Level 6 (64×64 = 4,096 cells)
  - 3D: Level 4 (16×16×16 = 4,096 cells)

---

### Benchmark 1: Simple Scalar Operation (2D)

**Operation**: `u += 1.0` vs `u = u + 1.0`

| Method | Time (100 iterations) | Relative | Speedup |
|--------|----------------------|----------|---------|
| In-place (`+=`) | TBD s | 1.0x (baseline) | - |
| Copy (`u = u +`) | TBD s | TBD | TBD |

**Expected**: In-place should be ~2x faster by avoiding field allocation and copy.

**Analysis**: [FILL AFTER RUNNING]
- [ ] In-place is significantly faster (>1.5x)
- [ ] In-place is moderately faster (1.2-1.5x)
- [ ] No significant difference (<1.2x)
- [ ] In-place is slower (unexpected!)

---

### Benchmark 2: Simple Scalar Operation (3D)

**Operation**: `u += 1.0` vs `u = u + 1.0`

| Method | Time (100 iterations) | Relative | Speedup |
|--------|----------------------|----------|---------|
| In-place (`+=`) | TBD s | 1.0x (baseline) | - |
| Copy (`u = u +`) | TBD s | TBD | TBD |

**Expected**: Similar speedup to 2D case.

**Analysis**: [FILL AFTER RUNNING]
- Does the speedup scale to higher dimensions?

---

### Benchmark 3: Field Arithmetic

**Operation**: `u += v` (field-field operation)

| Method | Time (100 iterations) | Relative | Speedup |
|--------|----------------------|----------|---------|
| In-place (`u += v`) | TBD s | 1.0x (baseline) | - |
| Copy (`u = u + v`) | TBD s | TBD | TBD |

**Expected**: In-place should be ~2x faster.

**Analysis**: [FILL AFTER RUNNING]

---

### Benchmark 4: Complex Expression

**Operation**: `u1.assign(u - dt * flux)` vs `u1.assign(u); u1 -= dt * flux`

| Method | Time (100 iterations) | Relative | Difference |
|--------|----------------------|----------|------------|
| `assign(u - dt * flux)` | TBD s | 1.0x (baseline) | - |
| `assign(u); u1 -= dt*flux` | TBD s | TBD | TBD |

**Expected**: The in-place version might be slower due to two passes, but could be faster if it avoids temporary allocation for `dt * flux`.

**Analysis**: [FILL AFTER RUNNING]
- Which approach is better for complex expressions?
- Should we recommend `assign()` for complex cases?

---

### Benchmark 5: Chained Operations

**Operation**: `u += a; u *= b; u -= c` vs `u = ((u + a) * b) - c`

| Method | Time (100 iterations) | Relative | Speedup |
|--------|----------------------|----------|---------|
| Chained in-place | TBD s | 1.0x (baseline) | - |
| Chained copy | TBD s | TBD | TBD |

**Expected**: In-place should be much faster (3-4x) by avoiding 3 temporary allocations.

**Analysis**: [FILL AFTER RUNNING]

---

### Benchmark 6: All Operators

**Operation**: Test all in-place operators for consistency

| Operator | Time (100 iterations) | Relative |
|----------|----------------------|----------|
| `+=` | TBD s | 1.0x |
| `-=` | TBD s | TBD |
| `*=` | TBD s | TBD |
| `/=` | TBD s | TBD |

**Expected**: All operators should have similar performance.

**Analysis**: [FILL AFTER RUNNING]
- Are all operators consistently implemented?
- Any outliers?

---

## Functional Test Results

### Test Suite Status

Run with: `pytest python/tests/test_inplace_operators.py -v`

**All Tests Passing?**

- [ ] `test_iadd_scalar` - u += 1.0
- [ ] `test_isub_scalar` - u -= 1.0
- [ ] `test_imul_scalar` - u *= 2.0
- [ ] `test_itruediv_scalar` - u /= 2.0
- [ ] `test_iadd_field` - u += v
- [ ] `test_isub_field` - u -= v
- [ ] `test_imul_field` - u *= v
- [ ] `test_itruediv_field` - u /= v
- [ ] `test_returns_self` - assert (u += 1) is u
- [ ] `test_modifies_in_place` - no new allocation
- [ ] `test_iadd_after_adaptation` - no stale reference
- [ ] `test_no_stale_reference` - reference stays valid
- [ ] `test_chained_operations` - (u += a) *= b
- [ ] `test_operator_precedence` - u += a * b

**Pass Rate**: [FILL IN] / 14 tests passed

---

### Issues Found

[FILL IN - Any bugs, edge cases, failures]

**Known Issues**:
- [ ] None
- [ ] [Issue 1 description]
- [ ] [Issue 2 description]

**Edge Cases Tested**:
- [ ] Empty fields
- [ ] Single-cell fields
- [ ] Very large fields (memory)
- [ ] Fields with different mesh sizes
- [ ] Chained adaptations (adapt, then in-place op, then adapt again)

---

## Performance vs Safety Trade-off

### User Experience Comparison

#### Before POC (Problematic)
```python
# User writes natural code
u1 = u - dt * flux

# Later, mesh adaptation invalidates u1
MRadaptation(config)

# CRASH! u1 references old mesh data
u1.resize()  # Segfault or wrong results
```

#### After POC (Solution A: In-place)
```python
# User must use in-place operations
u1.assign(u)
u1 -= dt * flux

# Adaptation is safe
MRadaptation(config)

# OK! u1 references current mesh data
u1.resize()
```

#### After POC (Solution B: Assign)
```python
# User uses assign (always safe)
u1.assign(u - dt * flux)

# Adaptation is safe
MRadaptation(config)

# OK! u1 references current mesh data
u1.resize()
```

### API Comparison

| Operation | Pre-POC | Post-POC (In-place) | Post-POC (Assign) |
|-----------|---------|---------------------|-------------------|
| Simple | `u1 = u + 1` | `u1.assign(u); u1 += 1` | `u1.assign(u + 1)` |
| Complex | `u1 = u - dt*v` | `u1.assign(u); u1 -= dt*v` | `u1.assign(u - dt*v)` |
| Chained | `u1 = ((u+a)*b)-c` | `u1.assign(u); u1+=a; u1*=b; u1-=c` | `u1.assign(((u+a)*b)-c)` |
| Natural? | Yes | No | Somewhat |
| Safe? | No | Yes | Yes |
| Fast? | Medium | Fast | Fast |

---

## Recommendation

[DECISION: Implement fully / Refine approach / Abandon - FILL AFTER TESTING]

### Rationale

[EXPLAIN WHY - Base on performance results and user experience]

**Options**:
1. **Implement in-place operators fully** - If performance gain is significant (>1.5x)
2. **Refine to only provide assign()** - If performance gain is minimal (<1.2x)
3. **Abandon in-place, keep status quo** - If implementation is too complex

**Decision Criteria**:
- Performance improvement: [FILL IN]
- Implementation complexity: [FILL IN]
- User experience impact: [FILL IN]
- Maintenance burden: [FILL IN]

### Next Steps

Based on the recommendation:

**If implementing fully**:
- [ ] Add pybind11 bindings for `__iadd__`, `__isub__`, `__imul__`, `__itruediv__`
- [ ] Add unit tests for all operators
- [ ] Update Python documentation with best practices
- [ ] Add warning in docs about stale references with `u = u + 1`
- [ ] Consider deprecating `u = u + 1` pattern (or add runtime warning)

**If providing only assign()**:
- [ ] Document `assign()` as the recommended way to copy fields
- [ ] Add examples showing `assign()` pattern
- [ ] Consider adding runtime warning for `u = u + 1` pattern
- [ ] Close POC branch, merge documentation updates

**If abandoning**:
- [ ] Document the stale reference issue clearly
- [ ] Provide workaround in documentation
- [ ] Close POC branch without merging

---

## Appendix: Code Examples

### The Stale Reference Problem

```python
import samurai_python as sam

# Setup
box = sam.Box2D([0.0, 0.0], [1.0, 1.0])
config = sam.MeshConfig2D(min_level=4, max_level=4)
mesh = sam.MRMesh2D(box, config)

# Create fields
u = sam.field.scalar(mesh, "u", init=1.0)

# Create expression (creates temporary field)
u1 = u + 1.0

# u1 now references the temporary field's mesh

# Mesh adaptation happens
new_config = sam.MeshConfig2D(min_level=5, max_level=5)
mesh = sam.MRMesh2D(box, new_config)

# u1's internal mesh pointer is now STALE
# u1.mesh points to the old mesh (freed or invalid)

# CRASH!
print(u1.mesh.level)  # Segfault or undefined behavior
```

### Solution 1: In-Place Operators

```python
import samurai_python as sam

# Setup
box = sam.Box2D([0.0, 0.0], [1.0, 1.0])
config = sam.MeshConfig2D(min_level=4, max_level=4)
mesh = sam.MRMesh2D(box, config)

# Create fields
u = sam.field.scalar(mesh, "u", init=1.0)
u1 = sam.field.scalar(mesh, "u1", init=0.0)

# Use in-place operation
u1.assign(u)
u1 += 1.0

# u1 is the same object, with same mesh reference

# Mesh adaptation
new_config = sam.MeshConfig2D(min_level=5, max_level=5)
mesh = sam.MRMesh2D(box, new_config)

# OK! u1 still references the current mesh
print(u1.mesh.level)  # Works correctly
```

### Solution 2: Assign Method

```python
import samurai_python as sam

# Setup (same as above)
mesh = sam.MRMesh2D(box, config)
u = sam.field.scalar(mesh, "u", init=1.0)
u1 = sam.field.scalar(mesh, "u1", init=0.0)

# Use assign (works with expressions)
u1.assign(u + 1.0)

# u1 has its own data, no expression field created

# Mesh adaptation
new_config = sam.MeshConfig2D(min_level=5, max_level=5)
mesh = sam.MRMesh2D(box, new_config)

# OK! u1 has its own data, independent of any expression
print(u1.mesh.level)  # Works correctly
```

### Performance Comparison

```python
import timeit

# Method 1: In-place
def inplace_method():
    u1.assign(u)
    u1 += 1.0
    u1 *= 2.0
    u1 -= 0.5

# Method 2: Assign with expression
def assign_method():
    u1.assign(((u + 1.0) * 2.0) - 0.5)

# Method 3: Naive (pre-POC, creates temporaries)
def naive_method():
    u1 = u + 1.0
    u1 = u1 * 2.0
    u1 = u1 - 0.5

time_inplace = timeit.timeit(inplace_method, number=1000)
time_assign = timeit.timeit(assign_method, number=1000)
time_naive = timeit.timeit(naive_method, number=1000)

print(f"In-place: {time_inplace:.4f}s")
print(f"Assign:   {time_assign:.4f}s")
print(f"Naive:    {time_naive:.4f}s")
```

---

## Implementation Notes

### Files Modified

- [ ] `include/samurai/field.hpp` - Added in-place operators
- [ ] `python/bindings/field.cpp` - Added pybind11 bindings
- [ ] `python/tests/test_inplace_operators.py` - Added tests
- [ ] `python/benchmarks/benchmark_poc_inplace.py` - This file

### Implementation Details

**C++ Implementation**:
```cpp
// In ScalarField class
ScalarField& operator+=(const ScalarField& other) {
    // In-place addition
    return *this;
}

ScalarField& operator+=(value_t scalar) {
    // In-place scalar addition
    return *this;
}
```

**Python Bindings**:
```cpp
// In field.cpp
class_<ScalarField>(m, "ScalarField")
    .def("__iadd__", &ScalarField::operator+=<ScalarField>,
         return_value_policy::reference_internal)
    .def("__iadd__", &ScalarField::operator+=<double>,
         return_value_policy::reference_internal);
```

### Testing Strategy

1. **Unit tests**: Verify correctness of each operator
2. **Performance tests**: Measure speedup vs copy operations
3. **Integration tests**: Test with mesh adaptation
4. **Memory tests**: Verify no leaks or invalid accesses

---

## Conclusion

[SUMMARY OF FINDINGS - FILL AFTER TESTING]

The POC demonstrates that in-place operators [do/do not] provide significant performance benefits while solving the stale reference problem. Based on the results, we recommend [ACTION].

**Impact**:
- Performance: [FILL IN]
- User experience: [FILL IN]
- Code complexity: [FILL IN]
- Maintenance: [FILL IN]

**Final Decision**: [FILL IN]
