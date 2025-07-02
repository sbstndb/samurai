# Subset System

## Introduction

The subset system in Samurai is one of its most powerful features, providing set algebra operations on mesh regions. This system allows efficient manipulation and iteration over specific parts of the mesh, enabling complex mesh operations and numerical schemes that work on different mesh regions.

## Core Concepts

### 1. Subset Operations

The subset system provides fundamental set operations:

- **Self**: Complete mesh at a given level
- **Intersection**: Common cells between different mesh regions
- **Union**: Combined cells from different regions
- **Difference**: Cells in one region but not another
- **Translate**: Shifted regions of the mesh

### 2. Subset Types

```cpp
template <class Derived>
class Subset
{
    // Base class for all subset operations
};

template <class Mesh>
class Self : public Subset<Self<Mesh>>
{
    // Represents the complete mesh
};

template <class Left, class Right>
class Intersection : public Subset<Intersection<Left, Right>>
{
    // Represents intersection of two subsets
};
```

## Basic Subset Operations

### 1. Self Operation

The `self` operation creates a subset representing the complete mesh at a specific level:

```cpp
// Create subset for entire mesh at level
auto subset = samurai::self(mesh);

// Create subset for specific level
auto level_subset = samurai::self(mesh[level]);

// Create subset for specific mesh region
auto region_subset = samurai::self(mesh[samurai::MRMeshID::cells]);
```

### 2. Intersection Operation

The `intersection` operation finds common cells between two mesh regions:

```cpp
// Intersection of two levels
auto fine_cells = samurai::intersection(mesh[level], mesh[level+1]).on(level);

// Intersection of different mesh regions
auto active_ghosts = samurai::intersection(
    mesh[samurai::MRMeshID::cells], 
    mesh[samurai::MRMeshID::ghosts]
).on(level);
```

### 3. Difference Operation

The `difference` operation finds cells that are in one region but not another:

```cpp
// Cells at level that don't have children
auto coarse_cells = samurai::difference(mesh[level], mesh[level+1]).on(level);

// Active cells that are not ghosts
auto interior_cells = samurai::difference(
    mesh[samurai::MRMeshID::cells], 
    mesh[samurai::MRMeshID::ghosts]
).on(level);
```

### 4. Union Operation

The `union` operation combines cells from different regions:

```cpp
// Combine cells from different levels
auto all_cells = samurai::union_(mesh[level], mesh[level+1]).on(level);

// Combine different mesh regions
auto all_active = samurai::union_(
    mesh[samurai::MRMeshID::cells], 
    mesh[samurai::MRMeshID::ghosts]
).on(level);
```

## Advanced Subset Operations

### 1. Translation

The `translate` operation shifts a subset by a specified offset:

```cpp
// Translate subset by offset
auto shifted = samurai::translate(subset, offset);

// Translate in specific direction
auto shifted_x = samurai::translate(subset, {1, 0});
auto shifted_y = samurai::translate(subset, {0, 1});
```

### 2. Level Specification

Subsets can be restricted to specific levels using the `.on()` method:

```cpp
// Apply operation on specific level
auto subset = samurai::intersection(mesh[level], mesh[level+1]).on(level);

// Apply operation on multiple levels
auto subset = samurai::self(mesh).on(level);
```

## Subset Application

### 1. Basic Application

Subsets are applied using the `apply` function or the function call operator:

```cpp
// Apply operation using apply function
samurai::apply(subset, [&](const auto& interval, const auto& index) {
    // Process interval
    for (auto i = interval.start; i < interval.end; i += interval.step) {
        // Process each cell in interval
    }
});

// Apply operation using function call operator
subset([&](const auto& interval, const auto& index) {
    // Process interval
});
```

### 2. Cell-based Application

For operations that need individual cell access:

```cpp
// Apply on each cell
samurai::apply(subset, [&](const auto& cell) {
    // Process individual cell
    field[cell] = compute_value(cell);
});
```

### 3. Interval-based Application

For operations that work on intervals (more efficient):

```cpp
// Apply on intervals
samurai::apply(subset, [&](const auto& interval, const auto& index) {
    // Process entire interval at once
    auto field_view = field(level, interval, index);
    for (std::size_t i = 0; i < field_view.size(); ++i) {
        field_view[i] = compute_value(level, interval.start + i, index);
    }
});
```

## Common Use Cases

### 1. Multiresolution Operations

```cpp
// Project fine solution to coarse level
auto fine_cells = samurai::intersection(mesh[level], mesh[level+1]).on(level);
fine_cells([&](const auto& interval, const auto& index) {
    auto j = index[0];
    for (auto i = interval.start; i < interval.end; i += interval.step) {
        // Average fine cells to coarse cell
        coarse_field(level, i, j) = 0.25 * (
            fine_field(level+1, 2*i, 2*j) +
            fine_field(level+1, 2*i+1, 2*j) +
            fine_field(level+1, 2*i, 2*j+1) +
            fine_field(level+1, 2*i+1, 2*j+1)
        );
    }
});
```

### 2. Ghost Cell Management

```cpp
// Update ghost cells from neighboring cells
auto ghost_cells = samurai::self(mesh[samurai::MRMeshID::ghosts]).on(level);
ghost_cells([&](const auto& interval, const auto& index) {
    // Copy values from interior cells to ghost cells
    for (auto i = interval.start; i < interval.end; i += interval.step) {
        ghost_field(level, i, index) = interior_field(level, i, index);
    }
});
```

### 3. Boundary Condition Application

```cpp
// Apply boundary conditions on boundary cells
auto boundary_cells = samurai::difference(
    mesh[samurai::MRMeshID::cells], 
    mesh[samurai::MRMeshID::ghosts]
).on(level);

boundary_cells([&](const auto& interval, const auto& index) {
    // Apply boundary condition
    for (auto i = interval.start; i < interval.end; i += interval.step) {
        field(level, i, index) = boundary_value;
    }
});
```

### 4. Stencil Operations

```cpp
// Apply stencil operation with neighbor access
auto interior_cells = samurai::difference(
    mesh[samurai::MRMeshID::cells], 
    mesh[samurai::MRMeshID::ghosts]
).on(level);

interior_cells([&](const auto& interval, const auto& index) {
    auto j = index[0];
    for (auto i = interval.start; i < interval.end; i += interval.step) {
        // 5-point stencil for 2D
        result(level, i, j) = 
            field(level, i, j) +
            field(level, i-1, j) + field(level, i+1, j) +
            field(level, i, j-1) + field(level, i, j+1);
    }
});
```

## Performance Considerations

### 1. Interval-based Operations

Interval-based operations are more efficient than cell-based operations:

```cpp
// Efficient: Process intervals
subset([&](const auto& interval, const auto& index) {
    // Process entire interval at once
});

// Less efficient: Process individual cells
samurai::apply(subset, [&](const auto& cell) {
    // Process individual cell
});
```

### 2. Subset Composition

Complex operations can be composed efficiently:

```cpp
// Efficient composition
auto complex_subset = samurai::intersection(
    samurai::difference(mesh[level], mesh[level+1]),
    samurai::self(mesh[samurai::MRMeshID::cells])
).on(level);
```

### 3. Memory Access Patterns

Subsets help maintain good memory access patterns:

```cpp
// Contiguous memory access within intervals
subset([&](const auto& interval, const auto& index) {
    auto field_view = field(level, interval, index);
    // field_view provides contiguous access
});
```

## Examples

### 1. Basic Subset Usage

```cpp
#include <samurai/subset.hpp>

int main() {
    // Create mesh
    samurai::MRMesh<Config> mesh(box, min_level, max_level);
    
    // Create field
    auto field = samurai::make_scalar_field<double>("u", mesh);
    
    // Create subset for fine cells
    auto fine_cells = samurai::intersection(mesh[level], mesh[level+1]).on(level);
    
    // Apply operation on subset
    fine_cells([&](const auto& interval, const auto& index) {
        auto j = index[0];
        for (auto i = interval.start; i < interval.end; i += interval.step) {
            field(level, i, j) = 1.0;
        }
    });
    
    return 0;
}
```

### 2. Multilevel Operations

```cpp
// Project solution from fine to coarse level
for (std::size_t level = mesh.max_level() - 1; level >= mesh.min_level(); --level) {
    auto fine_cells = samurai::intersection(mesh[level], mesh[level+1]).on(level);
    
    fine_cells([&](const auto& interval, const auto& index) {
        auto j = index[0];
        for (auto i = interval.start; i < interval.end; i += interval.step) {
            // 2D projection: average 4 fine cells to 1 coarse cell
            coarse_field(level, i, j) = 0.25 * (
                fine_field(level+1, 2*i, 2*j) +
                fine_field(level+1, 2*i+1, 2*j) +
                fine_field(level+1, 2*i, 2*j+1) +
                fine_field(level+1, 2*i+1, 2*j+1)
            );
        }
    });
}
```

### 3. Complex Subset Operations

```cpp
// Find cells that are active but not ghosts and have neighbors
auto interior_cells = samurai::difference(
    samurai::difference(mesh[samurai::MRMeshID::cells], mesh[samurai::MRMeshID::ghosts]),
    samurai::intersection(mesh[level], mesh[level+1])
).on(level);

interior_cells([&](const auto& interval, const auto& index) {
    // Apply numerical scheme on interior cells
    auto j = index[0];
    for (auto i = interval.start; i < interval.end; i += interval.step) {
        // Compute numerical scheme
        result(level, i, j) = compute_scheme(level, i, j);
    }
});
```

### 4. Translation Operations

```cpp
// Apply stencil with translation
auto interior = samurai::difference(
    mesh[samurai::MRMeshID::cells], 
    mesh[samurai::MRMeshID::ghosts]
).on(level);

// Apply 5-point stencil
interior([&](const auto& interval, const auto& index) {
    auto j = index[0];
    for (auto i = interval.start; i < interval.end; i += interval.step) {
        result(level, i, j) = 
            field(level, i, j) +
            field(level, i-1, j) + field(level, i+1, j) +
            field(level, i, j-1) + field(level, i, j+1);
    }
});
```

The subset system provides a powerful and efficient way to work with specific regions of the mesh, enabling complex numerical operations while maintaining good performance through interval-based processing and optimized memory access patterns. 