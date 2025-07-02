# Mesh System

## Introduction

The mesh system in Samurai is the foundation of the library, providing a unified representation for various mesh adaptation strategies. It is designed around the concept of intervals and hierarchical cell organization, enabling efficient operations on mesh subsets and supporting different refinement strategies.

## Core Concepts

### 1. Box and Domain

The computational domain is defined by a `Box` object that specifies the spatial extent:

```cpp
template <class T, std::size_t dim>
class Box
{
    point_t m_corner1;  // Lower corner
    point_t m_corner2;  // Upper corner
};
```

```cpp
// Example: Creating a 2D domain
samurai::Box<double, 2> box({0., 0.}, {1., 1.});
```

### 2. Intervals

Intervals are the fundamental building blocks of Samurai's mesh representation. An interval represents a range of cells along one dimension:

```cpp
template <class T>
class Interval
{
    T start;  // Starting index
    T end;    // Ending index (exclusive)
    T step;   // Step size
};
```

Intervals are used to efficiently represent large ranges of cells without storing each cell individually.

### 3. Cell Arrays

Cell arrays organize the mesh cells in a hierarchical structure:

```cpp
template <std::size_t dim, class interval_t, std::size_t max_refinement_level>
class CellArray
{
    using lca_type = LevelCellArray<dim, interval_t>;
    std::array<lca_type, max_refinement_level + 1> m_cells;
};
```

Each level contains a `LevelCellArray` that stores the cells at that refinement level.

### 4. Level Cell Arrays

Level cell arrays manage cells at a specific refinement level:

```cpp
template <std::size_t dim, class interval_t>
class LevelCellArray
{
    using mesh_interval_t = MeshInterval<interval_t>;
    std::vector<mesh_interval_t> m_intervals;
};
```

A `MeshInterval` combines an interval with multi-dimensional indices:

```cpp
template <class interval_t>
struct MeshInterval
{
    interval_t interval;
    xt::xtensor_fixed<typename interval_t::value_t, xt::xshape<dim-1>> index;
};
```

## Mesh Types

### 1. Multiresolution Mesh (MRMesh)

The multiresolution mesh is the primary mesh type in Samurai:

```cpp
template <class Config>
class MRMesh : public Mesh_base<MRMesh<Config>, Config>
{
    // Inherits from Mesh_base with CRTP pattern
};
```

Configuration for MRMesh:

```cpp
template <std::size_t dim_>
struct MRConfig
{
    static constexpr std::size_t dim = dim_;
    static constexpr std::size_t max_refinement_level = 20;
    using mesh_id_t = MRMeshID;
    using interval_t = Interval<int>;
};
```

### 2. Mesh ID System

Different mesh regions are identified using mesh IDs:

```cpp
enum class MRMeshID : std::size_t
{
    reference = 0,  // Reference mesh
    cells = 1,      // Active cells
    ghosts = 2,     // Ghost cells
    count = 3
};
```

## Mesh Operations

### 1. Mesh Construction

```cpp
// Create mesh from box
samurai::MRMesh<Config> mesh(box, min_level, max_level);

// Create mesh from cell list
samurai::CellList<dim> cell_list;
// ... populate cell_list
samurai::MRMesh<Config> mesh(cell_list, min_level, max_level);
```

### 2. Cell Access

```cpp
// Access cells at specific level
auto& level_cells = mesh[level];

// Get cell from coordinates
auto cell = mesh.get_cell(level, x, y, z);

// Get cell from indices
auto cell = mesh.get_cell(level, i, j, k);
```

### 3. Cell Information

```cpp
// Get cell center
auto center = cell.center();

// Get cell coordinates
auto coords = cell.coordinates();

// Get cell level
auto level = cell.level;

// Get cell index
auto index = cell.index;
```

## Cell Representation

### 1. Cell Structure

```cpp
template <std::size_t dim, class interval_t>
class Cell
{
    std::size_t level;
    typename interval_t::index_t index;
    interval_t interval;
    xt::xtensor_fixed<typename interval_t::value_t, xt::xshape<dim-1>> coords;
};
```

### 2. Cell Properties

```cpp
// Cell center coordinates
template <std::size_t d>
auto center() const -> typename interval_t::value_t
{
    return (interval.start + interval.end) * 0.5 * cell_length(level);
}

// Cell volume
auto volume() const -> double
{
    return std::pow(cell_length(level), dim);
}

// Cell length at level
auto cell_length(std::size_t level) const -> double
{
    return domain_length / (1 << level);
}
```

## Mesh Iteration

### 1. Cell Iteration

```cpp
// Iterate over all cells
samurai::for_each_cell(mesh, [&](const auto& cell) {
    // Process cell
    field[cell] = some_value;
});

// Iterate over cells at specific level
samurai::for_each_cell(mesh[level], [&](const auto& cell) {
    // Process cell at level
});
```

### 2. Interval Iteration

```cpp
// Iterate over intervals
samurai::for_each_interval(mesh, [&](std::size_t level, const auto& interval, const auto& index) {
    // Process interval
    for (auto i = interval.start; i < interval.end; i += interval.step) {
        // Process each cell in interval
    }
});
```

### 3. Level-wise Iteration

```cpp
// Iterate level by level
for (std::size_t level = mesh.min_level(); level <= mesh.max_level(); ++level) {
    samurai::for_each_cell(mesh[level], [&](const auto& cell) {
        // Process cell at current level
    });
}
```

## Mesh Adaptation

### 1. Cell Lists for Adaptation

Cell lists are used during mesh adaptation to track which cells need refinement or coarsening:

```cpp
template <std::size_t dim, class interval_t, std::size_t max_refinement_level>
class CellList
{
    std::array<LevelCellList<dim, interval_t>, max_refinement_level + 1> m_cells;
};
```

### 2. Adaptation Process

```cpp
// Create adaptation criterion
auto adaptation = samurai::make_MRAdapt(field);

// Apply adaptation
adaptation(epsilon, regularity);

// Update mesh after adaptation
mesh.update_mesh_neighbour();
```

## Memory Layout

### 1. Contiguous Storage

Cells are stored in contiguous memory for cache efficiency:

```
Memory layout:
[Level 0 cells] [Level 1 cells] [Level 2 cells] ...
```

### 2. Index Mapping

Each cell has a unique global index that maps to its position in the field arrays:

```cpp
// Global index computation
std::size_t global_index = cell.index + level_offset[level];
```

## Performance Optimizations

### 1. Interval Compression

Large ranges of cells are represented as single intervals, reducing memory usage:

```
Instead of: [0,1,2,3,4,5,6,7,8,9]
Use:        [0,10) with step=1
```

### 2. Level-based Organization

Cells are organized by level, enabling efficient level-wise operations:

```cpp
// Efficient level-wise access
auto& level_cells = mesh[level];
for (const auto& interval : level_cells) {
    // Process interval
}
```

### 3. Index Caching

Frequently accessed indices are cached to avoid recomputation:

```cpp
// Cached index access
auto cached_index = mesh.get_interval(level, interval, index);
```

## MPI Support

### 1. Domain Decomposition

For parallel computations, the mesh is decomposed across MPI ranks:

```cpp
// Partition mesh for MPI
mesh.partition_mesh(start_level, global_box);

// Get subdomain for current rank
auto& subdomain = mesh.subdomain();
```

### 2. Ghost Cell Management

Ghost cells are automatically managed for inter-process communication:

```cpp
// Update ghost cells
mesh.update_mesh_neighbour();

// Access ghost cells
auto& ghost_cells = mesh[samurai::MRMeshID::ghosts];
```

## Examples

### 1. Basic Mesh Creation

```cpp
#include <samurai/mr/mesh.hpp>

int main() {
    constexpr size_t dim = 2;
    using Config = samurai::MRConfig<dim>;
    
    // Define domain
    samurai::Box<double, dim> box({0., 0.}, {1., 1.});
    
    // Create mesh
    samurai::MRMesh<Config> mesh(box, 2, 6);
    
    // Access mesh information
    std::cout << "Min level: " << mesh.min_level() << std::endl;
    std::cout << "Max level: " << mesh.max_level() << std::endl;
    std::cout << "Total cells: " << mesh.nb_cells() << std::endl;
    
    return 0;
}
```

### 2. Mesh Iteration

```cpp
// Iterate and print cell information
samurai::for_each_cell(mesh, [&](const auto& cell) {
    std::cout << "Cell at level " << cell.level 
              << ", center: (" << cell.center(0) << ", " << cell.center(1) << ")"
              << ", index: " << cell.index << std::endl;
});
```

### 3. Level-wise Processing

```cpp
// Process each level separately
for (std::size_t level = mesh.min_level(); level <= mesh.max_level(); ++level) {
    std::cout << "Level " << level << " has " 
              << mesh.nb_cells(level) << " cells" << std::endl;
    
    samurai::for_each_cell(mesh[level], [&](const auto& cell) {
        // Process cells at this level
    });
}
```

This mesh system provides the foundation for all numerical operations in Samurai, enabling efficient and flexible mesh management for adaptive numerical simulations. 