# Mesh Adaptation

## Introduction

Mesh adaptation is a core feature of Samurai, enabling dynamic refinement and coarsening of the computational mesh based on solution characteristics. The adaptation system uses multiresolution analysis to determine where refinement is needed, allowing for efficient computation by concentrating computational effort where it's most needed.

## Core Concepts

### 1. Adaptation Types

Samurai supports multiresolution adaptation:

- **Multiresolution Adaptation**: Based on wavelet analysis using the Harten algorithm

### 2. Adaptation Process

The adaptation process typically involves:

1. **Analysis**: Evaluate the current solution using wavelet coefficients
2. **Criterion**: Determine which cells need refinement/coarsening based on detail coefficients
3. **Modification**: Refine or coarsen cells
4. **Update**: Update mesh connectivity and ghost cells

## Multiresolution Adaptation

### 1. Multiresolution Analysis

Multiresolution adaptation uses wavelet analysis to determine where refinement is needed:

```cpp
// Create multiresolution adaptation
auto MRadaptation = samurai::make_MRAdapt(field);

// Apply adaptation
MRadaptation(epsilon, regularity);
```

### 2. Adaptation Parameters

```cpp
// Epsilon: threshold for adaptation
double epsilon = 1e-4;  // Smaller values = more refinement

// Regularity: regularity parameter for wavelet analysis
double regularity = 1.0;  // Higher values = smoother solutions
```

### 3. Multiresolution Process

```cpp
// Complete multiresolution adaptation workflow
for (std::size_t iter = 0; iter < max_iterations; ++iter) {
    // 1. Update ghost cells
    samurai::update_ghost_mr(field);
    
    // 2. Apply numerical scheme
    auto rhs = scheme(field);
    field = field + dt * rhs;
    
    // 3. Adapt mesh
    MRadaptation(epsilon, regularity);
    
    // 4. Update solution after adaptation
    samurai::update_ghost_mr(field);
}
```

## Adaptation Algorithms

### 1. Graduation Algorithm

The graduation algorithm ensures mesh consistency by maintaining the 2:1 balance condition:

```cpp
// Apply graduation to ensure 2:1 balance
samurai::graduation(mesh);

// Graduation with specific parameters
samurai::make_graduation(ca, grad_width);
```

### 2. Prediction and Projection

```cpp
// Predict solution on fine levels
samurai::prediction(field);

// Project solution to coarse levels
samurai::projection(field);
```

### 3. Ghost Cell Update

```cpp
// Update ghost cells after adaptation
samurai::update_ghost_mr(field);

// Update specific mesh regions
samurai::update_ghost(mesh[samurai::MRMeshID::ghosts]);
```

## Adaptation Workflow

### 1. Basic Adaptation Loop

```cpp
// Initialize adaptation
auto adaptation = samurai::make_MRAdapt(u);

// Time loop with adaptation
for (std::size_t iter = 0; iter < max_iterations; ++iter) {
    // Update ghost cells
    samurai::update_ghost_mr(u);
    
    // Apply numerical scheme
    auto rhs = scheme(u);
    u = u + dt * rhs;
    
    // Adapt mesh
    adaptation(epsilon, regularity);
    
    // Update solution after adaptation
    samurai::update_ghost_mr(u);
}
```

### 2. Advanced Adaptation Loop

```cpp
// Create multiresolution adaptation
auto mr_adapt = samurai::make_MRAdapt(u);

// Time loop with adaptation
for (std::size_t iter = 0; iter < max_iterations; ++iter) {
    // Update ghost cells
    samurai::update_ghost_mr(u);
    
    // Apply numerical scheme
    auto rhs = scheme(u);
    u = u + dt * rhs;
    
    // Apply multiresolution adaptation
    mr_adapt(epsilon, regularity);
    
    // Ensure mesh consistency
    samurai::graduation(mesh);
    
    // Update solution
    samurai::update_ghost_mr(u);
}
```

## Performance Considerations

### 1. Adaptation Frequency

```cpp
// Adapt every N iterations
if (iter % adaptation_frequency == 0) {
    adaptation(epsilon, regularity);
}
```

### 2. Level Restrictions

```cpp
// Limit adaptation to specific levels
// This is handled internally by the multiresolution algorithm
```

### 3. Threshold Tuning

```cpp
// Adaptive threshold based on solution
double adaptive_epsilon = base_epsilon * solution_norm;
adaptation(adaptive_epsilon, regularity);
```

## Examples

### 1. Basic Multiresolution Adaptation

```cpp
#include <samurai/mr/adapt.hpp>

int main() {
    // Create mesh and field
    samurai::MRMesh<Config> mesh(box, min_level, max_level);
    auto u = samurai::make_scalar_field<double>("u", mesh);
    
    // Initialize solution
    samurai::for_each_cell(mesh, [&](const auto& cell) {
        u[cell] = initial_condition(cell.center());
    });
    
    // Create adaptation
    auto adaptation = samurai::make_MRAdapt(u);
    
    // Time loop
    for (std::size_t iter = 0; iter < max_iterations; ++iter) {
        // Update ghost cells
        samurai::update_ghost_mr(u);
        
        // Apply scheme
        auto rhs = scheme(u);
        u = u + dt * rhs;
        
        // Adapt mesh
        adaptation(1e-4, 1.0);
    }
    
    return 0;
}
```

### 2. Custom Adaptation Criterion

```cpp
// Define custom criterion for initial mesh setup
auto custom_criterion = [&](const auto& cell) {
    auto center = cell.center();
    
    // Refine near origin
    double distance = std::sqrt(center[0]*center[0] + center[1]*center[1]);
    if (distance < 0.1) {
        return true;
    }
    
    return false;
};

// Apply custom criterion to create initial adaptive mesh
samurai::for_each_cell(mesh, [&](const auto& cell) {
    if (custom_criterion(cell)) {
        // Mark for refinement
    }
});
```

### 3. Multi-level Adaptation

```cpp
// Create multiresolution adaptation
auto mr_adapt = samurai::make_MRAdapt(u);

// Apply adaptation with different parameters
mr_adapt(1e-4, 1.0);

// Ensure mesh consistency
samurai::graduation(mesh);
```

### 4. Adaptive Threshold

```cpp
// Compute adaptive threshold
double compute_adaptive_threshold(const auto& field) {
    double max_val = 0.0;
    samurai::for_each_cell(field.mesh(), [&](const auto& cell) {
        max_val = std::max(max_val, std::abs(field[cell]));
    });
    return base_threshold * max_val;
}

// Use adaptive threshold
double threshold = compute_adaptive_threshold(u);
adaptation(threshold, regularity);
```

## Monitoring and Statistics

### 1. Mesh Quality Monitoring

```cpp
// Monitor mesh quality
for (std::size_t level = mesh.min_level(); level <= mesh.max_level(); ++level) {
    std::cout << "Level " << level << ": " 
              << mesh.nb_cells(level) << " cells" << std::endl;
}
```

### 2. Adaptation Monitoring

```cpp
// Monitor adaptation process
std::cout << "Mesh levels: " << mesh.min_level() << " to " << mesh.max_level() << std::endl;
std::cout << "Total cells: " << mesh.nb_cells() << std::endl;
```

The mesh adaptation system in Samurai provides powerful tools for dynamic mesh management using multiresolution analysis, enabling efficient numerical simulations by concentrating computational effort where it's most needed while maintaining solution accuracy. 