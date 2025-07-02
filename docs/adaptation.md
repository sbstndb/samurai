# Mesh Adaptation

## Introduction

Mesh adaptation is a core feature of Samurai, enabling dynamic refinement and coarsening of the computational mesh based on solution characteristics. The adaptation system supports various strategies including multiresolution analysis, error indicators, and user-defined criteria, allowing for efficient computation by concentrating computational effort where it's most needed.

## Core Concepts

### 1. Adaptation Types

Samurai supports several adaptation strategies:

- **Multiresolution Adaptation**: Based on wavelet analysis
- **Error-based Adaptation**: Using error indicators
- **Gradient-based Adaptation**: Based on solution gradients
- **User-defined Adaptation**: Custom refinement criteria

### 2. Adaptation Process

The adaptation process typically involves:

1. **Analysis**: Evaluate the current solution
2. **Criterion**: Determine which cells need refinement/coarsening
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
    // 1. Predict solution on fine levels
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

## Error-based Adaptation

### 1. Error Indicators

Error-based adaptation uses various error indicators:

```cpp
// Gradient-based error indicator
auto error_indicator = samurai::make_gradient_indicator(field);

// Residual-based error indicator
auto residual_indicator = samurai::make_residual_indicator(field, scheme);

// User-defined error indicator
auto custom_indicator = [&](const auto& cell) {
    return std::abs(field[cell] - exact_solution(cell.center()));
};
```

### 2. Adaptation with Error Indicators

```cpp
// Create adaptation based on error indicator
auto adaptation = samurai::make_adaptation(field, error_indicator);

// Apply adaptation with threshold
adaptation(threshold, min_level, max_level);
```

## Adaptation Algorithms

### 1. Graduation Algorithm

The graduation algorithm ensures mesh consistency:

```cpp
// Apply graduation to ensure 2:1 balance
samurai::graduation(mesh);

// Graduation with specific parameters
samurai::graduation(mesh, min_level, max_level);
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

## Adaptation Criteria

### 1. Built-in Criteria

```cpp
// Multiresolution criterion
auto mr_criterion = samurai::make_MRCriterion(field, epsilon, regularity);

// Gradient criterion
auto gradient_criterion = samurai::make_gradient_criterion(field, threshold);

// Jump criterion (for discontinuous solutions)
auto jump_criterion = samurai::make_jump_criterion(field, threshold);
```

### 2. Custom Criteria

```cpp
// Define custom adaptation criterion
auto custom_criterion = [&](const auto& cell) {
    auto center = cell.center();
    auto value = field[cell];
    
    // Refine if solution is large
    if (std::abs(value) > threshold) {
        return true;
    }
    
    // Refine if near discontinuity
    if (is_near_discontinuity(center)) {
        return true;
    }
    
    return false;
};
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
// Create multiple adaptation criteria
auto mr_adapt = samurai::make_MRAdapt(u);
auto error_adapt = samurai::make_adaptation(u, error_indicator);

// Time loop with multiple adaptation strategies
for (std::size_t iter = 0; iter < max_iterations; ++iter) {
    // Update ghost cells
    samurai::update_ghost_mr(u);
    
    // Apply numerical scheme
    auto rhs = scheme(u);
    u = u + dt * rhs;
    
    // Apply multiresolution adaptation
    mr_adapt(epsilon, regularity);
    
    // Apply error-based adaptation
    error_adapt(error_threshold, min_level, max_level);
    
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
adaptation(epsilon, regularity, min_level, max_level);
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

### 2. Error-based Adaptation

```cpp
// Create error indicator
auto error_indicator = [&](const auto& cell) {
    auto center = cell.center();
    auto exact = exact_solution(center);
    return std::abs(u[cell] - exact);
};

// Create adaptation
auto adaptation = samurai::make_adaptation(u, error_indicator);

// Apply adaptation
adaptation(1e-3, min_level, max_level);
```

### 3. Custom Adaptation Criterion

```cpp
// Define custom criterion
auto custom_criterion = [&](const auto& cell) {
    auto center = cell.center();
    
    // Refine near origin
    double distance = std::sqrt(center[0]*center[0] + center[1]*center[1]);
    if (distance < 0.1) {
        return true;
    }
    
    // Refine if gradient is large
    auto gradient = compute_gradient(cell);
    if (xt::norm(gradient) > gradient_threshold) {
        return true;
    }
    
    return false;
};

// Create adaptation with custom criterion
auto adaptation = samurai::make_adaptation(u, custom_criterion);
adaptation(threshold, min_level, max_level);
```

### 4. Multi-criteria Adaptation

```cpp
// Create multiple adaptation strategies
auto mr_adapt = samurai::make_MRAdapt(u);
auto gradient_adapt = samurai::make_gradient_adaptation(u);
auto error_adapt = samurai::make_adaptation(u, error_indicator);

// Apply different adaptations
mr_adapt(1e-4, 1.0);
gradient_adapt(gradient_threshold);
error_adapt(error_threshold, min_level, max_level);

// Ensure mesh consistency
samurai::graduation(mesh);
```

### 5. Adaptive Threshold

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

### 1. Adaptation Statistics

```cpp
// Get adaptation statistics
auto stats = adaptation.get_statistics();
std::cout << "Refined cells: " << stats.refined_cells << std::endl;
std::cout << "Coarsened cells: " << stats.coarsened_cells << std::endl;
std::cout << "Total cells: " << stats.total_cells << std::endl;
```

### 2. Mesh Quality Monitoring

```cpp
// Monitor mesh quality
for (std::size_t level = mesh.min_level(); level <= mesh.max_level(); ++level) {
    std::cout << "Level " << level << ": " 
              << mesh.nb_cells(level) << " cells" << std::endl;
}
```

The mesh adaptation system in Samurai provides powerful and flexible tools for dynamic mesh management, enabling efficient numerical simulations by concentrating computational effort where it's most needed while maintaining solution accuracy. 