# Test Case: Linear Convection - Samurai

## Overview

The linear convection test case in Samurai is a fundamental benchmark for validating numerical convection schemes on adaptive meshes. It solves the linear convection equation with periodic boundary conditions and uses the WENO5 scheme for high-order accuracy.

## Modeled Equation

### Mathematical Formulation

The linear convection equation in dimension `d` is written as:

```
∂u/∂t + ∇·(v u) = 0
```

where:
- `u(x,t)` is the scalar state variable
- `v = (v₁, v₂, ..., v_d)` is the constant velocity vector
- `∇·` is the divergence operator

### Expanded Form

**In 1D:**
```
∂u/∂t + v₁ ∂u/∂x = 0
```

**In 2D:**
```
∂u/∂t + v₁ ∂u/∂x + v₂ ∂u/∂y = 0
```

**In 3D:**
```
∂u/∂t + v₁ ∂u/∂x + v₂ ∂u/∂y + v₃ ∂u/∂z = 0
```

## Problem Configuration

### Simulation Parameters

```cpp
// Simulation parameters
double left_box = -1;      // Left domain boundary
double right_box = 1;      // Right domain boundary
double Tf = 3;             // Final time
double dt = 0;             // Time step (automatically calculated if 0)
double cfl = 0.95;         // Courant-Friedrichs-Lewy number
double t = 0.;             // Initial time
std::string restart_file;  // Restart file (optional)
```

### Mesh Configuration

```cpp
// Multiresolution parameters
std::size_t min_level = 1;           // Minimum refinement level
std::size_t max_level = dim == 1 ? 6 : 4;  // Maximum level
double mr_epsilon = 1e-4;            // Multiresolution adaptation threshold
double mr_regularity = 1.;           // Estimated regularity for adaptation

// Domain configuration
point_t box_corner1, box_corner2;
box_corner1.fill(left_box);
box_corner2.fill(right_box);
Box box(box_corner1, box_corner2);

// Periodic boundary conditions
std::array<bool, dim> periodic;
periodic.fill(true);
```

### Output Configuration

```cpp
// Output parameters
fs::path path = fs::current_path();
std::string filename = "linear_convection_" + std::to_string(dim) + "D";
std::size_t nfiles = 0;  // Number of output files (0 = automatic)
```

## Initial Condition

### Mathematical Definition

The initial condition is a step function (characteristic function):

**In 1D:**
```
u₀(x) = 1  if x ∈ [-0.8, -0.3]
u₀(x) = 0  otherwise
```

**In 2D:**
```
u₀(x,y) = 1  if x ∈ [-0.8, -0.3] and y ∈ [0.3, 0.8]
u₀(x,y) = 0  otherwise
```

### Implementation

```cpp
// Create initial field
u = samurai::make_scalar_field<double>("u", mesh,
    [](const auto& coords)
    {
        if constexpr (dim == 1)
        {
            const auto& x = coords(0);
            return (x >= -0.8 && x <= -0.3) ? 1. : 0.;
        }
        else
        {
            const auto& x = coords(0);
            const auto& y = coords(1);
            return (x >= -0.8 && x <= -0.3 && 
                    y >= 0.3 && y <= 0.8) ? 1. : 0.;
        }
    });
```

### Initial Condition Visualization

```mermaid
graph LR
    A[Initial Condition] --> B[1D: Step Function]
    A --> C[2D: Rectangle]
    
    B --> D[Interval [-0.8, -0.3]]
    B --> E[Value 1.0]
    
    C --> F[Rectangle [-0.8, -0.3] × [0.3, 0.8]]
    C --> G[Value 1.0 inside]
    C --> H[Value 0.0 outside]
```

## Velocity Vector Configuration

### Velocity Vector Definition

```cpp
// Convection operator
samurai::VelocityVector<dim> velocity;
velocity.fill(1);  // Unit velocity in all directions

if constexpr (dim == 2)
{
    velocity(1) = -1;  // Diagonal velocity in 2D
}
```

### Physical Interpretation

**In 1D:**
- `v = (1)` : Convection to the right at unit velocity

**In 2D:**
- `v = (1, -1)` : Diagonal convection (top-right to bottom-left)

**In 3D:**
- `v = (1, 1, 1)` : Diagonal convection in all directions

## WENO5 Numerical Scheme

### WENO5 Scheme Principle

The WENO5 (Weighted Essentially Non-Oscillatory) scheme is a 5th-order scheme that combines multiple stencils to obtain a non-oscillatory reconstruction.

```cpp
// Create WENO5 convection operator
auto conv = samurai::make_convection_weno5<decltype(u)>(velocity);
```

### WENO5 Algorithm

```mermaid
graph TD
    A[Input Data] --> B[Local Flux Calculation]
    B --> C[Smoothness Indicator Calculation]
    C --> D[Non-Oscillatory Weight Calculation]
    D --> E[WENO Reconstruction]
    E --> F[Final Flux]
    
    subgraph "Smoothness Indicators"
        G[β₀ = (uᵢ₊₁ - uᵢ)² + (uᵢ - uᵢ₋₁)²]
        H[β₁ = (uᵢ₊₂ - uᵢ₊₁)² + (uᵢ₊₁ - uᵢ)²]
        I[β₂ = (uᵢ₊₃ - uᵢ₊₂)² + (uᵢ₊₂ - uᵢ₊₁)²]
    end
    
    subgraph "WENO Weights"
        J[ωᵢ = αᵢ / Σαⱼ]
        K[αᵢ = Cᵢ / (ε + βᵢ)²]
    end
```

### WENO5 Scheme Advantages

- **5th order accuracy** in space
- **Oscillation limiting** at discontinuities 