# Numerical Schemes - Samurai

## Overview

Samurai provides a comprehensive collection of numerical schemes for solving partial differential equations (PDEs) on adaptive meshes. These schemes are designed to work efficiently with the AMR system and support multi-physics simulations.

## Scheme Architecture

```mermaid
graph TB
    A[Numerical Schemes] --> B[Flux-Based Schemes]
    A --> C[Cell-Based Schemes]
    A --> D[Differential Operators]
    
    B --> E[Upwind Convection]
    B --> F[WENO5 Convection]
    B --> G[Diffusion]
    
    C --> H[Explicit Schemes]
    C --> I[Implicit Schemes]
    
    D --> J[Gradient]
    D --> K[Divergence]
    D --> L[Laplacian]
    
    subgraph "Scheme Types"
        M[Linear Homogeneous]
        N[Linear Heterogeneous]
        O[Non-Linear]
    end
```

## Flux-Based Schemes

### General Design

Flux-based schemes in Samurai are based on a conservative approach where fluxes are calculated at interfaces between cells.

```mermaid
graph LR
    A[Cell i-1] --> B[Flux i-1/2]
    B --> C[Cell i]
    C --> D[Flux i+1/2]
    D --> E[Cell i+1]
    
    subgraph "Flux Calculation"
        F[Variable Evaluation]
        G[Numerical Flux Calculation]
        H[Boundary Condition Application]
    end
```

### Flux-Based Scheme Configuration

```cpp
template <SchemeType scheme_type,
          std::size_t output_n_comp,
          std::size_t stencil_size,
          class Field>
struct FluxConfig
{
    static constexpr SchemeType type = scheme_type;
    static constexpr std::size_t output_n_comp = output_n_comp;
    static constexpr std::size_t stencil_size = stencil_size;
    using field_t = Field;
};
```

## Convection Schemes

### Linear Upwind Convection

The upwind scheme is the simplest and most robust convection scheme.

```cpp
template <class Field>
auto make_convection_upwind(const VelocityVector<Field::dim>& velocity)
```

**Upwind Scheme Principle:**

```mermaid
graph TD
    A[Velocity > 0] --> B[Use Left Value]
    A --> C[Velocity < 0] --> D[Use Right Value]
    
    B --> E[Flux = v * u_left]
    D --> F[Flux = v * u_right]
    
    subgraph "Stencil"
        G[Cell i-1] --> H[Cell i]
        H --> I[Cell i+1]
    end
```

**Scheme Coefficients:**

```cpp
// For v >= 0 (left upwind)
coeffs[left] = velocity(d);
coeffs[right] = 0;

// For v < 0 (right upwind)
coeffs[left] = 0;
coeffs[right] = velocity(d);
```

### WENO5 Convection

The WENO5 (Weighted Essentially Non-Oscillatory) scheme provides 5th order accuracy with oscillation limiting.

```cpp
template <class Field>
auto make_convection_weno5(const VelocityVector<Field::dim>& velocity)
```

**WENO5 Stencil Structure:**

```mermaid
graph LR
    A[i-2] --> B[i-1] --> C[i] --> D[i+1] --> E[i+2] --> F[i+3]
    
    subgraph "Local Stencils"
        G[Stencil 1: i-2, i-1, i]
        H[Stencil 2: i-1, i, i+1]
        I[Stencil 3: i, i+1, i+2]
    end
    
    subgraph "Reconstruction"
        J[Polynomial Calculation]
        K[Weight Calculation]
        L[Final Reconstruction]
    end
```

**WENO5 Algorithm:**

```mermaid
graph TD
    A[Input Data] --> B[Local Flux Calculation]
    B --> C[Smoothness Indicator Calculation]
    C --> D[Non-Oscillatory Weight Calculation]
    D --> E[WENO Reconstruction]
    E --> F[Final Flux]
    
    subgraph "Smoothness Indicators"
        G["β₀ = (uᵢ₊₁ - uᵢ)² + (uᵢ - uᵢ₋₁)²"]
        H["β₁ = (uᵢ₊₂ - uᵢ₊₁)² + (uᵢ₊₁ - uᵢ)²"]
        I["β₂ = (uᵢ₊₃ - uᵢ₊₂)² + (uᵢ₊₂ - uᵢ₊₁)²"]
    end
```

### Convection with Variable Velocity Field

```cpp
template <class Field, class VelocityField>
auto make_convection_upwind(const VelocityField& velocity_field)
```

**Variable Velocity Workflow:**

```mermaid
graph LR
    A[Velocity Field] --> B[Local Evaluation]
    B --> C[Direction Determination]
    C --> D[Upwind Scheme Application]
    D --> E[Resulting Flux]
    
    subgraph "Evaluation"
        F["Calculate v(x,t)"]
        G["Test v ≥ 0"]
        H[Stencil Selection]
    end
```

## Diffusion Schemes

### Linear Homogeneous Diffusion

```cpp
template <class Field, DirichletEnforcement dirichlet_enfcmt = Equation>
auto make_diffusion_order2(const DiffCoeff<Field::dim>& K)
```

**Diffusion Scheme Principle:**

```mermaid
graph TD
    A[Discrete Laplacian] --> B[Centered Finite Differences]
    B --> C[Diffusion Flux]
    C --> D["Operator -∇·(K∇u)"]
    
    subgraph "1D Stencil"
        E["uᵢ₋₁"] --> F["uᵢ"] --> G["uᵢ₊₁"]
        H["Flux i-1/2"] --> F
        F --> I["Flux i+1/2"]
    end
```

**Scheme Coefficients:**

```cpp
// Diffusion flux
coeffs[left] = -K(d) / h;
coeffs[right] = K(d) / h;

// Opérateur -Laplacien
coeffs[left] *= -1;
coeffs[right] *= -1;
```

### Multi-Component Diffusion

```cpp
template <class Field, DirichletEnforcement dirichlet_enfcmt = Equation>
auto make_multi_diffusion_order2(const DiffCoeff<Field::n_comp>& K)
```

**Multi-Component Structure:**

```mermaid
graph TB
    A[Multi-Component Field] --> B[Component 1]
    A --> C[Component 2]
    A --> D[Component n]
    
    B --> E[Diffusion K₁]
    C --> F[Diffusion K₂]
    D --> G[Diffusion Kₙ]
    
    E --> H[Resulting Field]
    F --> H
    G --> H
```

### Boundary Conditions for Diffusion

#### Dirichlet Conditions

```cpp
void set_dirichlet_config()
{
    // Equation: (u_ghost + u_cell)/2 = dirichlet_value
    // Coefficient: [1/2, 1/2] = dirichlet_value
    coeffs[cell] = -1/(h*h);
    coeffs[ghost] = -1/(h*h);
    rhs_coeffs = -2/(h*h) * dirichlet_value;
}
```

#### Neumann Conditions

```cpp
void set_neumann_config()
{
    // Equation: (u_ghost - u_cell)/h = neumann_value
    // Coefficient: [1/h², -1/h²] = (1/h) * neumann_value
    coeffs[cell] = -1/(h*h);
    coeffs[ghost] = 1/(h*h);
    rhs_coeffs = (1/h) * neumann_value;
}
```

## Cell-Based Schemes

### Explicit Schemes

```cpp
template <class cfg>
class ExplicitCellBasedScheme : public CellBasedScheme<cfg>
```

**Explicit Scheme Workflow:**

```mermaid
graph LR
    A[Current State] --> B[Flux Calculation]
    B --> C[Time Integration]
    C --> D[New State]
    
    subgraph "Integration"
        E[Explicit Euler Method]
        F[RK4 Method]
        G[Adams-Bashforth Method]
    end
```

### Implicit Schemes

```cpp
template <class cfg>
class ImplicitCellBasedScheme : public CellBasedScheme<cfg>
```

**Implicit Scheme Workflow:**

```mermaid
graph LR
    A[Current State] --> B[Matrix Assembly]
    B --> C[Linear System Resolution]
    C --> D[New State]
    
    subgraph "Resolution"
        E[Direct Solver]
        F[Iterative Solver]
        G[Preconditioning]
    end
```

## Differential Operators

### Gradient Operator

```cpp
template <class Field>
auto make_gradient()
```

**Gradient Calculation:**

```mermaid
graph TD
    A[Scalar Field] --> B["Calculate ∂u/∂x"]
    A --> C["Calculate ∂u/∂y"]
    A --> D["Calculate ∂u/∂z"]
    
    B --> E["Gradient ∇u"]
    C --> E
    D --> E
    
    subgraph "Finite Differences"
        F["∂u/∂x ≈ (uᵢ₊₁ - uᵢ₋₁)/(2h)"]
        G["∂u/∂y ≈ (uⱼ₊₁ - uⱼ₋₁)/(2h)"]
        H["∂u/∂z ≈ (uₖ₊₁ - uₖ₋₁)/(2h)"]
    end
```

### Divergence Operator

```cpp
template <class Field>
auto make_divergence()
```

**Divergence Calculation:**

```mermaid
graph TD
    A[Vector Field] --> B[X-Component]
    A --> C[Y-Component]
    A --> D[Z-Component]
    
    B --> E[∂vₓ/∂x]
    C --> F[∂vᵧ/∂y]
    D --> G[∂vᵤ/∂z]
    
    E --> H[Divergence ∇·v]
    F --> H
    G --> H
```

### Laplacian Operator

```cpp
template <class Field>
auto make_laplacian_order2()
{
    return make_diffusion_order2<Field>(1.0);
}
```

## Non-Linear Schemes

### Non-Linear Convection

```cpp
template <class Field>
auto make_convection_nonlinear()
```

**Non-Linearity Management:**

```mermaid
graph LR
    A[Non-Linear Field] --> B[Linearization]
    B --> C[Linear Scheme]
    C --> D[Non-Linear Correction]
    D --> E[Final Result]
    
    subgraph "Methods"
        F[Newton-Raphson]
        G[Fixed Point Iteration]
        H[Relaxation Scheme]
    end
```

## Time Integration

### Explicit Schemes

```mermaid
graph LR
    A[State tₙ] --> B[Flux Calculation]
    B --> C[Integration]
    C --> D[State tₙ₊₁]
    
    subgraph "Methods"
        E[Explicit Euler]
        F[RK2]
        G[RK4]
        H[Adams-Bashforth]
    end
```

### Implicit Schemes

```mermaid
graph LR
    A[State tₙ] --> B[System Assembly]
    B --> C[Resolution]
    C --> D[State tₙ₊₁]
    
    subgraph "Methods"
        E[Implicit Euler]
        F[Crank-Nicolson]
        G[BDF]
        H[Multi-Step Schemes]
    end
```

## Boundary Conditions

### Types of Boundary Conditions

```mermaid
graph TB
    A[Boundary Conditions] --> B[Dirichlet]
    A --> C[Neumann]
    A --> D[Periodic]
    A --> E[Robin]
    A --> F[Custom]
    
    B --> G["u = g on ∂Ω"]
    C --> H["∂u/∂n = h on ∂Ω"]
    D --> I["u(x) = u(x+L)"]
    E --> J["αu + β∂u/∂n = γ"]
    F --> K[Specific Conditions]
```

### Implementation of Boundary Conditions

```cpp
// Dirichlet Configuration
scheme.set_dirichlet_config();

// Neumann Configuration  
scheme.set_neumann_config();

// Periodic Configuration
scheme.set_periodic_config();
```

## Optimizations and Performance

### Compile-Time Optimizations

```cpp
// Use compile-time constants
static constexpr std::size_t stencil_size = 2;
static constexpr std::size_t output_n_comp = n_comp;

// Template specialization
template <std::size_t dim>
using VelocityVector = xt::xtensor_fixed<double, xt::xshape<dim>>;
```

### Runtime Optimizations

```mermaid
graph TB
    A[Flux Calculation] --> B{Optimizations}
    B --> C[SIMD Vectorization]
    B --> D[Cache Locality]
    B --> E[Parallelization]
    
    C --> F[Performance Improvement]
    D --> F
    E --> F
```

## Validation and Testing

### Convergence Tests

```cpp
// Test for a scheme
auto error = compute_convergence_error(scheme, exact_solution);
std::cout << "Convergence rate: " << error << std::endl;
```

### Conservation Tests

```mermaid
graph LR
    A[Initial State] --> B[Evolution]
    B --> C[Final State]
    
    A --> D[Calculate Conserved Quantity]
    C --> E[Calculate Conserved Quantity]
    
    D --> F{Conservation?}
    E --> F
    F -->|Yes| G[Test Passed]
    F -->|No| H[Test Failed]
```

## Complete Examples

### Example 1: Convection-Diffusion Equation

```cpp
#include <samurai/schemes/fv.hpp>

int main()
{
    // Mesh Configuration
    auto mesh = make_mesh();
    
    // Field Creation
    auto u = make_field<double, 1>("u", mesh);
    auto velocity = make_field<double, 2>("velocity", mesh);
    
    // Numerical Schemes
    auto convection = make_convection_upwind(velocity);
    auto diffusion = make_diffusion_order2(1.0);
    
    // Scheme Combination
    auto scheme = convection + diffusion;
    
    // Application
    scheme.apply(u);
    
    return 0;
}
```

### Example 2: Burgers Equation with WENO5

```cpp
#include <samurai/schemes/fv.hpp>

int main()
{
    // Configuration
    auto mesh = make_amr_mesh();
    auto u = make_field<double, 1>("u", mesh);
    
    // WENO5 Scheme for Burgers Equation
    auto burgers_scheme = make_convection_weno5(u);
    
    // Time Integration
    for (std::size_t step = 0; step < n_steps; ++step)
    {
        burgers_scheme.apply(u);
        update_time_step();
    }
    
    return 0;
}
```

### Example 3: Multi-Physics System

```cpp
#include <samurai/schemes/fv.hpp>

int main()
{
    // Multi-Component Fields
    auto rho = make_field<double, 1>("density", mesh);
    auto v = make_field<double, 2>("velocity", mesh);
    auto p = make_field<double, 1>("pressure", mesh);
    
    // Schemes for each equation
    auto mass_equation = make_convection_upwind(v);
    auto momentum_equation = make_convection_weno5(v) + make_diffusion_order2(mu);
    auto energy_equation = make_convection_upwind(v) + make_diffusion_order2(kappa);
    
    // Coupled System
    mass_equation.apply(rho);
    momentum_equation.apply(v);
    energy_equation.apply(p);
    
    return 0;
}
```

## Monitoring and Debugging

### Monitoring Schemes

```cpp
// Enable monitoring
scheme.set_monitoring(true);

// Display statistics
std::cout << "Scheme statistics:" << std::endl;
std::cout << "  - CFL number: " << scheme.get_cfl() << std::endl;
std::cout << "  - Max eigenvalue: " << scheme.get_max_eigenvalue() << std::endl;
std::cout << "  - Min eigenvalue: " << scheme.get_min_eigenvalue() << std::endl;
```

### Debugging Schemes

```cpp
// Validate coefficients
scheme.validate_coefficients();

// Check stability
if (!scheme.check_stability())
{
    std::cerr << "Warning: Scheme may be unstable!" << std::endl;
}
```

## Integration with AMR

### Adaptive Schemes

```mermaid
graph LR
    A[AMR Mesh] --> B[Level Identification]
    B --> C[Apply Schemes]
    C --> D[Projection/Prediction]
    D --> E[Update Solution]
    
    subgraph "Multi-Level Management"
        F[Schemes by Level]
        G[Synchronization]
        H[Conservation]
    end
```

### Schemes with Refinement

```cpp
// Apply on AMR mesh
for (std::size_t level = mesh.min_level(); level <= mesh.max_level(); ++level)
{
    auto level_scheme = make_scheme_for_level(level);
    level_scheme.apply(field);
}

// Synchronization between levels
synchronize_levels(field);
```

## Conclusion

The numerical schemes of Samurai offer a complete palette of tools for solving PDEs on adaptive meshes. They combine numerical precision, robustness, and efficiency, seamlessly integrating with the AMR system.

The modularity of the schemes allows for great flexibility in designing solvers for complex multi-physics problems. 