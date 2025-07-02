# Numerical Schemes

## Introduction

Samurai provides a comprehensive framework for implementing numerical schemes, particularly focused on finite volume methods. The scheme system is designed to be flexible, efficient, and easy to use, supporting both explicit and implicit time integration, various spatial discretizations, and different boundary condition treatments.

## Core Concepts

### 1. Scheme Types

Samurai supports two main types of finite volume schemes:

- **Cell-based Schemes**: Operate directly on cell values
- **Flux-based Schemes**: Compute fluxes at cell interfaces

### 2. Scheme Configuration

Schemes are configured using template parameters and configuration structures:

```cpp
template <class DerivedScheme, class cfg_, class bdry_cfg_>
class FVScheme
{
    using cfg = cfg_;
    using bdry_cfg = bdry_cfg_;
    // ...
};
```

## Finite Volume Schemes

### 1. Base Scheme Class

All finite volume schemes inherit from `FVScheme`:

```cpp
template <class DerivedScheme, class cfg_, class bdry_cfg_>
class FVScheme
{
public:
    using input_field_t = typename cfg_::input_field_t;
    using field_t = input_field_t;
    using mesh_t = typename field_t::mesh_t;
    using field_value_type = typename field_t::value_type;
    
    // Scheme application
    auto operator()(input_field_t& input_field);
    void apply(output_field_t& output_field, input_field_t& input_field);
    
    // Directional application
    auto operator()(std::size_t d, input_field_t& input_field);
    void apply(std::size_t d, output_field_t& output_field, input_field_t& input_field);
};
```

### 2. Cell-based Schemes

Cell-based schemes operate directly on cell values:

```cpp
template <class cfg>
class CellBasedScheme : public FVScheme<CellBasedScheme<cfg>, cfg, cfg::bdry_cfg>
{
public:
    // Cell-based stencil computation
    template <class Field>
    auto make_cell_based_stencil(Field& field);
};
```

### 3. Flux-based Schemes

Flux-based schemes compute fluxes at cell interfaces:

```cpp
template <class cfg>
class FluxBasedScheme : public FVScheme<FluxBasedScheme<cfg>, cfg, cfg::bdry_cfg>
{
public:
    // Flux computation
    template <class Field>
    auto make_flux_based_stencil(Field& field);
};
```

## Built-in Schemes

### 1. Diffusion Schemes

```cpp
// Second-order diffusion scheme
template <class Field, std::size_t dim>
auto make_diffusion_order2(Field& field, const DiffCoeff<dim>& K);

// Usage
samurai::DiffCoeff<dim> K;
K.fill(diffusion_coefficient);
auto diffusion = samurai::make_diffusion_order2(u, K);
```

### 2. Convection Schemes

```cpp
// Upwind scheme for convection
template <class Field, class Velocity>
auto make_convection_upwind(Field& field, const Velocity& velocity);

// Central difference scheme
template <class Field, class Velocity>
auto make_convection_central(Field& field, const Velocity& velocity);
```

### 3. Advection Schemes

```cpp
// Linear advection scheme
template <class Field>
auto make_advection_upwind(Field& field, double velocity);

// Usage
auto advection = samurai::make_advection_upwind(u, 1.0);
```

## Scheme Configuration

### 1. Scheme Configuration Structure

```cpp
template <class input_field_t_, std::size_t output_n_comp_ = 1>
struct SchemeConfig
{
    using input_field_t = input_field_t_;
    static constexpr std::size_t output_n_comp = output_n_comp_;
    using bdry_cfg = BoundaryConfigFV<1, Equation>;
};
```

### 2. Boundary Configuration

```cpp
template <std::size_t neighbourhood_width_ = 1, DirichletEnforcement dirichlet_enfcmt_ = Equation>
struct BoundaryConfigFV
{
    static constexpr std::size_t neighbourhood_width = neighbourhood_width_;
    static constexpr std::size_t stencil_size = 1 + 2 * neighbourhood_width;
    static constexpr std::size_t nb_ghosts = neighbourhood_width;
    static constexpr DirichletEnforcement dirichlet_enfcmt = dirichlet_enfcmt_;
};
```

## Boundary Conditions

### 1. Boundary Condition Types

```cpp
// Dirichlet boundary condition
template <std::size_t neighbourhood_width>
class Dirichlet;

// Neumann boundary condition
template <std::size_t neighbourhood_width>
class Neumann;

// Periodic boundary condition
template <std::size_t neighbourhood_width>
class Periodic;
```

### 2. Boundary Condition Application

```cpp
// Create boundary condition
auto bc = samurai::make_bc<samurai::Dirichlet<1>>(field, 0.0);

// Attach to field
field.attach_bc(bc);

// Apply boundary conditions
samurai::update_bc(field);
```

## Scheme Application

### 1. Explicit Application

```cpp
// Create scheme
auto scheme = samurai::make_diffusion_order2(u, K);

// Apply scheme explicitly
auto result = scheme(u);

// Or apply to existing field
scheme.apply(output_field, u);
```

### 2. Directional Application

```cpp
// Apply scheme in specific direction
auto result_x = scheme(0, u);  // x-direction
auto result_y = scheme(1, u);  // y-direction
```

### 3. Time Integration

```cpp
// Explicit Euler
for (std::size_t iter = 0; iter < max_iterations; ++iter) {
    auto rhs = scheme(u);
    u = u + dt * rhs;
}

// Implicit Euler (with PETSc)
auto implicit_scheme = samurai::make_implicit(scheme);
implicit_scheme.apply(u, dt);
```

## Stencil Operations

### 1. Stencil Definition

```cpp
template <std::size_t size, std::size_t dim>
class Stencil
{
    std::array<std::array<int, dim>, size> m_offsets;
    std::array<double, size> m_weights;
};
```

### 2. Stencil Application

```cpp
// Apply stencil to field
samurai::for_each_interval(mesh, [&](std::size_t level, const auto& interval, const auto& index) {
    auto j = index[0];
    for (auto i = interval.start; i < interval.end; i += interval.step) {
        double result = 0.0;
        for (std::size_t k = 0; k < stencil.size(); ++k) {
            auto offset = stencil.offset(k);
            result += stencil.weight(k) * field(level, i + offset[0], j + offset[1]);
        }
        output(level, i, j) = result;
    }
});
```

## PETSc Integration

### 1. Matrix Assembly

```cpp
// Create PETSc matrix
auto matrix = samurai::make_matrix<decltype(scheme)>(scheme);

// Assemble matrix
matrix.assemble();

// Solve linear system
auto solver = samurai::make_solver(matrix);
solver.solve(solution, rhs);
```

### 2. Nonlinear Solvers

```cpp
// Create nonlinear scheme
auto nonlinear_scheme = samurai::make_nonlinear_scheme(scheme);

// Solve nonlinear system
auto solver = samurai::make_nonlinear_solver(nonlinear_scheme);
solver.solve(u);
```

## Custom Scheme Implementation

### 1. Basic Custom Scheme

```cpp
template <class Field>
class CustomScheme
{
public:
    using input_field_t = Field;
    using field_t = input_field_t;
    using mesh_t = typename field_t::mesh_t;
    using field_value_type = typename field_t::value_type;
    
    static constexpr std::size_t output_n_comp = 1;
    
    template <class Config>
    auto make_stencil(Field& field)
    {
        return [&](auto& stencil)
        {
            // Define stencil coefficients
            stencil(0, 0) = -4.0;  // center
            stencil(-1, 0) = 1.0;  // left
            stencil(1, 0) = 1.0;   // right
            stencil(0, -1) = 1.0;  // bottom
            stencil(0, 1) = 1.0;   // top
        };
    }
};
```

### 2. Scheme with Configuration

```cpp
template <class Field>
struct CustomSchemeConfig
{
    using input_field_t = Field;
    static constexpr std::size_t output_n_comp = 1;
    using bdry_cfg = samurai::BoundaryConfigFV<1, samurai::Equation>;
    
    double coefficient = 1.0;
};

template <class Field>
class CustomScheme : public samurai::FVScheme<CustomScheme<Field>, CustomSchemeConfig<Field>, CustomSchemeConfig<Field>::bdry_cfg>
{
public:
    using config = CustomSchemeConfig<Field>;
    
    CustomScheme(double coeff = 1.0) : m_coeff(coeff) {}
    
    template <class Stencil>
    void make_stencil(Stencil& stencil)
    {
        stencil(0, 0) = -4.0 * m_coeff;
        stencil(-1, 0) = m_coeff;
        stencil(1, 0) = m_coeff;
        stencil(0, -1) = m_coeff;
        stencil(0, 1) = m_coeff;
    }
    
private:
    double m_coeff;
};
```

## Examples

### 1. Heat Equation

```cpp
#include <samurai/schemes/fv.hpp>

int main() {
    // Create mesh and fields
    samurai::MRMesh<Config> mesh(box, min_level, max_level);
    auto u = samurai::make_scalar_field<double>("u", mesh);
    auto unp1 = samurai::make_scalar_field<double>("unp1", mesh);
    
    // Set boundary conditions
    samurai::make_bc<samurai::Dirichlet<1>>(u, 0.0);
    samurai::make_bc<samurai::Dirichlet<1>>(unp1, 0.0);
    
    // Create diffusion scheme
    samurai::DiffCoeff<dim> K;
    K.fill(diffusion_coefficient);
    auto diffusion = samurai::make_diffusion_order2<decltype(u)>(K);
    
    // Time loop
    for (std::size_t iter = 0; iter < max_iterations; ++iter) {
        // Apply diffusion scheme
        auto rhs = diffusion(u);
        
        // Update solution
        unp1 = u + dt * rhs;
        
        // Swap fields
        std::swap(unp1.array(), u.array());
    }
    
    return 0;
}
```

### 2. Advection Equation

```cpp
// Create advection scheme
auto advection = samurai::make_advection_upwind(u, velocity);

// Apply scheme
samurai::for_each_interval(mesh, [&](std::size_t level, const auto& interval, const auto& index) {
    auto j = index[0];
    for (auto i = interval.start; i < interval.end; i += interval.step) {
        double dx = mesh.cell_length(level);
        unp1(level, i, j) = u(level, i, j) - 
                           dt / dx * (u(level, i, j) - u(level, i - 1, j));
    }
});
```

### 3. Implicit Scheme with PETSc

```cpp
// Create implicit scheme
auto implicit_diffusion = samurai::make_implicit(diffusion);

// Create PETSc matrix
auto matrix = samurai::make_matrix<decltype(implicit_diffusion)>(implicit_diffusion);

// Assemble matrix
matrix.assemble();

// Create solver
auto solver = samurai::make_solver(matrix);

// Solve system
solver.solve(unp1, u);
```

### 4. Custom Scheme

```cpp
// Define custom scheme
template <class Field>
class MyScheme : public samurai::FVScheme<MyScheme<Field>, MyConfig<Field>>
{
public:
    template <class Stencil>
    void make_stencil(Stencil& stencil)
    {
        // 5-point stencil
        stencil(0, 0) = -4.0;
        stencil(-1, 0) = 1.0;
        stencil(1, 0) = 1.0;
        stencil(0, -1) = 1.0;
        stencil(0, 1) = 1.0;
    }
};

// Use custom scheme
auto my_scheme = MyScheme<decltype(u)>{};
auto result = my_scheme(u);
```

The numerical scheme system in Samurai provides a flexible and efficient framework for implementing various finite volume methods, with support for both explicit and implicit time integration, complex boundary conditions, and integration with external linear algebra libraries like PETSc. 