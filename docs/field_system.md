# Field System

## Introduction

The field system in Samurai provides a flexible and efficient way to represent solution variables on the mesh. Fields are the primary data structures for storing and manipulating numerical solutions, supporting both scalar and vector quantities with various memory layouts and mathematical operations.

## Core Concepts

### 1. Field Types

Samurai provides two main field types:

- **ScalarField**: Single-component fields (e.g., temperature, pressure)
- **VectorField**: Multi-component fields (e.g., velocity, momentum)

### 2. Memory Layouts

Fields support different memory layouts for optimization:

- **AOS (Array of Structures)**: Components stored together for each cell
- **SOA (Structure of Arrays)**: Components stored separately across cells

## Field Classes

### 1. ScalarField

```cpp
template <class mesh_t, class value_t = double>
class ScalarField : public field_expression<ScalarField<mesh_t, value_t>>,
                    public inner_mesh_type<mesh_t>,
                    public detail::inner_field_types<ScalarField<mesh_t, value_t>>
{
public:
    using value_type = value_t;
    using mesh_type = mesh_t;
    static constexpr std::size_t dim = mesh_t::dim;
    static constexpr std::size_t n_comp = 1;
    static constexpr bool is_scalar = true;
    
    // Constructors
    ScalarField() = default;
    ScalarField(std::string name, mesh_t& mesh);
    
    // Access operators
    value_t& operator[](const cell_t& cell);
    const value_t& operator[](const cell_t& cell) const;
    
    // Interval access
    template <class... T>
    auto operator()(std::size_t level, const interval_t& interval, T... index);
};
```

### 2. VectorField

```cpp
template <class mesh_t, class value_t, std::size_t n_comp_, bool SOA = false>
class VectorField : public field_expression<VectorField<mesh_t, value_t, n_comp_, SOA>>,
                    public inner_mesh_type<mesh_t>,
                    public detail::inner_field_types<VectorField<mesh_t, value_t, n_comp_, SOA>>
{
public:
    using value_type = value_t;
    using mesh_type = mesh_t;
    static constexpr std::size_t dim = mesh_t::dim;
    static constexpr std::size_t n_comp = n_comp_;
    static constexpr bool is_soa = SOA;
    static constexpr bool is_scalar = false;
    
    // Access operators
    auto operator[](const cell_t& cell);
    auto operator[](const cell_t& cell) const;
    
    // Component access
    template <class... T>
    auto operator()(std::size_t item, std::size_t level, const interval_t& interval, T... index);
};
```

## Field Creation

### 1. Factory Functions

Samurai provides factory functions for creating fields:

```cpp
// Create scalar field
auto temperature = samurai::make_scalar_field<double>("T", mesh);

// Create vector field (AOS layout)
auto velocity = samurai::make_vector_field<double, 2>("u", mesh);

// Create vector field (SOA layout)
auto velocity_soa = samurai::make_vector_field<double, 2, true>("u", mesh);

// Create field with initial value
auto pressure = samurai::make_scalar_field<double>("p", mesh, 1.0);
```

### 2. Field Initialization

```cpp
// Initialize field with function
auto field = samurai::make_scalar_field<double>("f", mesh, [](const auto& coords) {
    return std::sin(coords[0]) * std::cos(coords[1]);
});

// Initialize with Gauss-Legendre quadrature
auto gl = samurai::GaussLegendre<2>{};
auto field = samurai::make_scalar_field<double>("f", mesh, initial_function, gl);
```

## Field Access

### 1. Cell-based Access

```cpp
// Access field value at cell
samurai::for_each_cell(mesh, [&](const auto& cell) {
    double value = temperature[cell];
    temperature[cell] = new_value;
});
```

### 2. Interval-based Access

```cpp
// Access field over intervals
samurai::for_each_interval(mesh, [&](std::size_t level, const auto& interval, const auto& index) {
    auto field_view = temperature(level, interval, index);
    
    // field_view is an xtensor view
    for (std::size_t i = 0; i < field_view.size(); ++i) {
        field_view[i] = compute_value(level, interval.start + i, index);
    }
});
```

### 3. Vector Field Component Access

```cpp
// Access specific component of vector field
samurai::for_each_cell(mesh, [&](const auto& cell) {
    auto u_component = velocity[cell][0];  // x-component
    auto v_component = velocity[cell][1];  // y-component
});
```

## Field Expressions

### 1. Mathematical Operations

Fields support mathematical operations through expression templates:

```cpp
// Arithmetic operations
auto result = field1 + field2;
auto scaled = field * 2.0;
auto diff = field1 - field2;
auto ratio = field1 / field2;

// Compound operations
auto result = field1 + field2 * 3.0 - field3 / 2.0;
```

### 2. Expression Templates

Field operations use expression templates to avoid unnecessary temporaries:

```cpp
// This creates an expression, not a temporary field
auto result = field1 + field2 * 2.0;

// The actual computation happens when the expression is assigned
field3 = result;
```

## Boundary Conditions

### 1. Boundary Condition Types

```cpp
// Dirichlet boundary condition
samurai::make_bc<samurai::Dirichlet<1>>(field, 0.0);

// Neumann boundary condition
samurai::make_bc<samurai::Neumann<1>>(field, 1.0);
```

### 2. Boundary Condition Management

```cpp
// Attach boundary condition
auto bc = samurai::make_bc<samurai::Dirichlet<1>>(field, 0.0);
field.attach_bc(bc);

// Copy boundary conditions from another field
field.copy_bc_from(other_field);
```

## Field Iterators

### 1. Iterator Types

```cpp
// Forward iterators
auto it = field.begin();
auto const_it = field.cbegin();

// Reverse iterators
auto rit = field.rbegin();
auto const_rit = field.rcbegin();
```

### 2. Iterator Usage

```cpp
// Iterate over field values
for (auto it = field.begin(); it != field.end(); ++it) {
    auto cell = it.cell();
    auto value = *it;
    // Process value
}

// Range-based for loop
for (auto& value : field) {
    // Process value
}
```

## Memory Management

### 1. Storage Types

Fields use different storage types based on their configuration:

```cpp
// Scalar field storage
using data_type = field_data_storage_t<value_t, 1>;

// Vector field storage (AOS)
using data_type = field_data_storage_t<value_t, n_comp, false>;

// Vector field storage (SOA)
using data_type = field_data_storage_t<value_t, n_comp, true>;
```

### 2. Resizing

```cpp
// Resize field to match mesh
field.resize();

// Check if resize is needed
if (field.size() != mesh.nb_cells()) {
    field.resize();
}
```

## Performance Optimizations

### 1. Expression Templates

Field operations use expression templates to avoid unnecessary temporaries:

```cpp
// This creates an expression, not a temporary field
auto result = field1 + field2 * 2.0;

// The actual computation happens when the expression is assigned
field3 = result;
```

### 2. View-based Access

Interval access returns views rather than copies:

```cpp
// Efficient view-based access
auto view = field(level, interval, index);
// view is a reference to the underlying data
```

### 3. Contiguous Memory Layout

Fields use contiguous memory layouts for cache efficiency:

```cpp
// Memory layout for AOS
[cell0_comp0, cell0_comp1, cell1_comp0, cell1_comp1, ...]

// Memory layout for SOA
[cell0_comp0, cell1_comp0, cell2_comp0, ..., cell0_comp1, cell1_comp1, ...]
```

## Examples

### 1. Basic Field Usage

```cpp
#include <samurai/field.hpp>

int main() {
    // Create mesh
    samurai::MRMesh<Config> mesh(box, min_level, max_level);
    
    // Create fields
    auto temperature = samurai::make_scalar_field<double>("T", mesh);
    auto velocity = samurai::make_vector_field<double, 2>("u", mesh);
    
    // Initialize fields
    samurai::for_each_cell(mesh, [&](const auto& cell) {
        auto center = cell.center();
        temperature[cell] = std::sin(center[0]) * std::cos(center[1]);
        velocity[cell] = {center[0], center[1]};
    });
    
    // Set boundary conditions
    samurai::make_bc<samurai::Dirichlet<1>>(temperature, 0.0);
    samurai::make_bc<samurai::Neumann<1>>(velocity, 0.0);
    
    return 0;
}
```

### 2. Field Operations

```cpp
// Create fields
auto field1 = samurai::make_scalar_field<double>("f1", mesh, 1.0);
auto field2 = samurai::make_scalar_field<double>("f2", mesh, 2.0);

// Perform operations
auto sum = field1 + field2;
auto product = field1 * field2;
auto scaled = field1 * 3.0;

// Apply to result field
auto result = samurai::make_scalar_field<double>("result", mesh);
result = sum + product * scaled;
```

### 3. Vector Field Operations

```cpp
// Create vector field
auto velocity = samurai::make_vector_field<double, 2>("u", mesh);

// Initialize components
samurai::for_each_cell(mesh, [&](const auto& cell) {
    auto center = cell.center();
    velocity[cell] = {std::sin(center[0]), std::cos(center[1])};
});

// Access components
samurai::for_each_cell(mesh, [&](const auto& cell) {
    auto u_comp = velocity[cell][0];
    auto v_comp = velocity[cell][1];
    
    // Process components
    double magnitude = std::sqrt(u_comp * u_comp + v_comp * v_comp);
    // Use magnitude
});
```

### 4. Field with Boundary Conditions

```cpp
// Create field with boundary conditions
auto field = samurai::make_scalar_field<double>("phi", mesh);

// Set different boundary conditions on different sides
samurai::make_bc<samurai::Dirichlet<1>>(field, 1.0);  // Left boundary
samurai::make_bc<samurai::Dirichlet<1>>(field, 0.0);  // Right boundary
samurai::make_bc<samurai::Neumann<1>>(field, 0.0);    // Top and bottom

// Initialize interior
samurai::for_each_cell(mesh, [&](const auto& cell) {
    if (!cell.is_ghost()) {
        field[cell] = initial_condition(cell.center());
    }
});
```

The field system provides a powerful and flexible interface for managing solution variables in Samurai, with support for efficient mathematical operations, boundary conditions, and various memory layouts optimized for different use cases. 