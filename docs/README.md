# Samurai Documentation

## Overview

Samurai is a C++ library for numerical simulations with adaptive mesh refinement (AMR). This documentation provides comprehensive information about the library's architecture, components, and usage.

## Table of Contents

### 1. [Overview](overview.md)
- Introduction to Samurai
- Core philosophy and design principles
- Architecture overview
- Key components
- Design patterns
- Performance considerations
- Usage patterns
- Integration with external libraries

### 2. [Mesh System](mesh_system.md)
- Core concepts (Box, Intervals, Cell Arrays)
- Mesh types (MRMesh, Mesh IDs)
- Mesh operations and cell access
- Cell representation and properties
- Mesh iteration patterns
- Mesh adaptation process
- Memory layout and performance optimizations
- MPI support for parallel computing

### 3. [Field System](field_system.md)
- Field types (ScalarField, VectorField)
- Memory layouts (AOS/SOA)
- Field creation and initialization
- Field access patterns (cell-based, interval-based)
- Field expressions and mathematical operations
- Boundary conditions
- Field iterators
- Memory management and performance optimizations
- Field I/O and restart capabilities

### 4. [Subset System](subset_system.md)
- Core concepts and subset operations
- Basic operations (Self, Intersection, Difference, Union)
- Advanced operations (Translation, Level specification)
- Subset application patterns
- Common use cases (multiresolution, ghost cells, boundary conditions)
- Performance considerations
- Examples and practical usage

### 5. [Numerical Schemes](numerical_schemes.md)
- Scheme types (Cell-based, Flux-based)
- Scheme configuration and boundary conditions
- Built-in schemes (Diffusion, Convection, Advection)
- Scheme application patterns
- Stencil operations
- PETSc integration for linear algebra
- Custom scheme implementation
- Examples and practical applications

### 6. [Mesh Adaptation](adaptation.md)
- Adaptation types and strategies
- Multiresolution adaptation
- Error-based adaptation
- Adaptation algorithms (Graduation, Prediction, Projection)
- Adaptation criteria and workflow
- Performance considerations
- Monitoring and statistics
- Examples and practical usage

## Quick Start Guide

### Basic Workflow

1. **Initialize Samurai**
   ```cpp
   auto& app = samurai::initialize("My Simulation", argc, argv);
   ```

2. **Create Mesh**
   ```cpp
   constexpr size_t dim = 2;
   using Config = samurai::MRConfig<dim>;
   samurai::Box<double, dim> box({0., 0.}, {1., 1.});
   samurai::MRMesh<Config> mesh(box, min_level, max_level);
   ```

3. **Create Fields**
   ```cpp
   auto u = samurai::make_scalar_field<double>("u", mesh);
   samurai::make_bc<samurai::Dirichlet<1>>(u, 0.);
   ```

4. **Initialize Solution**
   ```cpp
   samurai::for_each_cell(mesh, [&](const auto& cell) {
       u[cell] = initial_condition(cell.center());
   });
   ```

5. **Create Numerical Scheme**
   ```cpp
   samurai::DiffCoeff<dim> K;
   K.fill(diffusion_coefficient);
   auto diffusion = samurai::make_diffusion_order2<decltype(u)>(K);
   ```

6. **Time Loop with Adaptation**
   ```cpp
   auto adaptation = samurai::make_MRAdapt(u);
   
   for (std::size_t iter = 0; iter < max_iterations; ++iter) {
       samurai::update_ghost_mr(u);
       auto rhs = diffusion(u);
       u = u + dt * rhs;
       adaptation(epsilon, regularity);
   }
   ```

### Key Features

- **Unified Mesh Representation**: Single data structure for various adaptation strategies
- **Set Algebra**: Efficient operations on mesh subsets using intervals
- **Template-based Design**: Compile-time optimization and type safety
- **Flexible Interface**: Easy implementation of numerical schemes
- **MPI Support**: Parallel computing capabilities
- **PETSc Integration**: Advanced linear algebra and solvers
- **HDF5 I/O**: Data input/output and restart capabilities

## Examples

The documentation includes numerous examples demonstrating:

- Basic mesh and field operations
- Numerical scheme implementation
- Mesh adaptation strategies
- Boundary condition handling
- Performance optimization techniques
- Parallel computing with MPI
- Integration with external libraries

## Performance Considerations

Samurai is designed for high performance with:

- **Contiguous Memory Layouts**: Cache-efficient data storage
- **Expression Templates**: Avoid unnecessary temporary objects
- **Interval-based Operations**: Efficient processing of large cell ranges
- **Compile-time Optimization**: Template metaprogramming for static computations
- **Minimal Memory Allocation**: Optimized memory management during computation

## Integration

Samurai integrates with several external libraries:

- **PETSc**: Linear algebra and solvers
- **HDF5**: Data I/O and restart capabilities
- **MPI**: Parallel computing support
- **xtensor**: Multi-dimensional array operations
- **fmt**: String formatting
- **CLI11**: Command-line argument parsing

## Getting Help

For additional help and examples:

- Browse the [demos](../demos/) directory for complete examples
- Check the [tests](../tests/) directory for usage patterns
- Review the [tutorial](../demos/tutorial/) for step-by-step learning
- Explore the [FiniteVolume](../demos/FiniteVolume/) examples for advanced usage

## Contributing

When contributing to Samurai:

- Follow the existing code style and patterns
- Add appropriate tests for new features
- Update documentation for API changes
- Ensure compatibility with existing functionality
- Consider performance implications of changes

This documentation provides a comprehensive guide to using Samurai effectively for numerical simulations with adaptive mesh refinement. 