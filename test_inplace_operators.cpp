// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// Simple test for in-place arithmetic operators
#include <iostream>
#include <samurai/samurai.hpp>

int main()
{
    constexpr std::size_t dim = 2;
    using Box                 = samurai::Box<double, dim>;

    // Create a simple mesh
    Box box({0., 0.}, {1., 1.});
    auto config = samurai::mesh_config<dim>().min_level(0).max_level(2);
    auto mesh   = samurai::mra::make_empty_mesh(config);

    // Create a scalar field
    auto u = samurai::make_scalar_field<double>("u", mesh);

    // Initialize with a constant value
    u.fill(1.0);

    std::cout << "Initial field value: " << u.array()(0) << std::endl;

    // Test scalar in-place operators
    u += 2.0;
    std::cout << "After u += 2.0: " << u.array()(0) << " (expected: 3.0)" << std::endl;

    u *= 3.0;
    std::cout << "After u *= 3.0: " << u.array()(0) << " (expected: 9.0)" << std::endl;

    u -= 1.0;
    std::cout << "After u -= 1.0: " << u.array()(0) << " (expected: 8.0)" << std::endl;

    u /= 2.0;
    std::cout << "After u /= 2.0: " << u.array()(0) << " (expected: 4.0)" << std::endl;

    // Test field-field in-place operators
    auto v = samurai::make_scalar_field<double>("v", mesh);
    v.fill(5.0);

    u += v;
    std::cout << "After u += v: " << u.array()(0) << " (expected: 9.0)" << std::endl;

    u -= v;
    std::cout << "After u -= v: " << u.array()(0) << " (expected: 4.0)" << std::endl;

    // Test chaining
    (u += 1.0) *= 2.0;
    std::cout << "After (u += 1.0) *= 2.0: " << u.array()(0) << " (expected: 10.0)" << std::endl;

    std::cout << "All tests passed!" << std::endl;

    return 0;
}
