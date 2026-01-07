#!/usr/bin/env python3
"""Performance benchmark for in-place operators POC

This script measures the performance improvement of in-place operators
vs copy operations for field arithmetic operations.

Run with: python python/benchmarks/benchmark_poc_inplace.py
"""

import timeit
import numpy as np
import samurai_python as sam


def setup_mesh_2d():
    """Create a 2D mesh for benchmarking.

    Uses a fixed mesh size (level 6) to provide meaningful timing data.
    Level 6 gives us 64x64 = 4096 cells, which is large enough for
    reliable timing but small enough for quick iterations.
    """
    box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
    config = sam.MeshConfig2D()
    config.min_level = 6  # Larger mesh for better timing
    config.max_level = 6
    return sam.MRMesh2D(box, config)


def setup_mesh_3d():
    """Create a 3D mesh for benchmarking.

    Level 4 gives us 16x16x16 = 4096 cells, comparable to the 2D case.
    """
    box = sam.geometry.box([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    config = sam.MeshConfig3D()
    config.min_level = 4
    config.max_level = 4
    return sam.MRMesh3D(box, config)


def benchmark_simple_inplace_vs_copy_2d():
    """Compare u += 1.0 vs u = u + 1.0 in 2D.

    This is the simplest possible test - a single scalar addition.
    Expected: In-place should be ~2x faster by avoiding allocation.
    """
    mesh = setup_mesh_2d()

    # Benchmark 1: In-place operation
    def inplace_op():
        u = sam.field.scalar(mesh, "u", init=1.0)
        u += 1.0
        return u

    time_inplace = timeit.timeit(inplace_op, number=100)

    # Benchmark 2: Copy operation
    def copy_op():
        u = sam.field.scalar(mesh, "u", init=1.0)
        u = u + 1.0
        return u

    time_copy = timeit.timeit(copy_op, number=100)

    print(f"  In-place (+=):  {time_inplace:.4f}s")
    print(f"  Copy (u = u +): {time_copy:.4f}s")
    print(f"  Speedup:        {time_copy/time_inplace:.2f}x")

    return time_inplace, time_copy


def benchmark_simple_inplace_vs_copy_3d():
    """Compare u += 1.0 vs u = u + 1.0 in 3D.

    Tests if the performance advantage scales to higher dimensions.
    """
    mesh = setup_mesh_3d()

    def inplace_op():
        u = sam.field.scalar(mesh, "u", init=1.0)
        u += 1.0
        return u

    time_inplace = timeit.timeit(inplace_op, number=100)

    def copy_op():
        u = sam.field.scalar(mesh, "u", init=1.0)
        u = u + 1.0
        return u

    time_copy = timeit.timeit(copy_op, number=100)

    print(f"  In-place (+=):  {time_inplace:.4f}s")
    print(f"  Copy (u = u +): {time_copy:.4f}s")
    print(f"  Speedup:        {time_copy/time_inplace:.2f}x")

    return time_inplace, time_copy


def benchmark_field_arithmetic_2d():
    """Compare in-place vs copy for field-field operations.

    Tests u += v where both are fields, which is more realistic
    than scalar addition.
    """
    mesh = setup_mesh_2d()

    def inplace_op():
        u = sam.field.scalar(mesh, "u", init=1.0)
        v = sam.field.scalar(mesh, "v", init=2.0)
        u += v
        return u

    time_inplace = timeit.timeit(inplace_op, number=100)

    def copy_op():
        u = sam.field.scalar(mesh, "u", init=1.0)
        v = sam.field.scalar(mesh, "v", init=2.0)
        u = u + v
        return u

    time_copy = timeit.timeit(copy_op, number=100)

    print(f"  In-place (u += v):  {time_inplace:.4f}s")
    print(f"  Copy (u = u + v):   {time_copy:.4f}s")
    print(f"  Speedup:            {time_copy/time_inplace:.2f}x")

    return time_inplace, time_copy


def benchmark_complex_expression():
    """Compare assign vs direct for complex expressions.

    This tests a realistic finite volume update:
    u1.assign(u - dt * flux) vs u1.assign(u); u1 -= dt * flux

    The in-place version might be faster if it avoids temporary
    field allocation for the intermediate result.
    """
    mesh = setup_mesh_2d()
    dt = 0.01

    # Benchmark with assign (creates temporary)
    def with_assign():
        u = sam.field.scalar(mesh, "u", init=1.0)
        flux = sam.field.scalar(mesh, "flux", init=0.5)
        u1 = sam.field.scalar(mesh, "u1", init=0.0)
        u1.assign(u - dt * flux)
        return u1

    # Benchmark with in-place (no temporary)
    def with_inplace():
        u = sam.field.scalar(mesh, "u", init=1.0)
        flux = sam.field.scalar(mesh, "flux", init=0.5)
        u1 = sam.field.scalar(mesh, "u1", init=0.0)
        u1.assign(u)
        u1 -= dt * flux
        return u1

    time_assign = timeit.timeit(with_assign, number=100)
    time_inplace = timeit.timeit(with_inplace, number=100)

    print(f"  assign(u - dt * flux):     {time_assign:.4f}s")
    print(f"  assign(u); inplace(-=):    {time_inplace:.4f}s")
    print(f"  Difference:                {(time_assign-time_inplace)/time_assign*100:.1f}%")

    return time_assign, time_inplace


def benchmark_chain_operations():
    """Benchmark chained in-place operations.

    Tests multiple in-place operations in sequence:
    u += a; u *= b; u -= c

    This should be significantly faster than:
    u = ((u + a) * b) - c
    """
    mesh = setup_mesh_2d()

    def chained_inplace():
        u = sam.field.scalar(mesh, "u", init=1.0)
        a = sam.field.scalar(mesh, "a", init=0.5)
        b = sam.field.scalar(mesh, "b", init=2.0)
        c = sam.field.scalar(mesh, "c", init=0.1)
        u += a
        u *= b
        u -= c
        return u

    def chained_copy():
        u = sam.field.scalar(mesh, "u", init=1.0)
        a = sam.field.scalar(mesh, "a", init=0.5)
        b = sam.field.scalar(mesh, "b", init=2.0)
        c = sam.field.scalar(mesh, "c", init=0.1)
        u = ((u + a) * b) - c
        return u

    time_inplace = timeit.timeit(chained_inplace, number=100)
    time_copy = timeit.timeit(chained_copy, number=100)

    print(f"  Chained in-place:  {time_inplace:.4f}s")
    print(f"  Chained copy:      {time_copy:.4f}s")
    print(f"  Speedup:           {time_copy/time_inplace:.2f}x")

    return time_inplace, time_copy


def benchmark_all_operators():
    """Test all in-place operators for consistency.

    Ensures that +=, -=, *=, /= all work and have similar performance.
    """
    mesh = setup_mesh_2d()

    results = {}

    for op_name, op in [('+=', lambda a, b: a.__iadd__(b)),
                        ('-=', lambda a, b: a.__isub__(b)),
                        ('*=', lambda a, b: a.__imul__(b)),
                        ('/=', lambda a, b: a.__itruediv__(b))]:
        def test_op():
            u = sam.field.scalar(mesh, "u", init=2.0)
            v = sam.field.scalar(mesh, "v", init=4.0)
            op(u, v)
            return u

        time_taken = timeit.timeit(test_op, number=100)
        results[op_name] = time_taken
        print(f"  {op_name}: {time_taken:.4f}s")

    return results


def run_all_benchmarks():
    """Run all benchmarks and collect results."""

    print("=" * 70)
    print("POC In-Place Operator Performance Benchmarks")
    print("=" * 70)
    print()

    # Collect all results
    results = {}

    # Benchmark 1: Simple scalar operations (2D)
    print("1. Simple Scalar Operation (2D, 64x64 cells)")
    print("-" * 70)
    results['simple_2d'] = benchmark_simple_inplace_vs_copy_2d()
    print()

    # Benchmark 2: Simple scalar operations (3D)
    print("2. Simple Scalar Operation (3D, 16x16x16 cells)")
    print("-" * 70)
    results['simple_3d'] = benchmark_simple_inplace_vs_copy_3d()
    print()

    # Benchmark 3: Field-field operations
    print("3. Field Arithmetic (2D, u += v)")
    print("-" * 70)
    results['field_arithmetic'] = benchmark_field_arithmetic_2d()
    print()

    # Benchmark 4: Complex expression
    print("4. Complex Expression (u1.assign(u - dt * flux))")
    print("-" * 70)
    results['complex_expr'] = benchmark_complex_expression()
    print()

    # Benchmark 5: Chained operations
    print("5. Chained Operations (u += a; u *= b; u -= c)")
    print("-" * 70)
    results['chained'] = benchmark_chain_operations()
    print()

    # Benchmark 6: All operators
    print("6. All In-Place Operators")
    print("-" * 70)
    results['all_ops'] = benchmark_all_operators()
    print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("Key Performance Improvements:")
    print(f"  Simple scalar (2D):     {results['simple_2d'][1]/results['simple_2d'][0]:.2f}x speedup")
    print(f"  Simple scalar (3D):     {results['simple_3d'][1]/results['simple_3d'][0]:.2f}x speedup")
    print(f"  Field arithmetic:       {results['field_arithmetic'][1]/results['field_arithmetic'][0]:.2f}x speedup")
    print(f"  Chained ops:            {results['chained'][1]/results['chained'][0]:.2f}x speedup")
    print()

    # Calculate average speedup
    speedups = [
        results['simple_2d'][1]/results['simple_2d'][0],
        results['simple_3d'][1]/results['simple_3d'][0],
        results['field_arithmetic'][1]/results['field_arithmetic'][0],
        results['chained'][1]/results['chained'][0],
    ]
    avg_speedup = np.mean(speedups)

    print(f"Average speedup:          {avg_speedup:.2f}x")
    print()

    return results


if __name__ == "__main__":
    # Run all benchmarks
    results = run_all_benchmarks()

    print("=" * 70)
    print("Benchmark complete!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("1. Copy these results into python/POC_MUTATION_RESULTS.md")
    print("2. Run functional tests: pytest python/tests/test_inplace_operators.py")
    print("3. Update the recommendation section based on findings")
