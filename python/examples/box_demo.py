#!/usr/bin/env python3
"""
Demonstration of Samurai Box2D Python bindings

This simple demo shows how to use the Box2D class from Python.
"""

import sys
sys.path.insert(0, 'build/python')

import samurai_core
import numpy as np

print("=" * 60)
print("Samurai Box2D Python Bindings - Demo")
print("=" * 60)

# Create a 2D box
print("\n1. Creating a 2D box...")
box = samurai_core.Box2D(np.array([0., 0.]), np.array([1., 1.]))
print(f"   Box: {box}")
print(f"   Min corner: {box.min}")
print(f"   Max corner: {box.max}")
print(f"   Length: {box.length()}")

# Check validity
print(f"\n2. Is the box valid? {box.is_valid()}")

# Create a second box that intersects
print("\n3. Creating a second box...")
box2 = samurai_core.Box2D(np.array([0.5, 0.5]), np.array([1.5, 1.5]))
print(f"   Box2: {box2}")

# Check intersection
print(f"\n4. Do they intersect? {box.intersects(box2)}")

# Get intersection
intersection = box.intersection(box2)
print(f"   Intersection: {intersection}")
print(f"   Intersection area: {intersection.length()[0] * intersection.length()[1]}")

# Scale the box
print("\n5. Scaling the first box by 2x...")
scaled_box = box * 2.0
print(f"   Scaled box: {scaled_box}")
print(f"   New length: {scaled_box.length()}")

# Compute difference (box minus smaller box)
print("\n6. Computing box difference...")
big_box = samurai_core.Box2D(np.array([-1., -1.]), np.array([1., 1.]))
small_box = samurai_core.Box2D(np.array([-0.5, -0.5]), np.array([0.5, 0.5]))
difference = big_box.difference(small_box)
print(f"   Big box: {big_box}")
print(f"   Small box (hole): {small_box}")
print(f"   Number of resulting boxes: {len(difference)}")
for i, b in enumerate(difference):
    print(f"     [{i}]: {b}")

print("\n" + "=" * 60)
print("Demo completed successfully!")
print("=" * 60)
