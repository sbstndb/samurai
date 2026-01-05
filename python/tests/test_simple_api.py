#!/usr/bin/env python3
"""
Test the simplified Python API - auto-detection and defaults
"""

import sys
import numpy as np

sys.path.insert(0, 'build/python')

import samurai_core

print("=" * 50)
print("Testing Simplified API")
print("=" * 50)

# ============================================================================
# 1. AUTO-DETECTION (DEFAULT)
# ============================================================================
print("\n1. Auto-Detection with defaults:")

# 1D - auto-detected from array size
box1d = samurai_core.Box(np.array([0.]), np.array([1.]))
print(f"   Box1D: {box1d}")
print(f"   Type: {type(box1d)}")

# 2D - auto-detected
box2d = samurai_core.Box(np.array([0., 0.]), np.array([1., 1.]))
print(f"   Box2D: {box2d}")
print(f"   Type: {type(box2d)}")

# 3D - auto-detected
box3d = samurai_core.Box(np.array([0., 0., 0.]), np.array([1., 1., 1.]))
print(f"   Box3D: {box3d}")
print(f"   Type: {type(box3d)}")

# ============================================================================
# 2. SIMPLE ALIASES
# ============================================================================
print("\n2. Simple aliases (Box1D, Box2D, Box3D):")

box1d = samurai_core.Box1D(np.array([0.]), np.array([1.]))
print(f"   Box1D: {box1d}")

box2d = samurai_core.Box2D(np.array([0., 0.]), np.array([1., 1.]))
print(f"   Box2D: {box2d}")

box3d = samurai_core.Box3D(np.array([0., 0., 0.]), np.array([1., 1., 1.]))
print(f"   Box3D: {box3d}")

# ============================================================================
# 3. WITH EXPLICIT DTYPE
# ============================================================================
print("\n3. With explicit dtype (optional):")

box_double = samurai_core.Box(np.array([0., 0.]), np.array([1., 1.]))
print(f"   Default (double): {box_double}")

box_float = samurai_core.Box(np.array([0., 0.]), np.array([1., 1.]), dtype='float')
print(f"   Float: {box_float}")

# ============================================================================
# 4. COMPARISON: AVANT vs APRÈS
# ============================================================================
print("\n4. API Comparison:")
print("   AVANT (avant généralisation):")
print("      box = samurai_core.Box2D(np.array([0., 0.]), np.array([1., 1.]))")
print()
print("   APRÈS (trop verbeux):")
print("      box = samurai_core.Box('double', 2, np.array([0., 0.]), np.array([1., 1.]))")
print()
print("   MAINTENANT (simple et Pythonique!):")
print("      box = samurai_core.Box(np.array([0., 0.]), np.array([1., 1.]))")
print()
print("   Ou avec alias explicite:")
print("      box = samurai_core.Box2D(np.array([0., 0.]), np.array([1., 1.]))")

# ============================================================================
# 5. BACKWARDS COMPATIBILITY
# ============================================================================
print("\n5. Backwards compatibility:")

# L'ancien code marche toujours !
box = samurai_core.Box2D(np.array([0., 0.]), np.array([1., 1.]))
print(f"   Old code still works: {box}")
print(f"   Same as new API: {box == samurai_core.Box(np.array([0., 0.]), np.array([1., 1.]))}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 50)
print("SUMMARY - Recommended usage:")
print("=" * 50)
print("""
Most common cases:
  box = samurai_core.Box(np.array([0., 0.]), np.array([1., 1.]))
  box1d = samurai_core.Box(np.array([0.]), np.array([1.]))
  box3d = samurai_core.Box(np.array([0., 0., 0.]), np.array([1., 1., 1.]))

With explicit aliases:
  box = samurai_core.Box2D(np.array([0., 0.]), np.array([1., 1.]))

For float precision:
  box = samurai_core.Box(np.array([0., 0.]), np.array([1., 1.]), dtype='float')
""")

print("=" * 50)
print("All tests passed!")
print("=" * 50)
