#!/usr/bin/env bash

# Ultra-simple comparison of configure+build time (in seconds) for
# the demo target finite-volume-advection-2d using Make vs Ninja.

set -euo pipefail

TARGET="finite-volume-advection-2d"
BUILD_MAKE="build-make"
BUILD_NINJA="build-ninja"

rm -rf "$BUILD_MAKE" "$BUILD_NINJA"

# Detect cores
NPROC=$( (command -v nproc >/dev/null 2>&1 && nproc) || sysctl -n hw.ncpu 2>/dev/null || echo 1 )

# Ensure cmake exists
command -v cmake >/dev/null 2>&1 || { echo "cmake not found" >&2; exit 1; }

echo "Target: $TARGET"
echo "Cores: $NPROC"

# Makefiles timing (seconds)
if command -v make >/dev/null 2>&1; then
  SECONDS=0
  cmake -S . -B "$BUILD_MAKE" -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release -DBUILD_DEMOS=ON -DWITH_PETSC=OFF
  cmake --build "$BUILD_MAKE" --parallel "$NPROC" --target "$TARGET"
  MAKE_S=$SECONDS
  echo "Make: ${MAKE_S}s"
else
  echo "Make: skipped (make not found)"
fi

# Ninja timing (seconds)
if command -v ninja >/dev/null 2>&1; then
  SECONDS=0
  cmake -S . -B "$BUILD_NINJA" -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DBUILD_DEMOS=ON -DWITH_PETSC=OFF
  cmake --build "$BUILD_NINJA" --parallel "$NPROC" --target "$TARGET"
  NINJA_S=$SECONDS
  echo "Ninja: ${NINJA_S}s"
else
  echo "Ninja: skipped (ninja not found)"
fi

