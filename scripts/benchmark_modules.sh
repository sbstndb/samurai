#!/usr/bin/env bash

# Benchmark compile time of finite-volume-advection-2d
# with C++20 modules OFF vs ON using Clang + Ninja.
#
# Prerequisites:
# - CMake >= 3.28
# - Ninja
# - clang-20, clang++-20, clang-scan-deps-20 (or adjust env vars below)

set -euo pipefail

TARGET=${TARGET:-finite-volume-advection-2d}
BUILD_DIR=${BUILD_DIR:-build-mod-bench}
CXX=${CXX:-clang++-20}
CC=${CC:-clang-20}
GEN=${GEN:-Ninja}
NPROC=$( (command -v nproc >/dev/null 2>&1 && nproc) || sysctl -n hw.ncpu 2>/dev/null || echo 1 )

echo "Compiler: $CXX ($CC)"
echo "Generator: $GEN"
echo "Cores: $NPROC"
echo "Build dir: $BUILD_DIR"

configure_build() {
  local use_modules=$1
  cmake -S . -B "$BUILD_DIR" -G "$GEN" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_DEMOS=ON -DWITH_PETSC=OFF \
    -DCMAKE_CXX_COMPILER="$CXX" -DCMAKE_C_COMPILER="$CC" \
    -DSAMURAI_USE_MODULES="$use_modules"
}

build_target() {
  cmake --build "$BUILD_DIR" --parallel "$NPROC" --target "$TARGET"
}

clean_target() {
  if command -v ninja >/dev/null 2>&1 && [ "$GEN" = "Ninja" ]; then
    ninja -C "$BUILD_DIR" -t clean || true
  else
    cmake --build "$BUILD_DIR" --target clean || true
  fi
}

time_build() {
  local label=$1
  SECONDS=0
  build_target
  local secs=$SECONDS
  echo "$label: ${secs}s"
}

# Modules OFF
configure_build OFF
clean_target
time_build "Build (modules OFF)"

# Modules ON
configure_build ON
clean_target
time_build "Build (modules ON)"

echo "Done."

