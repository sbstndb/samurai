cmake_minimum_required(VERSION 3.16)
project(csir_unified LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options to control what to build/fetch
option(CSIR_BUILD_TESTS "Build unit tests" ON)
option(CSIR_BUILD_BENCHMARKS "Build benchmarks" OFF)
option(CSIR_BUILD_DEMOS "Build demos" ON)
option(CSIR_FETCH_DEPS "Fetch test/benchmark dependencies if not found" OFF)

# Header-only library exposing include path for consumers
add_library(csir_unified_lib INTERFACE)
# Expose both this module dir and the project include root (for "samurai/..." includes)
target_include_directories(csir_unified_lib INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../..
)

# Optional: fmt (needed by core headers)
set(_have_fmt FALSE)
find_package(fmt QUIET CONFIG)
if (fmt_FOUND)
  set(_have_fmt TRUE)
elseif (CSIR_FETCH_DEPS)
  include(FetchContent)
  FetchContent_Declare(
    fmt
    URL https://github.com/fmtlib/fmt/archive/refs/tags/10.2.1.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
  FetchContent_MakeAvailable(fmt)
  set(_have_fmt TRUE)
endif()

if (_have_fmt)
  target_link_libraries(csir_unified_lib INTERFACE fmt::fmt)
endif()

# Try to find dependencies locally first; optionally fetch if requested
set(_have_gtest FALSE)
set(_have_benchmark FALSE)
set(_have_xtensor FALSE)

if (CSIR_BUILD_TESTS)
  find_package(GTest QUIET CONFIG)
  if (GTest_FOUND)
    set(_have_gtest TRUE)
  elseif (CSIR_FETCH_DEPS)
    include(FetchContent)
    FetchContent_Declare(
      googletest
      URL https://github.com/google/googletest/archive/refs/tags/release-1.11.0.zip
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(googletest)
    set(_have_gtest TRUE)
  endif()
endif()

if (CSIR_BUILD_BENCHMARKS)
  find_package(benchmark QUIET CONFIG)
  if (benchmark_FOUND)
    set(_have_benchmark TRUE)
  elseif (CSIR_FETCH_DEPS)
    include(FetchContent)
    FetchContent_Declare(
      benchmark
      URL https://github.com/google/benchmark/archive/refs/tags/v1.6.1.zip
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(benchmark)
    set(_have_benchmark TRUE)
  endif()
endif()

# xtensor (headers) and xtl for subset algebra utilities
find_package(xtensor QUIET CONFIG)
if (xtensor_FOUND)
  set(_have_xtensor TRUE)
elseif (CSIR_FETCH_DEPS)
  include(FetchContent)
  # xtl first
  FetchContent_Declare(
    xtl
    URL https://github.com/xtensor-stack/xtl/archive/refs/tags/0.7.5.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
  # xtensor
  FetchContent_Declare(
    xtensor
    URL https://github.com/xtensor-stack/xtensor/archive/refs/tags/0.24.6.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
  FetchContent_MakeAvailable(xtl xtensor)
  set(_have_xtensor TRUE)
endif()

if (_have_xtensor)
  target_link_libraries(csir_unified_lib INTERFACE xtensor xtl)
endif()

# Add subdirectories conditionally
if (CSIR_BUILD_TESTS AND _have_gtest)
  add_subdirectory(test)
elseif (CSIR_BUILD_TESTS)
  message(STATUS "GTest not found; tests will be skipped (enable CSIR_FETCH_DEPS to fetch).")
endif()

if (CSIR_BUILD_BENCHMARKS AND _have_benchmark)
  add_subdirectory(benchmark)
elseif (CSIR_BUILD_BENCHMARKS)
  message(STATUS "benchmark not found; benchmarks will be skipped (enable CSIR_FETCH_DEPS to fetch).")
endif()

if (CSIR_BUILD_DEMOS)
  add_subdirectory(demo)
endif()
