# CMakeLists.txt for Samurai Python bindings
# Minimal proof-of-concept: Box2D only

cmake_minimum_required(VERSION 3.16)

# Find Python and pybind11
find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development.Module)

# Try to find pybind11 - first via system package, then via FetchContent
find_package(pybind11 2.10 QUIET)

if(NOT pybind11_FOUND)
    message(STATUS "pybind11 not found, installing via FetchContent...")

    include(FetchContent)
    FetchContent_Declare(
        pybind11
        URL https://github.com/pybind/pybind11/archive/refs/tags/v2.13.0.tar.gz
    )

    FetchContent_MakeAvailable(pybind11)
    message(STATUS "pybind11 v2.13.0 fetched successfully")
else()
    message(STATUS "pybind11 found: ${pybind11_VERSION}")
endif()

# Create the Python modules
pybind11_add_module(samurai_core
    box_bindings.cpp
)

pybind11_add_module(samurai_fields
    cell_bindings.cpp
    field_bindings.cpp
)

pybind11_add_module(samurai_vector_fields
    vector_field_bindings.cpp
)

# Set C++ standard
target_compile_features(samurai_core PRIVATE cxx_std_20)
target_compile_features(samurai_fields PRIVATE cxx_std_20)
target_compile_features(samurai_vector_fields PRIVATE cxx_std_20)

# Include Samurai headers
target_include_directories(samurai_core
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)

target_include_directories(samurai_fields
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)

target_include_directories(samurai_vector_fields
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)

# Link against Samurai (header-only, so just need the interface)
target_link_libraries(samurai_core
    PRIVATE
        samurai
)

target_link_libraries(samurai_fields
    PRIVATE
        samurai
)

target_link_libraries(samurai_vector_fields
    PRIVATE
        samurai
)

# Optimization for release builds
if(CMAKE_BUILD_TYPE MATCHES Release)
    target_compile_options(samurai_core PRIVATE
        -O3
        -march=native
    )
    target_compile_options(samurai_fields PRIVATE
        -O3
        -march=native
    )
    target_compile_options(samurai_vector_fields PRIVATE
        -O3
        -march=native
    )
endif()

# Set output directory to be importable by Python
set_target_properties(samurai_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python
)

set_target_properties(samurai_fields PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python
)

set_target_properties(samurai_vector_fields PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python
)

# Print build information
message(STATUS "")
message(STATUS "=== Samurai Python Bindings Configuration ===")
message(STATUS "Python executable: ${Python_EXECUTABLE}")
message(STATUS "Python include dirs: ${Python_INCLUDE_DIRS}")
message(STATUS "Python library dirs: ${Python_LIBRARY_DIRS}")
message(STATUS "pybind11 version: ${pybind11_VERSION}")
message(STATUS "Module output: ${CMAKE_BINARY_DIR}/python")
message(STATUS "C++ standard: C++20")
message(STATUS "============================================")
message(STATUS "")
