# Samurai Python Bindings - Standalone CMake Configuration
#
# This CMakeLists.txt is designed to build the Python bindings independently
# from the main samurai C++ library. It requires samurai to be installed
# and findable via CMake's find_package().
#
# Usage:
#   mkdir build && cd build
#   cmake .. -DCMAKE_PREFIX_PATH=/path/to/samurai/install
#   cmake --build .
#
# Or via pip/conda:
#   pip install .
#   conda build conda-recipe/

cmake_minimum_required(VERSION 3.20)
project(samurai_python LANGUAGES CXX)

# ============================================================================
# OPTIONS
# ============================================================================

option(SAMURAI_PYTHON_STANDALONE "Build as standalone package (requires installed samurai)" ON)
option(BUILD_PYTHON_TESTS "Build Python tests" OFF)

# ============================================================================
# FIND DEPENDENCIES
# ============================================================================

message(STATUS "")
message(STATUS "=== Samurai Python Bindings - Standalone Build ===")
message(STATUS "")

# Find Python
find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)
message(STATUS "Python executable: ${Python_EXECUTABLE}")
message(STATUS "Python include dirs: ${Python_INCLUDE_DIRS}")
message(STATUS "Python site-packages: ${Python_SITEROOT}")
message(STATUS "Python site-arch: ${Python_SITEARCH}")

# Find pybind11
# Try to find pybind11 locally first
find_package(pybind11 2.13 CONFIG)

# Force use of conda pybind11 if available
if(DEFINED ENV{CONDA_PREFIX})
    set(pybind11_INCLUDE_DIR "$ENV{CONDA_PREFIX}/include" CACHE PATH "Pybind11 include dir from conda")
    message(STATUS "Using conda pybind11 from: ${pybind11_INCLUDE_DIR}")
endif()

if(NOT pybind11_FOUND)
    message(STATUS "pybind11 not found, fetching from GitHub...")
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.13.6
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(pybind11)
    message(STATUS "pybind11 fetched and configured")
else()
    message(STATUS "pybind11 found: ${pybind11_VERSION}")
endif()

# Find samurai C++ library
if(SAMURAI_PYTHON_STANDALONE)
    # Try to find installed samurai
    find_package(samurai CONFIG REQUIRED)

    # Display samurai configuration
    message(STATUS "Samurai C++ library found:")
    message(STATUS "  Version: ${samurai_VERSION}")
    if(TARGET samurai::samurai)
        message(STATUS "  Target: samurai::samurai")

        # Get samurai include directories
        get_target_property(SAMURAI_INCLUDE_DIRS samurai::samurai INTERFACE_INCLUDE_DIRECTORIES)
        if(SAMURAI_INCLUDE_DIRS)
            message(STATUS "  Include dirs: ${SAMURAI_INCLUDE_DIRS}")
        endif()

        # Get samurai link libraries
        get_target_property(SAMURAI_LINK_LIBS samurai::samurai INTERFACE_LINK_LIBRARIES)
        if(SAMURAI_LINK_LIBS)
            message(STATUS "  Link libraries: ${SAMURAI_LINK_LIBS}")
        endif()

        # Get samurai compile definitions
        get_target_property(SAMURAI_DEFS samurai::samurai INTERFACE_COMPILE_DEFINITIONS)
        if(SAMURAI_DEFS)
            message(STATUS "  Definitions: ${SAMURAI_DEFS}")
        endif()
    endif()
else()
    # Fallback: try to use parent project (for backward compatibility)
    if(TARGET samurai)
        message(STATUS "Using samurai from parent project")
        add_alias(samurai::samurai samurai)
    else()
        message(FATAL_ERROR "samurai C++ library not found. "
                           "Please install samurai first or set CMAKE_PREFIX_PATH.")
    endif()
endif()

message(STATUS "")

# ============================================================================
# CREATE PYTHON MODULE
# ============================================================================

# Collect all binding source files
set(BINDING_SOURCES
    src/bindings/main.cpp
    src/bindings/box_bindings.cpp
    src/bindings/domain_builder_bindings.cpp
    src/bindings/mesh_config_bindings.cpp
    src/bindings/mesh_bindings.cpp
    src/bindings/field_bindings.cpp
    src/bindings/interval_bindings.cpp
    src/bindings/algorithm_bindings.cpp
    src/bindings/operator_bindings.cpp
    src/bindings/bc_bindings.cpp
    src/bindings/mra_config_bindings.cpp
    src/bindings/adapt_bindings.cpp
    src/bindings/io_bindings.cpp
)

# Create the pybind11 module
pybind11_add_module(samurai_python ${BINDING_SOURCES})

# Set target properties for better symbol visibility and optimization
set_target_properties(samurai_python PROPERTIES
    CXX_VISIBILITY_PRESET "hidden"
    VISIBILITY_INLINES_HIDDEN ON
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME "samurai_python"
)

# ============================================================================
# LINK LIBRARIES
# ============================================================================

# Link against samurai C++ library
target_link_libraries(samurai_python
    PRIVATE
        samurai::samurai
)

# Ensure C++20 standard
target_compile_features(samurai_python PRIVATE cxx_std_20)

# ============================================================================
# INCLUDE DIRECTORIES
# ============================================================================

# Add Python include directories
target_include_directories(samurai_python
    PRIVATE
        ${Python_INCLUDE_DIRS}
)

# Include directories for bindings to find samurai headers
# When using installed samurai, the headers are in the install location
if(SAMURAI_PYTHON_STANDALONE AND TARGET samurai::samurai)
    # The include directories are already propagated via the target
    # But we need them for the bindings compilation
    get_target_property(SAMURAI_INCLUDE_DIRS samurai::samurai INTERFACE_INCLUDE_DIRECTORIES)
    if(SAMURAI_INCLUDE_DIRS)
        target_include_directories(samurai_python
            PRIVATE
                ${SAMURAI_INCLUDE_DIRS}
        )
    endif()
endif()

# ============================================================================
# COMPILE DEFINITIONS
# ============================================================================

# Propagate samurai compile definitions (important for template instantiation)
if(TARGET samurai::samurai)
    get_target_property(SAMURAI_DEFS samurai::samurai INTERFACE_COMPILE_DEFINITIONS)
    if(SAMURAI_DEFS)
        target_compile_definitions(samurai_python
            PRIVATE
                ${SAMURAI_DEFS}
        )
    endif()
endif()

# Platform-specific definitions
if(WIN32)
    target_compile_definitions(samurai_python PRIVATE "WIN32_LEAN_AND_MEAN")
endif()

# ============================================================================
# OUTPUT DIRECTORY
# ============================================================================

# Set output directory for the module
set_target_properties(samurai_python PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib
)

# ============================================================================
# PYTHON UTILITIES COPY
# ============================================================================

# Copy pure Python utilities to build directory
add_custom_command(TARGET samurai_python POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/src/samurai_python
        ${CMAKE_CURRENT_BINARY_DIR}/lib/samurai_python
    COMMENT "Copying Python utilities to build directory"
)

# ============================================================================
# INSTALLATION
# ============================================================================

include(GNUInstallDirs)

# Install the compiled Python module
install(TARGETS samurai_python
    LIBRARY DESTINATION ${Python_SITEARCH}
)

# Install pure Python utilities
install(DIRECTORY src/samurai_python/
    DESTINATION ${Python_SITEROOT}/samurai_python
    FILES_MATCHING PATTERN "*.py"
)

# Install __init__.py for main package (wraps the compiled module)
install(FILES src/samurai/__init__.py
    DESTINATION ${Python_SITEROOT}/samurai
)

# ============================================================================
# TESTS
# ============================================================================

if(BUILD_PYTHON_TESTS)
    enable_testing()

    # Add test directory
    add_test(NAME python_import
        COMMAND ${Python_EXECUTABLE} -c "import sys; sys.path.insert(0, '${CMAKE_CURRENT_BINARY_DIR}/lib'); import samurai_python; print(samurai_python.__version__)"
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib
    )

    # Add pytest tests if pytest is available
    find_package(Pytest)
    if(Pytest_FOUND)
        add_test(NAME pytest
            COMMAND ${Pytest_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests -v
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        )
    endif()
endif()

# ============================================================================
# SUMMARY
# ============================================================================

message(STATUS "")
message(STATUS "=== Samurai Python Bindings Configuration Summary ===")
message(STATUS "  Build type:          ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard:        C++20")
message(STATUS "  Python version:      ${Python_VERSION}")
message(STATUS "  Module name:         samurai_python")
message(STATUS "  Output directory:    ${CMAKE_CURRENT_BINARY_DIR}/lib")
message(STATUS "  Install directory:   ${Python_SITEARCH}")
message(STATUS "  Standalone build:    ${SAMURAI_PYTHON_STANDALONE}")
message(STATUS "====================================================")
message(STATUS "")

# ============================================================================
# DEVELOPMENT HELPER TARGETS
# ============================================================================

# Target to build and install locally
add_custom_target(dev-install
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_CURRENT_BINARY_DIR} --target install
            --prefix ${CMAKE_CURRENT_BINARY_DIR}/install
    COMMAND ${CMAKE_COMMAND} -E env
            PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/install/${Python_SITEROOT}:$ENV{PYTHONPATH}
            ${Python_EXECUTABLE} -c "import samurai_python; print(f'Development install successful: {samurai_python.__version__}')"
    COMMENT "Building and installing for development"
)

# Print helpful development message
message(STATUS "Development targets:")
message(STATUS "  cmake --build . --target samurai_python  - Build the module")
message(STATUS "  cmake --build . --target dev-install     - Build and install locally")
message(STATUS "  cmake --install . --prefix /path/to/install - Install to specific location")
message(STATUS "")
