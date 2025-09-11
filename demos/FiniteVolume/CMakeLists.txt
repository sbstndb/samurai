# List of files requiring PETSc
set(PETSC_DEMOS
heat.cpp:finite-volume-heat
heat_heterogeneous.cpp:finite-volume-heat-heterogeneous
heat_nonlinear.cpp:finite-volume-heat-nonlinear
stokes_2d.cpp:finite-volume-stokes-2d
lid_driven_cavity.cpp:finite-volume-lid-driven-cavity
nagumo.cpp:finite-volume-nagumo
manual_block_matrix_assembly.cpp:manual_block_matrix_assembly
linear_convection.cpp:finite-volume-linear-convection
)

# List of standard files (without PETSc)
set(STANDARD_DEMOS
AMR_Burgers_Hat.cpp:finite-volume-amr-burgers-hat
level_set.cpp:finite-volume-level-set
level_set_from_scratch.cpp:finite-volume-level-set-from-scratch
advection_1d.cpp:finite-volume-advection-1d
advection_2d.cpp:finite-volume-advection-2d
advection_2d_user_bc.cpp:finite-volume-advection-2d-user-bc
scalar_burgers_2d.cpp:finite-volume-scalar-burgers-2d
linear_convection_obstacle.cpp:finite-volume-linear-convection-obstacle
burgers.cpp:finite-volume-burgers
burgers_mra.cpp:finite-volume-burgers-mra
burgers_os.cpp:finite-volume-burgers-os
)

# Create executables with PETSc
if(${WITH_PETSC})
    message(STATUS "Building demos with PETSc support")

    include(CMakeFindDependencyMacro)
    find_dependency(PkgConfig)
    pkg_check_modules(PETSC REQUIRED PETSc)

    include_directories(${PETSC_INCLUDE_DIRS})

    foreach(demo_entry ${PETSC_DEMOS})
        string(REPLACE ":" ";" demo_parts ${demo_entry})
        list(GET demo_parts 0 source_file)
        list(GET demo_parts 1 executable_name)

        add_executable(${executable_name} ${source_file})
        target_compile_definitions(${executable_name} PUBLIC SAMURAI_WITH_PETSC)
        target_link_libraries(${executable_name} samurai CLI11::CLI11 ${PETSC_LINK_LIBRARIES})
    endforeach()
endif()

# Create standard executables
foreach(demo_entry ${STANDARD_DEMOS})
    string(REPLACE ":" ";" demo_parts ${demo_entry})
    list(GET demo_parts 0 source_file)
    list(GET demo_parts 1 executable_name)

    add_executable(${executable_name} ${source_file})
    target_link_libraries(${executable_name} samurai CLI11::CLI11)

    # Experimental: add a small C++20 module to advection_2d if supported by CMake/Compiler
    if(executable_name STREQUAL "finite-volume-advection-2d")
        # Guard on CMake version and generator for C++20 modules integration
        if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.28" AND SAMURAI_USE_MODULES)
            # Restrict to compilers with supported module scanning (Clang/MSVC)
            if((CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR MSVC) AND (CMAKE_GENERATOR MATCHES "Ninja" OR (MSVC AND CMAKE_GENERATOR MATCHES "Visual Studio")))
                # Enable module scanning in newer CMake when available
                set(CMAKE_EXPERIMENTAL_CXX_MODULE_CMAKE_API "cxx_modules")
                # Ensure scanning is enabled (default ON in recent CMake, explicit here for clarity)
                set(CMAKE_CXX_SCAN_FOR_MODULES ON)
                if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
                    # Try both unversioned and versioned scanner binaries
                    find_program(CLANG_SCAN_DEPS_EXECUTABLE NAMES clang-scan-deps clang-scan-deps-20 clang-scan-deps-19 clang-scan-deps-18)
                    if(NOT CLANG_SCAN_DEPS_EXECUTABLE)
                        message(STATUS "clang++ detected but 'clang-scan-deps' not found; disabling modules for ${executable_name}. Using header fallback.")
                        set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
                    endif()
                endif()
                if(CMAKE_CXX_SCAN_FOR_MODULES)
                    # Enable dynamic dependency scanning for Clang + Ninja
                    set(CMAKE_EXPERIMENTAL_CXX_MODULE_DYNDEP 1)
                    target_sources(${executable_name}
                        PRIVATE
                            FILE_SET advection_mods TYPE CXX_MODULES FILES
                                ${CMAKE_CURRENT_SOURCE_DIR}/advection_utils.ixx
                    )
                    target_compile_definitions(${executable_name} PRIVATE SAMURAI_USE_MODULE_ADVECTION_UTILS)
                endif()
            else()
                message(STATUS "C++20 modules disabled for ${executable_name} (compiler '${CMAKE_CXX_COMPILER_ID}' and generator '${CMAKE_GENERATOR}' not supported). Using header fallback.")
            endif()
        endif()
    endif()
endforeach()

# Specific options for MSVC
if(MSVC)
    target_compile_options(finite-volume-level-set-from-scratch PUBLIC /bigobj)
    target_compile_options(finite-volume-level-set PUBLIC /bigobj)
endif()

# add_subdirectory(BZ)
