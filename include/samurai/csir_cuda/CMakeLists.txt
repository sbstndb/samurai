cmake_minimum_required(VERSION 3.18)
project(CSIR_CUDA LANGUAGES CXX CUDA)

# Find CUDA and set architecture flags
# The user specified an RTX 1000 ADA laptop, which corresponds to sm_89.
# We include other architectures for broader compatibility.
find_package(CUDA REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES 89)

# --- Options ---
option(CSIR_CUDA_BUILD_DEMOS "Build demo applications" ON)
option(CSIR_CUDA_BUILD_BENCHMARKS "Build benchmark applications" ON)

# --- CUDA Source Files ---
# For now, this is a placeholder. The actual implementation will be built here.
add_library(csir_cuda_lib src/csir_cuda.cu)


# --- Demos ---
if(CSIR_CUDA_BUILD_DEMOS)
    # Placeholder for demo targets
    add_executable(demo_1d_cuda demo/demo_1d.cu)
    target_link_libraries(demo_1d_cuda PRIVATE csir_cuda_lib)

    # add_executable(demo_2d_cuda demo/demo_2d.cu)
    # target_link_libraries(demo_2d_cuda PRIVATE csir_cuda_lib)

    # add_executable(demo_3d_cuda demo/demo_3d.cu)
    # target_link_libraries(demo_3d_cuda PRIVATE csir_cuda_lib)
endif()


# --- Benchmarks ---
if(CSIR_CUDA_BUILD_BENCHMARKS)
    add_executable(benchmark_2d_cuda benchmark/benchmark_2d.cu)
    target_link_libraries(benchmark_2d_cuda PRIVATE csir_cuda_lib)
endif()

