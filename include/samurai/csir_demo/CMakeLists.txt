cmake_minimum_required(VERSION 3.16)
project(csir_demo CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -mtune=native")

add_executable(demo main.cpp)
add_executable(demo_3d main_3d.cpp)
add_executable(benchmark_2d benchmark_2d.cpp)