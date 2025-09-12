add_executable(csir_unified_benchmarks benchmark_1d.cpp benchmark_2d.cpp benchmark_3d.cpp)
target_link_libraries(csir_unified_benchmarks benchmark::benchmark_main csir_unified_lib)

# Samurai subset-based benchmarks (force evaluation inside apply)
add_executable(samurai_subset_benchmarks
  samurai_benchmark_1d.cpp
  samurai_benchmark_2d.cpp
  samurai_benchmark_3d.cpp
)
target_link_libraries(samurai_subset_benchmarks benchmark::benchmark_main csir_unified_lib)
