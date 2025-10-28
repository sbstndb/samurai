add_executable(csir_unified_tests test_csir_1d.cpp test_csir.cpp test_csir_3d.cpp test_subset_like.cpp)

target_link_libraries(csir_unified_tests GTest::gtest_main csir_unified_lib)
