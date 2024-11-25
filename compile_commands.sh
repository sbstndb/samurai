## This file build samueai deos with different compile flags
## 1) With GCC, slower to faster
## 2) With LLVM

# 1) With GCC
cmake -B build_gcc_O0 -S . -DCMAKE_BUILD_TYPE="Debug" -DCMAKE_CXX_FLAGS="-mtune=native -march=native -O0" -DCMAKE_CXX_COMPILER=g++ -DBUILD_DEMOS=ON
cmake -B build_gcc_O1 -S . -DCMAKE_BUILD_TYPE="Debug" -DCMAKE_CXX_FLAGS="-mtune=native -march=native -O1" -DCMAKE_CXX_COMPILER=g++ -DBUILD_DEMOS=ON
cmake -B build_gcc_O2 -S . -DCMAKE_BUILD_TYPE="Debug" -DCMAKE_CXX_FLAGS="-mtune=native -march=native -O2" -DCMAKE_CXX_COMPILER=g++ -DBUILD_DEMOS=ON
cmake -B build_gcc_O3 -S . -DCMAKE_BUILD_TYPE="Debug" -DCMAKE_CXX_FLAGS="-mtune=native -march=native -O3" -DCMAKE_CXX_COMPILER=g++ -DBUILD_DEMOS=ON
cmake -B build_gcc_Ofast -S . -DCMAKE_BUILD_TYPE="Debug" -DCMAKE_CXX_FLAGS="-mtune=native -march=native -Ofast" -DCMAKE_CXX_COMPILER=g++ -DBUILD_DEMOS=ON
cmake -B build_gcc_without_native -S . -DCMAKE_BUILD_TYPE="Debug" -DCMAKE_CXX_FLAGS="-Ofast" -DCMAKE_CXX_COMPILER=g++ -DBUILD_DEMOS=ON
cmake -B build_gcc_full -S . -DCMAKE_BUILD_TYPE="Debug" -DCMAKE_CXX_FLAGS="-mtune=native -march=native -Ofast -mavx2 -funroll-loops -fpeel-loops -ftree-vectorize" -DCMAKE_CXX_COMPILER=g++ -DBUILD_DEMOS=ON
cmake -B build_gcc_xsimd -S . -DCMAKE_BUILD_TYPE="Debug" -DCMAKE_CXX_FLAGS="-mtune=native -march=native -Ofast -mavx2 -funroll-loops -fpeel-loops -ftree-vectorize" -DCMAKE_CXX_COMPILER=g++ -DBUILD_DEMOS=ON -DXTENSOR_USE_XSIMD=ON
cmake -B build_gcc_omp -S . -DCMAKE_BUILD_TYPE="Debug" -DCMAKE_CXX_FLAGS="-mtune=native -march=native -Ofast -mavx2 -funroll-loops -fpeel-loops -ftree-vectorize" -DCMAKE_CXX_COMPILER=g++ -DBUILD_DEMOS=ON -DWITH_OPENMP=ON

# 2) with LLVM

cmake -B build_clang_O2 -S . -DCMAKE_BUILD_TYPE="Debug" -DCMAKE_CXX_FLAGS="-mtune=native -march=native -O2" -DCMAKE_CXX_COMPILER=clang++ -DBUILD_DEMOS=ON -DCMAKE_EXE_LINKER_FLAGS="-pie"
cmake -B build_clang_O3 -S . -DCMAKE_BUILD_TYPE="Debug" -DCMAKE_CXX_FLAGS="-mtune=native -march=native -O3" -DCMAKE_CXX_COMPILER=clang++ -DBUILD_DEMOS=ON -DCMAKE_EXE_LINKER_FLAGS="-pie"
cmake -B build_clang_Ofast -S . -DCMAKE_BUILD_TYPE="Debug" -DCMAKE_CXX_FLAGS="-mtune=native -march=native -Ofast" -DCMAKE_CXX_COMPILER=clang++ -DBUILD_DEMOS=ON -DCMAKE_EXE_LINKER_FLAGS="-pie"
cmake -B build_clang_without_native -S . -DCMAKE_BUILD_TYPE="Debug" -DCMAKE_CXX_FLAGS="-Ofast" -DCMAKE_CXX_COMPILER=clang++ -DBUILD_DEMOS=ON -DCMAKE_EXE_LINKER_FLAGS="-pie"
cmake -B build_clang_full -S . -DCMAKE_BUILD_TYPE="Debug" -DCMAKE_CXX_FLAGS="-mtune=native -march=native -Ofast -mavx2 -funroll-loops -fpeel-loops -ftree-vectorize" -DCMAKE_CXX_COMPILER=clang++ -DBUILD_DEMOS=ON -DCMAKE_EXE_LINKER_FLAGS="-pie"
cmake -B build_clang_xsimd -S . -DCMAKE_BUILD_TYPE="Debug" -DCMAKE_CXX_FLAGS="-mtune=native -march=native -Ofast -mavx2 -funroll-loops -fpeel-loops -ftree-vectorize" -DCMAKE_CXX_COMPILER=clang++ -DBUILD_DEMOS=ON -DXTENSOR_USE_XSIMD=ON -DCMAKE_EXE_LINKER_FLAGS="-pie"
cmake -B build_clang_omp -S . -DCMAKE_BUILD_TYPE="Debug" -DCMAKE_CXX_FLAGS="-mtune=native -march=native -Ofast -mavx2 -funroll-loops -fpeel-loops -ftree-vectorize" -DCMAKE_CXX_COMPILER=clang++ -DBUILD_DEMOS=ON -DWITH_OPENMP=ON -DCMAKE_EXE_LINKER_FLAGS="-pie"

