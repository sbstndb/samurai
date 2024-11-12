export BUILD_DIR="build_gcc"

# spack load xtensor
# spack load xsimd
# spack load xtl

export FLAGS="-mtune=native -march=native -Ofast -mavx2 -mfma -g"

rm -rvf ${BUILD_DIR}
cmake -B ${BUILD_DIR} -S . -DBUILD_DEMOS=ON -DCMAKE_CXX_COMPILER=g++
cmake --build ${BUILD_DIR} -j 3 -- -k 
