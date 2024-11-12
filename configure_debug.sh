export BUILD_DIR="build_debug"

# spack load xtensor
# spack load xsimd
# spack load xtl

export FLAGS="-mtune=native -march=native -O1 -g"

rm -rvf ${BUILD_DIR}
cmake -B ${BUILD_DIR} -S . -DBUILD_DEMOS=ON 
cmake --build ${BUILD_DIR} -j 3 -- -k 
