export BUILD_DIR="build_debug"

# spack load xtensor
# spack load xsimd
# spack load xtl

export FLAGS="-mtune=native -march=native -Ofast -mavx2 -mfma -g"

rm -rvf ${BUILD_DIR}
cmake -B ${BUILD_DIR} -S . -DBUILD_DEMOS=ON -DBUILD_TESTS=ON -DBUILD_BENCHMARKS=ON -DSAMURAI_CHECK_NAN=ON \
	-DWITH_STATS=ON \
	-DSANITIZERS=ON -DCCPCHECK=ON -DENABLE_COVERAGE=ON 
	-DCMAKE_CXX_COMPILER=g++
cmake --build ${BUILD_DIR} -j 3 -- -k 
