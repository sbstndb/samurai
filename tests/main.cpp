#ifdef SAMURAI_WITH_MPI
#include <boost/mpi.hpp>
#endif
#include <cstdlib>
#include <gtest/gtest.h>

int main(int argc, char* argv[])
{
#ifdef SAMURAI_WITH_MPI
    boost::mpi::environment env(argc, argv);
#endif
#ifdef SAMURAI_FIELD_CONTAINER_CUDA_THRUST
    setenv("HDF5_USE_FILE_LOCKING", "FALSE", 1);
#endif
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
