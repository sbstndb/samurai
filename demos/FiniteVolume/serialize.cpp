// Copyright 2018-2024 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#include <samurai/io/hdf5.hpp>
#include <samurai/io/restart.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/samurai.hpp>
#include <samurai/schemes/fv.hpp>

#include <filesystem>


#include <boost/mpi.hpp>
#include <boost/serialization/serialization.hpp>


namespace fs = std::filesystem;

template <class Field>
void save(const fs::path& path, const std::string& filename, const Field& u, const std::string& suffix = "")
{
    auto mesh   = u.mesh();
    auto level_ = samurai::make_field<std::size_t, 1>("level", mesh);

    if (!fs::exists(path))
    {
        fs::create_directory(path);
    }

    samurai::for_each_cell(mesh,
                           [&](const auto& cell)
                           {
                               level_[cell] = cell.level;
                           });

    samurai::save(path, fmt::format("{}{}", filename, suffix), mesh, u, level_);
    samurai::dump(path, fmt::format("{}_restart{}", filename, suffix), mesh, u);
}



struct SimpleData{
        int a;
        double b;
        double values[100] ;
        double values2[10000];


        template <class Archive>
                void serialize(Archive & ar, const unsigned int){
                        ar & a ;
                        ar & b ;
                        ar & values ;
                        ar & values2 ;
                }

};


void benchmark_raw_mpi_simple(int rank, int size, int num_iterations){
	SimpleData data; 
	if (rank == 0){
		for (int i = 0 ; i < num_iterations; i++){
			MPI_Send(&data.a, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
			MPI_Send(&data.b, 1, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD);
			MPI_Send(data.values, 100, MPI_DOUBLE, 1, 2, MPI_COMM_WORLD);
                        MPI_Send(data.values2, 10000, MPI_DOUBLE, 1, 3, MPI_COMM_WORLD);
										      
		}
		int ack ; 
	       MPI_Recv(&ack, 1, MPI_INT, 1, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}	
	else if (rank == 1){
		for (int i = 0 ; i < num_iterations; i++){
                        MPI_Recv(&data.a, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        MPI_Recv(&data.b, 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);		
			MPI_Recv(data.values, 100, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        MPI_Recv(data.values2, 10000, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			
		}
	        int ack = 1;
	        MPI_Send(&ack, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
	}
}




void benchmark_boost_mpi_simple(int num_iterations) {
    boost::mpi::communicator world;  
    int rank = world.rank();
    SimpleData data;


    if (rank == 0) {
        double start = MPI_Wtime(); 
        for (int i = 0; i < num_iterations; ++i) {
            world.send(1, 0, data); 
        }
        int ack;
        world.recv(1, 1, ack);  
        double end = MPI_Wtime();
    } else if (rank == 1) {
        for (int i = 0; i < num_iterations; ++i) {
            world.recv(0, 0, data);
        }
        int ack = 1;
        world.send(0, 1, ack); 
    }
}


int main(int argc, char* argv[])
{
    auto& app = samurai::initialize("Finite volume example for the linear convection equation", argc, argv);

    static constexpr std::size_t dim = 2;
    using Config                     = samurai::MRConfig<dim, 3>;
    using Box                        = samurai::Box<double, dim>;
    using point_t                    = typename Box::point_t;

    std::cout << "------------------------- Linear convection -------------------------" << std::endl;

    //--------------------//
    // Program parameters //
    //--------------------//

    // Simulation parameters
    double left_box  = -1;
    double right_box = 1;

    // Time integration
    double Tf  = 3;
    double dt  = 0;
    double cfl = 0.95;
    double t   = 0.;
    std::string restart_file;

    // Multiresolution parameters
    std::size_t min_level = 1;
    std::size_t max_level = dim == 1 ? 6 : 4;
    double mr_epsilon     = 1e-4; // Threshold used by multiresolution
    double mr_regularity  = 1.;   // Regularity guess for multiresolution

    // Output parameters
    fs::path path        = fs::current_path();
    std::string filename = "linear_convection_" + std::to_string(dim) + "D";
    std::size_t nfiles   = 0;

    app.add_option("--left", left_box, "The left border of the box")->capture_default_str()->group("Simulation parameters");
    app.add_option("--right", right_box, "The right border of the box")->capture_default_str()->group("Simulation parameters");
    app.add_option("--Ti", t, "Initial time")->capture_default_str()->group("Simulation parameters");
    app.add_option("--Tf", Tf, "Final time")->capture_default_str()->group("Simulation parameters");
    app.add_option("--restart-file", restart_file, "Restart file")->capture_default_str()->group("Simulation parameters");
    app.add_option("--dt", dt, "Time step")->capture_default_str()->group("Simulation parameters");
    app.add_option("--cfl", cfl, "The CFL")->capture_default_str()->group("Simulation parameters");
    app.add_option("--min-level", min_level, "Minimum level of the multiresolution")->capture_default_str()->group("Multiresolution");
    app.add_option("--max-level", max_level, "Maximum level of the multiresolution")->capture_default_str()->group("Multiresolution");
    app.add_option("--mr-eps", mr_epsilon, "The epsilon used by the multiresolution to adapt the mesh")
        ->capture_default_str()
        ->group("Multiresolution");
    app.add_option("--mr-reg",
                   mr_regularity,
                   "The regularity criteria used by the multiresolution to "
                   "adapt the mesh")
        ->capture_default_str()
        ->group("Multiresolution");
    app.add_option("--path", path, "Output path")->capture_default_str()->group("Output");
    app.add_option("--filename", filename, "File name prefix")->capture_default_str()->group("Output");
    app.add_option("--nfiles", nfiles, "Number of output files")->capture_default_str()->group("Output");
    app.allow_extras();
    SAMURAI_PARSE(argc, argv);

    //--------------------//
    // Problem definition //
    //--------------------//

    point_t box_corner1, box_corner2;
    box_corner1.fill(left_box);
    box_corner2.fill(right_box);
    Box box(box_corner1, box_corner2);
    std::array<bool, dim> periodic;
    periodic.fill(true);
    samurai::MRMesh<Config> mesh;
    auto u = samurai::make_field<1>("u", mesh);

    if (restart_file.empty())
    {
        mesh = {box, min_level, max_level, periodic};
        // Initial solution
        u = samurai::make_field<1>("u",
                                   mesh,
                                   [](const auto& coords)
                                   {
                                       if constexpr (dim == 1)
                                       {
                                           const auto& x = coords(0);
                                           return (x >= -0.8 && x <= -0.3) ? 1. : 0.;
                                       }
                                       else
                                       {
                                           const auto& x = coords(0);
                                           const auto& y = coords(1);
                                           return (x >= -0.8 && x <= -0.3 && y >= 0.3 && y <= 0.8) ? 1. : 0.;
                                       }
                                   });
    }
    else
    {
        samurai::load(restart_file, mesh, u);
    }

    // Send mesh to other mpi rank
    // using boost::mpi


        auto field_mpi = samurai::make_field<1>("u", mesh);

	auto rep = 1000 ; 

	boost::mpi::communicator world ;
	int rank = world.rank() ; 
	int size = world.size() ; 

	double start, end ; 
    world.barrier();
    start = MPI_Wtime() ; 
    if (rank == 0 ){
	// send
	for (int i = 0 ; i < rep ; i++)
		world.send(1, 1, mesh); 	
    }
    else if (rank == 1){
	// recv
	for (int i = 0 ; i < rep ; i++)
	world.recv(0, 1, mesh);
    }
    world.barrier();
    end = MPI_Wtime() ; 
    std::cout << "Boost MPI comm for mesh  : " << (end - start)/rep << "s" << std::endl ; 



	MPI_Barrier(MPI_COMM_WORLD) ; 
	start = MPI_Wtime() ; 
	benchmark_raw_mpi_simple(rank, size, rep);
	end = MPI_Wtime() ; 
        std::cout << "Raw MPI SimpleData: " << (end - start) / rep << " s/op\n";


        MPI_Barrier(MPI_COMM_WORLD) ;
        start = MPI_Wtime() ;
        benchmark_boost_mpi_simple(rep);
        end = MPI_Wtime() ;
        std::cout << "boost MPI SimpleData: " << (end - start) / rep << " s/op\n";
    




    samurai::finalize();
    return 0;
}
