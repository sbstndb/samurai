// Copyright 2018-2024 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause
#include <CLI/CLI.hpp>

#include <samurai/hdf5.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/schemes/fv.hpp>
#include <samurai/samurai.hpp>


#include "xtensor/xarray.hpp"
#include "xtensor/xmath.hpp"


#include <chrono>

#include <filesystem>
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
}


int main(int argc, char* argv[])
{
    samurai::initialize(argc, argv);
    static constexpr std::size_t dim = 2;
    using Config                     = samurai::MRConfig<dim>;
    using Box                        = samurai::Box<double, dim>;
    using point_t                    = typename Box::point_t;

    std::cout << "------------------------- Test@sbstndbs -------------------------" << std::endl;
    //--------------------//
    // Program parameters //
    //--------------------//
    double left_box  = -1;
    double right_box = 1;

    //min_level == max_level in this example
    std::size_t min_level = 10;
    std::size_t max_level = min_level;

    int size_x = pow(2, max_level); 
    int size = size_x*size_x ; 

    std::cout << "Size : " << size  << std::endl ; 



    point_t box_corner1, box_corner2;
    box_corner1.fill(left_box);
    box_corner2.fill(right_box);
    Box box(box_corner1, box_corner2);
    samurai::MRMesh<Config> mesh{box, min_level, max_level};

    auto x = samurai::make_field<double, 1>("x", mesh);
    auto b = samurai::make_field<double, 1>("b", mesh);
    auto y = samurai::make_field<double, 1>("y", mesh);

    // init raw xtensor vectors
    auto xxx = xt::ones<double>({size});

    auto x_tensor = xt::ones<double>({size}) ; 
    auto b_tensor = xt::ones<double>({size}) ;
    xt::xarray<double> y_tensor = xt::ones<double>({size}) ;
    // init with c++ vectors
    std::vector<double> x_vector(size, 1.0); 
    std::vector<double> b_vector(size, 1.0) ;
    std::vector<double> y_vector(size, 1.0) ;    

// initialization
    samurai::for_each_cell(mesh,
                           [&](auto& cell)
                           {
                               x[cell] = 1.0;
			       b[cell] = 1.0;
			       y[cell] = 1.0;
                           });

	double a = 2.0 ;
// compute 
//
//
    auto start_samurai = std::chrono::high_resolution_clock::now() ; 
    samurai::for_each_cell(mesh,
                           [&](auto& cell)
                           {
                               y[cell] = a * x[cell] + b[cell] ; 
                           });
// y should be 4.0
    auto end_samurai = std::chrono::high_resolution_clock::now() ;
    auto duration_samurai = end_samurai - start_samurai;
    

    auto start_xtensor = std::chrono::high_resolution_clock::now() ;
    y_tensor = xt::eval(a * x_tensor + b_tensor) ; 
    auto end_xtensor = std::chrono::high_resolution_clock::now() ;
    auto duration_xtensor = end_xtensor - start_xtensor;


    auto start_vector = std::chrono::high_resolution_clock::now() ;
    for (int i = 0 ; i < size ; i++){
	    y_vector[i] = a * x_vector[i] + b_vector[i] ;
    }
    auto end_vector = std::chrono::high_resolution_clock::now() ;
    auto duration_vector = end_vector - start_vector;



    std::cout << " Time for Samurai : " << duration_samurai.count() << std::endl ; 
    std::cout << " Time for Xtensor : " << duration_xtensor.count() << std::endl ;

    std::cout << " Time for Vector  : " << duration_vector.count() << std::endl ;




//    save("test", "x.file", x, "field") ; 
//    save("test", "b.file", b, "field") ;
//    save("test", "y.file", y, "field") ;
	
    samurai::finalize();
    return 0;
}

