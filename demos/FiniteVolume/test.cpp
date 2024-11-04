// Copyright 2018-2024 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause
#include <CLI/CLI.hpp>

#include <samurai/hdf5.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/schemes/fv.hpp>
#include <samurai/samurai.hpp>

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

    // Simulation parameters
    double left_box  = -1;
    double right_box = 1;

//    double dt            = Tf / 100;
//    bool explicit_scheme = false;
//    double cfl           = 0.95;

    // Multiresolution parameters
    std::size_t min_level = 3;
    std::size_t max_level = 3;


    //--------------------//
    // Problem definition //
    //--------------------//

    point_t box_corner1, box_corner2;
    box_corner1.fill(left_box);
    box_corner2.fill(right_box);
    Box box(box_corner1, box_corner2);
    samurai::MRMesh<Config> mesh{box, min_level, max_level};

    auto x = samurai::make_field<double, 1>("x", mesh);
    auto b = samurai::make_field<double, 1>("b", mesh);
    auto y = samurai::make_field<double, 1>("y", mesh);

// initialization
    samurai::for_each_cell(mesh,
                           [&](auto& cell)
                           {
                               x[cell] = 1.0;
			       b[cell] = 2.0;
			       y[cell] = -1.0;
                           });

	double a = 2.0 ;
// compute
    samurai::for_each_cell(mesh,
                           [&](auto& cell)
                           {
                               y[cell] = a * x[cell] + b[cell] ; 
                           });
// y should be 4.0


	save("test", "x.file", x, "field") ; 
        save("test", "b.file", b, "field") ;
        save("test", "y.file", y, "field") ;
	



    samurai::finalize();
    return 0;
}
