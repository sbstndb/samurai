#include <iostream>
#include "csir.hpp"

// Helper function to create a simple cube mesh
csir::CSIR_Level_3D create_cube_mesh(std::size_t level, int min_coord, int max_coord) {
    csir::CSIR_Level_3D mesh_3d;
    mesh_3d.level = level;

    // Create a 2D slice
    csir::CSIR_Level slice_2d;
    slice_2d.level = level;
    slice_2d.intervals_ptr.push_back(0);
    for (int y = min_coord; y < max_coord; ++y) {
        slice_2d.y_coords.push_back(y);
        slice_2d.intervals.push_back({min_coord, max_coord});
        slice_2d.intervals_ptr.push_back(slice_2d.intervals.size());
    }

    // Stack the slices to form a cube
    for (int z = min_coord; z < max_coord; ++z) {
        mesh_3d.slices[z] = slice_2d;
    }

    return mesh_3d;
}

int main() {
    std::cout << "--- CSIR Standalone 3D Demo ---" << std::endl;

    // 1. Create a 10x10x10 cube mesh at level 2
    std::cout << "\n1. Creating Mesh A (10x10x10 cube) at Level 2..." << std::endl;
    auto mesh_A_lvl2 = create_cube_mesh(2, 0, 10);
    csir::print_level_3d(mesh_A_lvl2);

    // 2. Create a 5x5x5 cube mesh at level 3, offset to the corner
    std::cout << "\n2. Creating Mesh B (5x5x5 cube, offset) at Level 3..." << std::endl;
    auto mesh_B_lvl3 = create_cube_mesh(3, 15, 20);
    csir::print_level_3d(mesh_B_lvl3);

    // 3. Project mesh A from level 2 to level 3
    // A 10x10x10 cube at level 2 should become a 20x20x20 cube at level 3
    std::cout << "\n3. Projecting Mesh A from Level 2 to Level 3..." << std::endl;
    auto mesh_A_lvl3 = csir::project_to_level_3d(mesh_A_lvl2, 3);
    csir::print_level_3d(mesh_A_lvl3);

    // 4. Intersect the two meshes at level 3
    // We expect an intersection where the 20x20x20 cube (0,0,0 to 20,20,20)
    // overlaps with the 5x5x5 cube (15,15,15 to 20,20,20)
    std::cout << "\n4. Intersecting (Projected Mesh A) and (Mesh B)..." << std::endl;
    auto intersection_result = csir::intersection_3d(mesh_A_lvl3, mesh_B_lvl3);
    csir::print_level_3d(intersection_result);

    std::cout << "\nDemo finished. The result should be a 5x5x5 cube from (15,15,15) to (20,20,20)." << std::endl;

    return 0;
}
