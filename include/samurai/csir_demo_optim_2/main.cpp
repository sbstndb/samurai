#include <iostream>
#include "csir.hpp"

// Helper function to create a simple square/rectangular mesh
csir::CSIR_Level create_square_mesh(std::size_t level, int min_coord_x, int max_coord_x, int min_coord_y, int max_coord_y) {
    csir::CSIR_Level mesh;
    mesh.level = level;
    mesh.intervals_ptr.push_back(0);

    for (int y = min_coord_y; y < max_coord_y; ++y) {
        mesh.y_coords.push_back(y);
        mesh.intervals.push_back({min_coord_x, max_coord_x});
        mesh.intervals_ptr.push_back(mesh.intervals.size());
    }
    return mesh;
}

int main() {
    std::cout << "--- CSIR Standalone Demo: Complex Multi-Level Operations ---" << std::endl;

    // =========== Part 1: Create a frame (ring) at Level 4 ===========
    std::cout << "\n1. Creating a frame at Level 4..." << std::endl;
    auto outer_square_lvl4 = create_square_mesh(4, 0, 20, 0, 20);
    auto hole_lvl4 = create_square_mesh(4, 5, 15, 5, 15);
    auto frame_lvl4 = csir::difference(outer_square_lvl4, hole_lvl4);
    std::cout << "--- Frame at Level 4 (20x20 with a 10x10 hole) ---" << std::endl;
    csir::print_level(frame_lvl4);

    // =========== Part 2: Create a cross shape at Level 5 ===========
    std::cout << "\n2. Creating a cross shape at Level 5..." << std::endl;
    // Note: The coordinates are chosen to fit inside the future projected hole
    auto vertical_bar_lvl5 = create_square_mesh(5, 18, 22, 10, 30);
    auto horizontal_bar_lvl5 = create_square_mesh(5, 10, 30, 18, 22);
    auto cross_lvl5 = csir::union_(vertical_bar_lvl5, horizontal_bar_lvl5);
    std::cout << "--- Cross at Level 5 ---" << std::endl;
    csir::print_level(cross_lvl5);

    // =========== Part 3: Project the frame to Level 5 ===========
    std::cout << "\n3. Projecting the frame from Level 4 to Level 5..." << std::endl;
    // The 20x20 frame with a 10x10 hole should become a 40x40 frame with a 20x20 hole
    auto frame_lvl5 = csir::project_to_level(frame_lvl4, 5);
    std::cout << "--- Frame at Level 5 (40x40 with a 20x20 hole) ---" << std::endl;
    csir::print_level(frame_lvl5);

    // =========== Part 4: Final Assembly - Union of Frame and Cross ===========
    std::cout << "\n4. Final Assembly: Union of the projected frame and the cross..." << std::endl;
    auto final_assembly = csir::union_(frame_lvl5, cross_lvl5);
    std::cout << "--- FINAL RESULT ---" << std::endl;
    csir::print_level(final_assembly);

    std::cout << "\nDemo finished. The result should be a large frame with a cross in its center." << std::endl;

    return 0;
}
