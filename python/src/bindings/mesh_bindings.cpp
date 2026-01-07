// Samurai Python Bindings - MRMesh class
//
// Bindings for samurai::MRMesh class (Multiresolution Mesh)
// Uses the recommended samurai::mra::make_mesh() factory function

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <samurai/box.hpp>
#include <samurai/domain_builder.hpp>
#include <samurai/mesh_config.hpp>
#include <samurai/mr/mesh.hpp>

namespace py = pybind11;

// Helper to bind common Mesh_base methods for any mesh type
template <class Mesh>
void bind_mesh_base_common_methods(py::class_<Mesh>& cls)
{
    using namespace samurai;

    // nb_cells property (total cells) and method (cells per level)
    // Note: Can't have both property and method with same name in pybind11
    // So we use nb_cells as property and nb_cells_at_level as method
    cls.def_property_readonly(
        "nb_cells",
        [](const Mesh& mesh) -> std::size_t
        {
            return mesh.nb_cells();
        },
        "Total number of cells in the mesh");

    cls.def(
        "nb_cells_at_level",
        [](const Mesh& mesh, std::size_t level) -> std::size_t
        {
            return mesh.nb_cells(level);
        },
        py::arg("level"),
        "Number of cells at a given refinement level");

    // Level properties
    cls.def_property(
        "min_level",
        [](const Mesh& mesh) -> std::size_t
        {
            return mesh.min_level();
        },
        [](Mesh& mesh, std::size_t level) -> Mesh&
        {
            mesh.min_level() = level;
            return mesh;
        },
        "Minimum refinement level (read/write)");

    cls.def_property(
        "max_level",
        [](const Mesh& mesh) -> std::size_t
        {
            return mesh.max_level();
        },
        [](Mesh& mesh, std::size_t level) -> Mesh&
        {
            mesh.max_level() = level;
            return mesh;
        },
        "Maximum refinement level (read/write)");

    // Configuration properties
    cls.def_property_readonly("graduation_width", &Mesh::graduation_width, "AMR graduation width");

    cls.def_property_readonly("ghost_width", &Mesh::ghost_width, "Ghost width (for stencil operations)");

    cls.def_property_readonly("max_stencil_radius", &Mesh::max_stencil_radius, "Maximum stencil radius");

    // Cell lengths
    cls.def("cell_length", &Mesh::cell_length, py::arg("level"), "Length of a cell at given refinement level");

    cls.def_property_readonly("min_cell_length", &Mesh::min_cell_length, "Minimum cell length in the mesh");

    // Periodicity
    cls.def_property_readonly(
        "periodic",
        [](const Mesh& mesh) -> bool
        {
            return mesh.is_periodic();
        },
        "Check if mesh is periodic in any direction");

    cls.def(
        "is_periodic",
        [](const Mesh& mesh, std::size_t d) -> bool
        {
            return mesh.is_periodic(d);
        },
        py::arg("direction"),
        "Check if mesh is periodic in a specific direction");

    cls.def_property_readonly("periodicity", &Mesh::periodicity, "Array of periodicity flags for each direction");

    // String representation
    cls.def("__repr__",
            [](const Mesh& mesh)
            {
                std::ostringstream oss;
                oss << "MRMesh" << Mesh::dim << "D(";
                oss << "min_level=" << mesh.min_level();
                oss << ", max_level=" << mesh.max_level();
                oss << ", nb_cells=" << mesh.nb_cells();
                oss << ")";
                return oss.str();
            });

    cls.def("__str__",
            [](const Mesh& mesh)
            {
                std::ostringstream oss;
                oss << "MRMesh" << Mesh::dim << "D";
                oss << " [L" << mesh.min_level() << "-" << mesh.max_level() << "]";
                oss << " [" << mesh.nb_cells() << " cells]";
                return oss.str();
            });
}

// Template function to bind MRMesh for a specific dimension
// We use auto return type from make_mesh() to handle the complete_mesh_config wrapper
template <std::size_t dim>
void bind_mr_mesh(py::module_& m, const std::string& name)
{
    using Box    = samurai::Box<double, dim>;
    using Config = samurai::mesh_config<dim>;

    // The make_mesh function returns MRMesh<complete_mesh_config<Config, MRMeshId>>
    // We need to bind this type directly
    using CompleteConfig = samurai::complete_mesh_config<Config, samurai::MRMeshId>;
    using Mesh           = samurai::MRMesh<CompleteConfig>;

    auto cls = py::class_<Mesh>(m, name.c_str(), R"pbdoc(
        Multiresolution Mesh (MRMesh)

        Adaptive mesh refinement mesh with multiresolution analysis capabilities.

        Note: Creating MRMesh is computationally intensive. Use small level ranges for testing.

        Examples
        --------
        >>> import samurai as sam
        >>> box = sam.Box1D([0.], [1.])
        >>> config = sam.MeshConfig1D()
        >>> config.min_level = 0
        >>> config.max_level = 1
        >>> mesh = sam.MRMesh1D(box, config)

        Creating a mesh with obstacles:

        >>> domain = sam.DomainBuilder2D([-1., -1.], [1., 1.])
        >>> domain.remove([0.0, 0.0], [0.4, 0.4])  # Create obstacle
        >>> mesh = sam.MRMesh2D(domain, config)

        Attributes
        ----------
        min_level : int
            Minimum refinement level
        max_level : int
            Maximum refinement level
        nb_cells : int
            Total number of cells
        graduation_width : int
            AMR graduation width
        ghost_width : int
            Ghost width for stencil operations
    )pbdoc");

    // Constructor using samurai::mra::make_mesh factory function with Box
    // This is the RECOMMENDED approach that handles config conversion properly
    cls.def(py::init(
                [](const Box& box, const Config& user_config)
                {
                    // Use the official factory function that:
                    // 1. Wraps mesh_config in complete_mesh_config<Config, MRMeshId>
                    // 2. Calls parse_args() and sets start_level
                    // 3. Returns the properly constructed MRMesh
                    return samurai::mra::make_mesh(box, user_config);
                }),
            py::arg("box"),
            py::arg("config"),
            "Create MRMesh from Box and MeshConfig (using mra::make_mesh factory)");

    // Constructor using samurai::mra::make_mesh factory function with DomainBuilder
    // Allows creating meshes with holes/obstacles
    cls.def(py::init(
                [](const samurai::DomainBuilder<dim>& domain_builder, const Config& user_config)
                {
                    // Use the official factory function with DomainBuilder
                    // Note: Periodic BC and MPI are NOT supported with DomainBuilder
                    return samurai::mra::make_mesh(domain_builder, user_config);
                }),
            py::arg("domain_builder"),
            py::arg("config"),
            "Create MRMesh from DomainBuilder with obstacles (using mra::make_mesh factory)");

    // Bind all common methods from Mesh_base
    bind_mesh_base_common_methods<Mesh>(cls);

    // Dimension property (read-only)
    cls.def_property_readonly(
        "dim",
        [](const Mesh&)
        {
            return dim;
        },
        "Dimension of the mesh");
}

// Module initialization function for MRMesh bindings
void init_mesh_bindings(py::module_& m)
{
    // ============================================================
    // BREAKING CHANGE: No longer bind MRMesh classes to main module
    // Users must use sam.mesh.MRMesh1D, sam.mesh.MRMesh2D, etc.
    // Or use the new factory: sam.mesh.make(box, config=None, min_level=None, ...)
    // ============================================================

    // ============================================================
    // Create mesh submodule for organized API access
    // ============================================================
    py::module_ mesh_module = m.def_submodule("mesh",
        "Mesh classes for Samurai AMR simulations\n\n"
        "Factory Functions:\n"
        "  make(box_or_domain, config=None, min_level=None, max_level=None, ...) - Create MRMesh\n\n"
        "Classes:\n"
        "  MRMesh1D, MRMesh2D, MRMesh3D - Dimension-specific MRMesh\n\n"
        "Examples:\n"
        "    >>> import samurai_python as sam\n"
        "    >>> # Factory function with inline config (recommended)\n"
        "    >>> mesh = sam.mesh.make(box, min_level=4, max_level=8)\n"
        "    >>> # Factory function with explicit config\n"
        "    >>> config = sam.config.make(2, min_level=4, max_level=8)\n"
        "    >>> mesh = sam.mesh.make(box, config)\n"
        "    >>> # Direct class access\n"
        "    >>> mesh = sam.mesh.MRMesh2D(box, config)\n");

    // Bind MRMesh classes ONLY to mesh submodule (not to main module)
    bind_mr_mesh<1>(mesh_module, "MRMesh1D");
    bind_mr_mesh<2>(mesh_module, "MRMesh2D");
    bind_mr_mesh<3>(mesh_module, "MRMesh3D");

    // ============================================================
    // Helper template to create MRMesh with inline config parameters
    // ============================================================
    auto make_mesh_with_config_1d = [](
        const samurai::Box<double, 1>& box,
        std::size_t min_level,
        std::size_t max_level,
        std::size_t start_level,
        std::size_t graduation_width,
        int max_stencil_radius,
        double scaling_factor,
        double approx_box_tol,
        bool periodic,
        py::object periodic_per_direction,
        bool disable_minimal_ghost_width) -> py::object
    {
        using Config = samurai::mesh_config<1>;
        Config cfg;
        cfg.min_level(min_level);
        cfg.max_level(max_level);
        if (start_level != std::numeric_limits<std::size_t>::max()) cfg.start_level(start_level);
        if (graduation_width != std::numeric_limits<std::size_t>::max()) cfg.graduation_width(graduation_width);
        if (max_stencil_radius >= 0) cfg.max_stencil_radius(max_stencil_radius);
        if (scaling_factor >= 0.0) cfg.scaling_factor(scaling_factor);
        if (approx_box_tol >= 0.0) cfg.approx_box_tol(approx_box_tol);
        if (!periodic_per_direction.is_none()) {
            cfg.periodic(periodic);  // 1D only uses scalar
        } else if (periodic) {
            cfg.periodic(periodic);
        }
        if (disable_minimal_ghost_width) cfg.disable_minimal_ghost_width();

        auto mesh = samurai::mra::make_mesh(box, cfg);
        return py::cast(mesh);
    };

    auto make_mesh_with_config_2d = [](
        const samurai::Box<double, 2>& box,
        std::size_t min_level,
        std::size_t max_level,
        std::size_t start_level,
        std::size_t graduation_width,
        int max_stencil_radius,
        double scaling_factor,
        double approx_box_tol,
        bool periodic,
        py::object periodic_per_direction,
        bool disable_minimal_ghost_width) -> py::object
    {
        using Config = samurai::mesh_config<2>;
        Config cfg;
        cfg.min_level(min_level);
        cfg.max_level(max_level);
        if (start_level != std::numeric_limits<std::size_t>::max()) cfg.start_level(start_level);
        if (graduation_width != std::numeric_limits<std::size_t>::max()) cfg.graduation_width(graduation_width);
        if (max_stencil_radius >= 0) cfg.max_stencil_radius(max_stencil_radius);
        if (scaling_factor >= 0.0) cfg.scaling_factor(scaling_factor);
        if (approx_box_tol >= 0.0) cfg.approx_box_tol(approx_box_tol);
        if (!periodic_per_direction.is_none()) {
            auto per = periodic_per_direction.cast<std::array<bool, 2>>();
            cfg.periodic(per);
        } else if (periodic) {
            cfg.periodic(periodic);
        }
        if (disable_minimal_ghost_width) cfg.disable_minimal_ghost_width();

        auto mesh = samurai::mra::make_mesh(box, cfg);
        return py::cast(mesh);
    };

    auto make_mesh_with_config_3d = [](
        const samurai::Box<double, 3>& box,
        std::size_t min_level,
        std::size_t max_level,
        std::size_t start_level,
        std::size_t graduation_width,
        int max_stencil_radius,
        double scaling_factor,
        double approx_box_tol,
        bool periodic,
        py::object periodic_per_direction,
        bool disable_minimal_ghost_width) -> py::object
    {
        using Config = samurai::mesh_config<3>;
        Config cfg;
        cfg.min_level(min_level);
        cfg.max_level(max_level);
        if (start_level != std::numeric_limits<std::size_t>::max()) cfg.start_level(start_level);
        if (graduation_width != std::numeric_limits<std::size_t>::max()) cfg.graduation_width(graduation_width);
        if (max_stencil_radius >= 0) cfg.max_stencil_radius(max_stencil_radius);
        if (scaling_factor >= 0.0) cfg.scaling_factor(scaling_factor);
        if (approx_box_tol >= 0.0) cfg.approx_box_tol(approx_box_tol);
        if (!periodic_per_direction.is_none()) {
            auto per = periodic_per_direction.cast<std::array<bool, 3>>();
            cfg.periodic(per);
        } else if (periodic) {
            cfg.periodic(periodic);
        }
        if (disable_minimal_ghost_width) cfg.disable_minimal_ghost_width();

        auto mesh = samurai::mra::make_mesh(box, cfg);
        return py::cast(mesh);
    };

    // ============================================================
    // Factory function: sam.mesh.make(box_or_domain, config=None, min_level=None, ...)
    // Two calling conventions:
    //   1. With explicit config: mesh = sam.mesh.make(box, config)
    //   2. With inline config: mesh = sam.mesh.make(box, min_level=4, max_level=8, ...)
    // ============================================================
    mesh_module.def("make",
        [&](const py::object& box_or_domain,
           const py::object& config_obj,
           std::size_t min_level,
           std::size_t max_level,
           std::size_t start_level,
           std::size_t graduation_width,
           int max_stencil_radius,
           double scaling_factor,
           double approx_box_tol,
           bool periodic,
           py::object periodic_per_direction,
           bool disable_minimal_ghost_width,
           py::kwargs kwargs) -> py::object
        {
            // Case 1: Config object provided
            if (!config_obj.is_none())
            {
                // Try Box1D + MeshConfig1D
                try {
                    using Box1D = samurai::Box<double, 1>;
                    using Config1D = samurai::mesh_config<1>;
                    auto box = py::cast<Box1D>(box_or_domain);
                    auto cfg = py::cast<Config1D>(config_obj);
                    auto mesh = samurai::mra::make_mesh(box, cfg);
                    return py::cast(mesh);
                } catch (...) {}

                // Try Box2D + MeshConfig2D
                try {
                    using Box2D = samurai::Box<double, 2>;
                    using Config2D = samurai::mesh_config<2>;
                    auto box = py::cast<Box2D>(box_or_domain);
                    auto cfg = py::cast<Config2D>(config_obj);
                    auto mesh = samurai::mra::make_mesh(box, cfg);
                    return py::cast(mesh);
                } catch (...) {}

                // Try Box3D + MeshConfig3D
                try {
                    using Box3D = samurai::Box<double, 3>;
                    using Config3D = samurai::mesh_config<3>;
                    auto box = py::cast<Box3D>(box_or_domain);
                    auto cfg = py::cast<Config3D>(config_obj);
                    auto mesh = samurai::mra::make_mesh(box, cfg);
                    return py::cast(mesh);
                } catch (...) {}

                // Try DomainBuilder1D + MeshConfig1D
                try {
                    using DomainBuilder1D = samurai::DomainBuilder<1>;
                    using Config1D = samurai::mesh_config<1>;
                    auto domain = py::cast<DomainBuilder1D>(box_or_domain);
                    auto cfg = py::cast<Config1D>(config_obj);
                    auto mesh = samurai::mra::make_mesh(domain, cfg);
                    return py::cast(mesh);
                } catch (...) {}

                // Try DomainBuilder2D + MeshConfig2D
                try {
                    using DomainBuilder2D = samurai::DomainBuilder<2>;
                    using Config2D = samurai::mesh_config<2>;
                    auto domain = py::cast<DomainBuilder2D>(box_or_domain);
                    auto cfg = py::cast<Config2D>(config_obj);
                    auto mesh = samurai::mra::make_mesh(domain, cfg);
                    return py::cast(mesh);
                } catch (...) {}

                // Try DomainBuilder3D + MeshConfig3D
                try {
                    using DomainBuilder3D = samurai::DomainBuilder<3>;
                    using Config3D = samurai::mesh_config<3>;
                    auto domain = py::cast<DomainBuilder3D>(box_or_domain);
                    auto cfg = py::cast<Config3D>(config_obj);
                    auto mesh = samurai::mra::make_mesh(domain, cfg);
                    return py::cast(mesh);
                } catch (...) {}

                throw std::runtime_error("Box/Domain and Config dimension mismatch or unsupported types");
            }

            // Case 2: Inline config parameters
            // Detect dimension from box_or_domain by trying each type
            // Try Box1D
            try {
                using Box1D = samurai::Box<double, 1>;
                auto box = py::cast<Box1D>(box_or_domain);
                return make_mesh_with_config_1d(box, min_level, max_level, start_level,
                    graduation_width, max_stencil_radius, scaling_factor, approx_box_tol,
                    periodic, periodic_per_direction, disable_minimal_ghost_width);
            } catch (...) {}

            // Try Box2D
            try {
                using Box2D = samurai::Box<double, 2>;
                auto box = py::cast<Box2D>(box_or_domain);
                return make_mesh_with_config_2d(box, min_level, max_level, start_level,
                    graduation_width, max_stencil_radius, scaling_factor, approx_box_tol,
                    periodic, periodic_per_direction, disable_minimal_ghost_width);
            } catch (...) {}

            // Try Box3D
            try {
                using Box3D = samurai::Box<double, 3>;
                auto box = py::cast<Box3D>(box_or_domain);
                return make_mesh_with_config_3d(box, min_level, max_level, start_level,
                    graduation_width, max_stencil_radius, scaling_factor, approx_box_tol,
                    periodic, periodic_per_direction, disable_minimal_ghost_width);
            } catch (...) {}

            // Try DomainBuilder1D
            try {
                using DomainBuilder1D = samurai::DomainBuilder<1>;
                using Config1D = samurai::mesh_config<1>;
                auto domain = py::cast<DomainBuilder1D>(box_or_domain);
                Config1D cfg;
                cfg.min_level(min_level);
                cfg.max_level(max_level);
                if (start_level != std::numeric_limits<std::size_t>::max()) cfg.start_level(start_level);
                if (graduation_width != std::numeric_limits<std::size_t>::max()) cfg.graduation_width(graduation_width);
                if (max_stencil_radius >= 0) cfg.max_stencil_radius(max_stencil_radius);
                if (scaling_factor >= 0.0) cfg.scaling_factor(scaling_factor);
                if (approx_box_tol >= 0.0) cfg.approx_box_tol(approx_box_tol);
                if (!periodic_per_direction.is_none()) {
                    cfg.periodic(periodic);  // 1D only uses scalar
                } else if (periodic) {
                    cfg.periodic(periodic);
                }
                if (disable_minimal_ghost_width) cfg.disable_minimal_ghost_width();
                auto mesh = samurai::mra::make_mesh(domain, cfg);
                return py::cast(mesh);
            } catch (...) {}

            // Try DomainBuilder2D
            try {
                using DomainBuilder2D = samurai::DomainBuilder<2>;
                using Config2D = samurai::mesh_config<2>;
                auto domain = py::cast<DomainBuilder2D>(box_or_domain);
                Config2D cfg;
                cfg.min_level(min_level);
                cfg.max_level(max_level);
                if (start_level != std::numeric_limits<std::size_t>::max()) cfg.start_level(start_level);
                if (graduation_width != std::numeric_limits<std::size_t>::max()) cfg.graduation_width(graduation_width);
                if (max_stencil_radius >= 0) cfg.max_stencil_radius(max_stencil_radius);
                if (scaling_factor >= 0.0) cfg.scaling_factor(scaling_factor);
                if (approx_box_tol >= 0.0) cfg.approx_box_tol(approx_box_tol);
                if (!periodic_per_direction.is_none()) {
                    auto per = periodic_per_direction.cast<std::array<bool, 2>>();
                    cfg.periodic(per);
                } else if (periodic) {
                    cfg.periodic(periodic);
                }
                if (disable_minimal_ghost_width) cfg.disable_minimal_ghost_width();
                auto mesh = samurai::mra::make_mesh(domain, cfg);
                return py::cast(mesh);
            } catch (...) {}

            // Try DomainBuilder3D
            try {
                using DomainBuilder3D = samurai::DomainBuilder<3>;
                using Config3D = samurai::mesh_config<3>;
                auto domain = py::cast<DomainBuilder3D>(box_or_domain);
                Config3D cfg;
                cfg.min_level(min_level);
                cfg.max_level(max_level);
                if (start_level != std::numeric_limits<std::size_t>::max()) cfg.start_level(start_level);
                if (graduation_width != std::numeric_limits<std::size_t>::max()) cfg.graduation_width(graduation_width);
                if (max_stencil_radius >= 0) cfg.max_stencil_radius(max_stencil_radius);
                if (scaling_factor >= 0.0) cfg.scaling_factor(scaling_factor);
                if (approx_box_tol >= 0.0) cfg.approx_box_tol(approx_box_tol);
                if (!periodic_per_direction.is_none()) {
                    auto per = periodic_per_direction.cast<std::array<bool, 3>>();
                    cfg.periodic(per);
                } else if (periodic) {
                    cfg.periodic(periodic);
                }
                if (disable_minimal_ghost_width) cfg.disable_minimal_ghost_width();
                auto mesh = samurai::mra::make_mesh(domain, cfg);
                return py::cast(mesh);
            } catch (...) {}

            throw std::runtime_error("Unsupported box_or_domain type (expected Box or DomainBuilder)");
        },
        py::arg("box_or_domain"),
        py::arg("config") = py::none(),
        py::arg("min_level") = std::numeric_limits<std::size_t>::max(),
        py::arg("max_level") = std::numeric_limits<std::size_t>::max(),
        py::arg("start_level") = std::numeric_limits<std::size_t>::max(),
        py::arg("graduation_width") = std::numeric_limits<std::size_t>::max(),
        py::arg("max_stencil_radius") = -1,
        py::arg("scaling_factor") = -1.0,
        py::arg("approx_box_tol") = -1.0,
        py::arg("periodic") = false,
        py::arg("periodic_per_direction") = py::none(),
        py::arg("disable_minimal_ghost_width") = false,
        R"pbdoc(
        Create an MRMesh from Box/Domain and optional config.

        Two calling conventions:

        1. With explicit config object:
           mesh = sam.mesh.make(box, config)

        2. With inline config parameters (recommended):
           mesh = sam.mesh.make(box, min_level=4, max_level=8)

        Parameters
        ----------
        box_or_domain : Box or DomainBuilder
            Geometric domain definition
        config : MeshConfig, optional
            Mesh configuration object (if provided, inline params are ignored)
        min_level : int, optional
            Minimum refinement level (if config not provided)
        max_level : int, optional
            Maximum refinement level (if config not provided)
        start_level : int, optional
            Starting refinement level (if config not provided)
        graduation_width : int, optional
            AMR graduation width (if config not provided)
        max_stencil_radius : int, optional
            Maximum stencil radius (if config not provided)
        scaling_factor : float, optional
            Coordinate scaling factor (if config not provided)
        approx_box_tol : float, optional
            Approximation tolerance for box (if config not provided)
        periodic : bool, optional
            Set periodicity in all directions (if config not provided)
        periodic_per_direction : list[bool], optional
            Set periodicity per direction (if config not provided)
        disable_minimal_ghost_width : bool, optional
            Disable minimal ghost width (if config not provided)

        Returns
        -------
        MRMesh
            Dimension-specific MRMesh object (MRMesh1D, MRMesh2D, or MRMesh3D)

        Examples
        --------
        >>> import samurai_python as sam
        >>> box = sam.geometry.box([0., 0.], [1., 1.])
        >>>
        >>> # Method 1: With inline config (recommended)
        >>> mesh = sam.mesh.make(box, min_level=4, max_level=8)
        >>>
        >>> # Method 2: With explicit config
        >>> config = sam.config.make(2, min_level=4, max_level=8)
        >>> mesh = sam.mesh.make(box, config)
        >>>
        >>> # Method 3: With domain builder
        >>> domain = sam.geometry.domain_builder([-1., -1.], [1., 1.])
        >>> mesh = sam.mesh.make(domain, min_level=2, max_level=6)
    )pbdoc");
}
