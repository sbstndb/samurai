// Samurai Python Bindings - MeshConfig class
//
// Bindings for samurai::mesh_config class
// Provides fluent interface for mesh configuration

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <samurai/mesh_config.hpp>

namespace py = pybind11;

// Type aliases for mesh_config with default template parameters
using MeshConfig1D = samurai::mesh_config<1>;
using MeshConfig2D = samurai::mesh_config<2>;
using MeshConfig3D = samurai::mesh_config<3>;

// Helper to bind MeshConfig methods that return *this for chaining
template <std::size_t dim, class Config>
void bind_mesh_config_common_methods(py::class_<Config>& cls)
{
    using namespace samurai;

    // Min level
    cls.def_property(
        "min_level",
        [](const Config& cfg)
        {
            return cfg.min_level();
        },
        [](Config& cfg, std::size_t level)
        {
            cfg.min_level(level);
            return cfg;
        },
        "Minimum refinement level (read/write, returns config for chaining)");

    // Max level
    cls.def_property(
        "max_level",
        [](const Config& cfg)
        {
            return cfg.max_level();
        },
        [](Config& cfg, std::size_t level)
        {
            cfg.max_level(level);
            return cfg;
        },
        "Maximum refinement level (read/write, returns config for chaining)");

    // Start level
    cls.def_property(
        "start_level",
        [](const Config& cfg)
        {
            return cfg.start_level();
        },
        [](Config& cfg, std::size_t level)
        {
            cfg.start_level(level);
            return cfg;
        },
        "Starting refinement level (read/write, returns config for chaining)");

    // Graduation width
    cls.def_property(
        "graduation_width",
        [](const Config& cfg)
        {
            return cfg.graduation_width();
        },
        [](Config& cfg, std::size_t width)
        {
            cfg.graduation_width(width);
            return cfg;
        },
        "Graduation width for AMR (read/write, returns config for chaining)");

    // Max stencil radius
    cls.def_property(
        "max_stencil_radius",
        [](const Config& cfg)
        {
            return cfg.max_stencil_radius();
        },
        [](Config& cfg, int radius)
        {
            cfg.max_stencil_radius(radius);
            return cfg;
        },
        "Maximum stencil radius (read/write, returns config for chaining)");

    // Max stencil size (derived from radius)
    cls.def_property(
        "max_stencil_size",
        [](const Config& cfg)
        {
            return cfg.max_stencil_size();
        },
        [](Config& cfg, int size)
        {
            cfg.max_stencil_size(size);
            return cfg;
        },
        "Maximum stencil size (read/write, returns config for chaining)");

    // Scaling factor
    cls.def_property(
        "scaling_factor",
        [](const Config& cfg)
        {
            return cfg.scaling_factor();
        },
        [](Config& cfg, double factor)
        {
            cfg.scaling_factor(factor);
            return cfg;
        },
        "Scaling factor for coordinates (read/write, returns config for chaining)");

    // Approx box tolerance
    cls.def_property(
        "approx_box_tol",
        [](const Config& cfg)
        {
            return cfg.approx_box_tol();
        },
        [](Config& cfg, double tol)
        {
            cfg.approx_box_tol(tol);
            return cfg;
        },
        "Approximation tolerance for box (read/write, returns config for chaining)");

    // Ghost width (read-only)
    cls.def_property_readonly("ghost_width", &Config::ghost_width, "Ghost width (read-only, computed from stencil)");

    // Periodic (scalar - set all directions)
    cls.def(
        "set_periodic",
        [](Config& cfg, bool periodic) -> Config&
        {
            cfg.periodic(periodic);
            return cfg;
        },
        py::arg("periodic"),
        "Set periodicity in all directions (returns config for chaining)");

    // Periodic (array - per direction)
    cls.def(
        "set_periodic_per_direction",
        [](Config& cfg, const std::array<bool, dim>& periodic) -> Config&
        {
            cfg.periodic(periodic);
            return cfg;
        },
        py::arg("periodic"),
        "Set periodicity per direction (returns config for chaining)");

    cls.def(
        "get_periodic",
        [](const Config& cfg, std::size_t i)
        {
            if (i >= dim)
            {
                throw std::out_of_range("Periodic index out of range");
            }
            return cfg.periodic(i);
        },
        py::arg("direction"),
        "Get periodicity in specific direction");

    // Disable minimal ghost width
    cls.def(
        "disable_minimal_ghost_width",
        [](Config& cfg) -> Config&
        {
            cfg.disable_minimal_ghost_width();
            return cfg;
        },
        "Disable minimal ghost width (returns config for chaining). "
        "Required for reconstruction and transfer functions.");

    // String representation
    cls.def("__repr__",
            [](const Config& cfg)
            {
                constexpr std::size_t d = Config::dim;
                std::ostringstream oss;
                oss << "MeshConfig" << d << "D(";
                oss << "min_level=" << cfg.min_level();
                oss << ", max_level=" << cfg.max_level();
                oss << ", start_level=" << cfg.start_level();
                oss << ", graduation_width=" << cfg.graduation_width();
                oss << ")";
                return oss.str();
            });

    cls.def("__str__",
            [](const Config& cfg)
            {
                constexpr std::size_t d = Config::dim;
                std::ostringstream oss;
                oss << "MeshConfig" << d << "D";
                oss << " [min=" << cfg.min_level();
                oss << ", max=" << cfg.max_level();
                oss << ", start=" << cfg.start_level();
                oss << "]";
                return oss.str();
            });
}

// Template function to bind MeshConfig for any dimension
template <std::size_t dim>
void bind_mesh_config(py::module_& m, const std::string& name)
{
    using Config = samurai::mesh_config<dim>;

    auto cls = py::class_<Config>(m, name.c_str(), R"pbdoc(
        Mesh configuration class with fluent interface.

        Used to configure mesh parameters for AMR/MR algorithms.

        Parameters
        ----------
        None - creates default configuration

        Examples
        --------
        >>> import samurai as sam
        >>> config = sam.MeshConfig2D()
        >>> config.min_level = 2
        >>> config.max_level = 6
        >>> # Or use method chaining
        >>> config = sam.MeshConfig2D().min_level(2).max_level(6)
        >>> # Or use constructor with defaults
        >>> config = sam.MeshConfig2D(min_level=2, max_level=6)

        Attributes
        ----------
        min_level : int
            Minimum refinement level (default: 0)
        max_level : int
            Maximum refinement level (default: 6)
        start_level : int
            Starting refinement level (default: 6)
        graduation_width : int
            AMR graduation width (default: depends on config)
        max_stencil_radius : int
            Maximum stencil radius
        max_stencil_size : int
            Maximum stencil size (2 * radius)
        scaling_factor : float
            Coordinate scaling factor
        approx_box_tol : float
            Approximation tolerance for box
        ghost_width : int (read-only)
            Ghost width, computed from stencil
    )pbdoc");

    // Default constructor
    cls.def(py::init<>(), "Create default mesh configuration");

    // Constructor with keyword arguments (Pythonic API)
    cls.def(py::init(
            [](std::size_t min_level, std::size_t max_level,
               std::size_t start_level, std::size_t graduation_width,
               int max_stencil_radius, double scaling_factor,
               double approx_box_tol, bool periodic,
               py::object periodic_per_direction,
               bool disable_minimal_ghost) -> Config
            {
                Config cfg;

                // Always set min_level (now has default of 0)
                cfg.min_level(min_level);

                // Always set max_level (has default of 6)
                cfg.max_level(max_level);

                // Only set if provided (use sentinel values)
                if (start_level != std::numeric_limits<std::size_t>::max()) {
                    cfg.start_level(start_level);
                }
                if (graduation_width != std::numeric_limits<std::size_t>::max()) {
                    cfg.graduation_width(graduation_width);
                }
                if (max_stencil_radius >= 0) {
                    cfg.max_stencil_radius(max_stencil_radius);
                }
                if (scaling_factor >= 0.0) {
                    cfg.scaling_factor(scaling_factor);
                }
                if (approx_box_tol >= 0.0) {
                    cfg.approx_box_tol(approx_box_tol);
                }

                // Handle periodicity
                if (!periodic_per_direction.is_none()) {
                    if constexpr (dim == 1) {
                        cfg.periodic(periodic);
                    } else {
                        auto per = periodic_per_direction.cast<std::array<bool, dim>>();
                        cfg.periodic(per);
                    }
                } else if (periodic) {
                    cfg.periodic(periodic);
                }

                // Handle ghost width
                if (disable_minimal_ghost) {
                    cfg.disable_minimal_ghost_width();
                }

                return cfg;
            }),
        py::arg("min_level") = 0,
        py::arg("max_level") = 6,
        py::arg("start_level") = std::numeric_limits<std::size_t>::max(),
        py::arg("graduation_width") = std::numeric_limits<std::size_t>::max(),
        py::arg("max_stencil_radius") = -1,
        py::arg("scaling_factor") = -1.0,
        py::arg("approx_box_tol") = -1.0,
        py::arg("periodic") = false,
        py::arg("periodic_per_direction") = py::none(),
        py::arg("disable_minimal_ghost_width") = false,
        R"pbdoc(
        Create mesh configuration with optional parameters.

        Parameters
        ----------
        min_level : int, optional
            Minimum refinement level (default: 0)
        max_level : int, optional
            Maximum refinement level (default: 6)
        start_level : int, optional
            Starting refinement level
        graduation_width : int, optional
            AMR graduation width
        max_stencil_radius : int, optional
            Maximum stencil radius
        scaling_factor : float, optional
            Coordinate scaling factor
        approx_box_tol : float, optional
            Approximation tolerance for box
        periodic : bool, optional
            Set periodicity in all directions (default: False)
        periodic_per_direction : list[bool], optional
            Set periodicity per direction (overrides periodic)
        disable_minimal_ghost_width : bool, optional
            Disable minimal ghost width (default: False)

        Examples
        --------
        >>> config = sam.MeshConfig2D(min_level=4, max_level=10)
        >>> config = sam.MeshConfig2D(min_level=2, max_level=8, periodic=True)
        >>> config = sam.MeshConfig2D(min_level=0, max_level=5,
        ...                            periodic_per_direction=[True, False])

        Notes
        -----
        Method chaining is still supported:
        >>> config = sam.MeshConfig2D().min_level(4).max_level(10)
        )pbdoc");

    // Bind all common methods
    bind_mesh_config_common_methods<dim>(cls);

    // Dimension property (read-only)
    cls.def_property_readonly(
        "dim",
        [](const Config&)
        {
            return Config::dim;
        },
        "Dimension of the mesh configuration");
}

// Helper template function to create MeshConfig with parameters
template <std::size_t dim>
samurai::mesh_config<dim> create_mesh_config_helper(
    std::size_t min_level,
    std::size_t max_level,
    std::size_t start_level,
    std::size_t graduation_width,
    int max_stencil_radius,
    double scaling_factor,
    double approx_box_tol,
    bool periodic,
    py::object periodic_per_direction,
    bool disable_minimal_ghost)
{
    samurai::mesh_config<dim> cfg;

    cfg.min_level(min_level);
    cfg.max_level(max_level);

    if (start_level != std::numeric_limits<std::size_t>::max()) {
        cfg.start_level(start_level);
    }
    if (graduation_width != std::numeric_limits<std::size_t>::max()) {
        cfg.graduation_width(graduation_width);
    }
    if (max_stencil_radius >= 0) {
        cfg.max_stencil_radius(max_stencil_radius);
    }
    if (scaling_factor >= 0.0) {
        cfg.scaling_factor(scaling_factor);
    }
    if (approx_box_tol >= 0.0) {
        cfg.approx_box_tol(approx_box_tol);
    }

    if (!periodic_per_direction.is_none()) {
        if constexpr (dim == 1) {
            cfg.periodic(periodic);
        } else {
            auto per = periodic_per_direction.cast<std::array<bool, dim>>();
            cfg.periodic(per);
        }
    } else if (periodic) {
        cfg.periodic(periodic);
    }

    if (disable_minimal_ghost) {
        cfg.disable_minimal_ghost_width();
    }

    return cfg;
}

// Module initialization function for MeshConfig bindings
void init_mesh_config_bindings(py::module_& m)
{
    // ============================================================
    // BREAKING CHANGE: No longer bind MeshConfig classes to main module
    // Users must use sam.config.MeshConfig1D, sam.config.MeshConfig2D, etc.
    // Or use the new factory: sam.config.make(dim, min_level=0, max_level=6, ...)
    // ============================================================

    // ============================================================
    // Create config submodule for organized API access
    // ============================================================
    py::module_ config = m.def_submodule(
        "config",
        "Configuration classes for Samurai AMR simulations\n\n"
        "Factory Functions:\n"
        "  make(dim, min_level=0, max_level=6, ...) - Create MeshConfig with explicit dimension\n\n"
        "Classes:\n"
        "  MeshConfig1D, MeshConfig2D, MeshConfig3D - Dimension-specific MeshConfig\n"
        "  MRAConfig - Multiresolution adaptation configuration\n\n"
        "Examples:\n"
        "    >>> import samurai_python as sam\n"
        "    >>> # Factory function (recommended)\n"
        "    >>> cfg = sam.config.make(2, min_level=4, max_level=8)\n"
        "    >>> # Direct class access\n"
        "    >>> cfg = sam.config.MeshConfig2D(min_level=2, max_level=8)\n");

    // Bind MeshConfig classes ONLY to config submodule (not to main module)
    bind_mesh_config<1>(config, "MeshConfig1D");
    bind_mesh_config<2>(config, "MeshConfig2D");
    bind_mesh_config<3>(config, "MeshConfig3D");

    // ============================================================
    // Factory function: sam.config.make(dim, min_level, max_level, ...)
    // Creates MeshConfig with explicit dimension parameter
    // ============================================================
    config.def("make",
        [](int dim,
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
            if (dim == 1) {
                auto cfg = create_mesh_config_helper<1>(
                    min_level, max_level, start_level, graduation_width,
                    max_stencil_radius, scaling_factor, approx_box_tol,
                    periodic, periodic_per_direction, disable_minimal_ghost_width);
                return py::cast(cfg);
            } else if (dim == 2) {
                auto cfg = create_mesh_config_helper<2>(
                    min_level, max_level, start_level, graduation_width,
                    max_stencil_radius, scaling_factor, approx_box_tol,
                    periodic, periodic_per_direction, disable_minimal_ghost_width);
                return py::cast(cfg);
            } else if (dim == 3) {
                auto cfg = create_mesh_config_helper<3>(
                    min_level, max_level, start_level, graduation_width,
                    max_stencil_radius, scaling_factor, approx_box_tol,
                    periodic, periodic_per_direction, disable_minimal_ghost_width);
                return py::cast(cfg);
            } else {
                throw std::runtime_error("Unsupported dimension: " + std::to_string(dim) + " (must be 1, 2, or 3)");
            }
        },
        py::arg("dim"),
        py::arg("min_level") = 0,
        py::arg("max_level") = 6,
        py::arg("start_level") = std::numeric_limits<std::size_t>::max(),
        py::arg("graduation_width") = std::numeric_limits<std::size_t>::max(),
        py::arg("max_stencil_radius") = -1,
        py::arg("scaling_factor") = -1.0,
        py::arg("approx_box_tol") = -1.0,
        py::arg("periodic") = false,
        py::arg("periodic_per_direction") = py::none(),
        py::arg("disable_minimal_ghost_width") = false,
        R"pbdoc(
        Create a MeshConfig by specifying dimension explicitly.

        Parameters
        ----------
        dim : int
            Spatial dimension (1, 2, or 3)
        min_level : int, optional
            Minimum refinement level (default: 0)
        max_level : int, optional
            Maximum refinement level (default: 6)
        start_level : int, optional
            Starting refinement level
        graduation_width : int, optional
            AMR graduation width
        max_stencil_radius : int, optional
            Maximum stencil radius
        scaling_factor : float, optional
            Coordinate scaling factor
        approx_box_tol : float, optional
            Approximation tolerance for box
        periodic : bool, optional
            Set periodicity in all directions (default: False)
        periodic_per_direction : list[bool], optional
            Set periodicity per direction (overrides periodic)
        disable_minimal_ghost_width : bool, optional
            Disable minimal ghost width (default: False)

        Returns
        -------
        MeshConfig
            Dimension-specific MeshConfig object (MeshConfig1D, MeshConfig2D, or MeshConfig3D)

        Examples
        --------
        >>> import samurai_python as sam
        >>> config = sam.config.make(2, min_level=4, max_level=8)
        >>> config = sam.config.make(2, min_level=2, max_level=6, periodic=True)
        >>> config = sam.config.make(3, min_level=0, max_level=5,
        ...                          periodic_per_direction=[True, False, True])
    )pbdoc");

    // Note: MRAConfig will be added in init_mra_config_bindings()
    // since it's initialized after this function
}
