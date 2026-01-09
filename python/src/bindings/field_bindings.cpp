// Samurai Python Bindings - ScalarField and VectorField classes
//
// Bindings for samurai::ScalarField and samurai::VectorField classes
// Uses NumPy buffer protocol for zero-copy interoperability

#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <samurai/algorithm.hpp>
#include <samurai/box.hpp>
#include <samurai/cell.hpp>
#include <samurai/field.hpp>
#include <samurai/io/hdf5.hpp>
#include <samurai/io/restart.hpp>
#include <samurai/mesh_config.hpp>
#include <samurai/mr/mesh.hpp>

namespace py = pybind11;

// Type aliases matching MRMesh bindings
// Use the default interval type from Samurai
using default_interval = samurai::Interval<double, std::size_t>;

template <std::size_t dim>
using MRMesh = samurai::MRMesh<samurai::complete_mesh_config<samurai::mesh_config<dim>, samurai::MRMeshId>>;

template <std::size_t dim>
using ScalarField = samurai::ScalarField<MRMesh<dim>, double>;

template <std::size_t dim, std::size_t n_comp, bool SOA = false>
using VectorField = samurai::VectorField<MRMesh<dim>, double, n_comp, SOA>;

template <std::size_t dim>
using Cell = samurai::Cell<dim, default_interval>;

// ============================================================
// Reduction helpers for ScalarField
// ============================================================

template <std::size_t dim>
double field_sum(const ScalarField<dim>& field)
{
    return xt::sum(field.array())();
}

template <std::size_t dim>
double field_mean(const ScalarField<dim>& field)
{
    return xt::mean(field.array())();
}

template <std::size_t dim>
double field_max(const ScalarField<dim>& field)
{
    return xt::amax(field.array())();
}

template <std::size_t dim>
double field_min(const ScalarField<dim>& field)
{
    return xt::amin(field.array())();
}

// ============================================================
// Reduction helpers for VectorField
// ============================================================

// Sum over all elements (all components)
template <std::size_t dim, std::size_t n_comp>
double vectorfield_sum_all(const VectorField<dim, n_comp, false>& field)
{
    return xt::sum(field.array())();
}

template <std::size_t dim, std::size_t n_comp>
double vectorfield_sum_all(const VectorField<dim, n_comp, true>& field)
{
    return xt::sum(field.array())();
}

// Sum by component (returns numpy array)
template <std::size_t dim, std::size_t n_comp>
py::array_t<double> vectorfield_sum_by_component(const VectorField<dim, n_comp, false>& field)
{
    // AOS layout: shape (n_cells, n_components)
    auto& arr           = field.array();
    std::size_t n_cells = arr.shape()[0];

    // Create result array
    py::array_t<double> result(std::vector<std::size_t>{n_comp});
    auto res_buf    = result.request();
    double* res_ptr = static_cast<double*>(res_buf.ptr);

    // Sum for each component
    for (std::size_t c = 0; c < n_comp; ++c)
    {
        double sum = 0.0;
        for (std::size_t i = 0; i < n_cells; ++i)
        {
            sum += arr(i, c);
        }
        res_ptr[c] = sum;
    }

    return result;
}

template <std::size_t dim, std::size_t n_comp>
py::array_t<double> vectorfield_sum_by_component(const VectorField<dim, n_comp, true>& field)
{
    // SOA layout: shape (n_components, n_cells)
    auto& arr           = field.array();
    std::size_t n_cells = arr.shape()[1];

    // Create result array
    py::array_t<double> result(std::vector<std::size_t>{n_comp});
    auto res_buf    = result.request();
    double* res_ptr = static_cast<double*>(res_buf.ptr);

    // Sum for each component
    for (std::size_t c = 0; c < n_comp; ++c)
    {
        double sum = 0.0;
        for (std::size_t i = 0; i < n_cells; ++i)
        {
            sum += arr(c, i);
        }
        res_ptr[c] = sum;
    }

    return result;
}

// Mean over all elements
template <std::size_t dim, std::size_t n_comp>
double vectorfield_mean(const VectorField<dim, n_comp, false>& field)
{
    return xt::mean(field.array())();
}

template <std::size_t dim, std::size_t n_comp>
double vectorfield_mean(const VectorField<dim, n_comp, true>& field)
{
    return xt::mean(field.array())();
}

// Max over all elements
template <std::size_t dim, std::size_t n_comp>
double vectorfield_max(const VectorField<dim, n_comp, false>& field)
{
    return xt::amax(field.array())();
}

template <std::size_t dim, std::size_t n_comp>
double vectorfield_max(const VectorField<dim, n_comp, true>& field)
{
    return xt::amax(field.array())();
}

// Min over all elements
template <std::size_t dim, std::size_t n_comp>
double vectorfield_min(const VectorField<dim, n_comp, false>& field)
{
    return xt::amin(field.array())();
}

template <std::size_t dim, std::size_t n_comp>
double vectorfield_min(const VectorField<dim, n_comp, true>& field)
{
    return xt::amin(field.array())();
}

// ============================================================
// Magnitude helpers for VectorField
// ============================================================

template <std::size_t dim, std::size_t n_comp>
py::array_t<double> vectorfield_magnitude(const VectorField<dim, n_comp, false>& field)
{
    // AOS layout: shape (n_cells, n_components)
    auto& arr           = field.array();
    std::size_t n_cells = arr.shape()[0];

    // Create output array
    py::array_t<double> result(std::vector<std::size_t>{n_cells});
    auto res_buf    = result.request();
    double* res_ptr = static_cast<double*>(res_buf.ptr);

    // Compute magnitude for each cell
    for (std::size_t i = 0; i < n_cells; ++i)
    {
        double sum_sq = 0.0;
        for (std::size_t c = 0; c < n_comp; ++c)
        {
            double val = arr(i, c);
            sum_sq += val * val;
        }
        res_ptr[i] = std::sqrt(sum_sq);
    }

    return result;
}

template <std::size_t dim, std::size_t n_comp>
py::array_t<double> vectorfield_magnitude(const VectorField<dim, n_comp, true>& field)
{
    // SOA layout: shape (n_components, n_cells)
    auto& arr           = field.array();
    std::size_t n_cells = arr.shape()[1];

    // Create output array
    py::array_t<double> result(std::vector<std::size_t>{n_cells});
    auto res_buf    = result.request();
    double* res_ptr = static_cast<double*>(res_buf.ptr);

    // Compute magnitude for each cell
    for (std::size_t i = 0; i < n_cells; ++i)
    {
        double sum_sq = 0.0;
        for (std::size_t c = 0; c < n_comp; ++c)
        {
            double val = arr(c, i);
            sum_sq += val * val;
        }
        res_ptr[i] = std::sqrt(sum_sq);
    }

    return result;
}

// ============================================================
// Field arithmetic operation helpers
// ============================================================

// Field - scalar operations (immediate evaluation, return new field)
template <std::size_t dim>
ScalarField<dim> field_sub_scalar(const ScalarField<dim>& field, double scalar)
{
    auto& mesh  = const_cast<typename ScalarField<dim>::mesh_t&>(field.mesh());
    // Initialize with 0 to avoid garbage values in ghost cells
    auto result = samurai::make_scalar_field<double>(field.name() + "_sub", mesh, 0.0);
    result      = field - scalar;
    return result;
}

template <std::size_t dim>
ScalarField<dim> scalar_sub_field(double scalar, const ScalarField<dim>& field)
{
    auto& mesh  = const_cast<typename ScalarField<dim>::mesh_t&>(field.mesh());
    // Initialize with 0 to avoid garbage values in ghost cells
    auto result = samurai::make_scalar_field<double>("scalar_sub", mesh, 0.0);
    result      = scalar - field;
    return result;
}

template <std::size_t dim>
ScalarField<dim> field_add_scalar(const ScalarField<dim>& field, double scalar)
{
    auto& mesh  = const_cast<typename ScalarField<dim>::mesh_t&>(field.mesh());
    // Initialize with 0 to avoid garbage values in ghost cells
    auto result = samurai::make_scalar_field<double>(field.name() + "_add", mesh, 0.0);
    result      = field + scalar;
    return result;
}

template <std::size_t dim>
ScalarField<dim> field_mul_scalar(const ScalarField<dim>& field, double scalar)
{
    auto& mesh  = const_cast<typename ScalarField<dim>::mesh_t&>(field.mesh());
    // Initialize with 0 to avoid garbage values in ghost cells
    auto result = samurai::make_scalar_field<double>(field.name() + "_mul", mesh, 0.0);
    result      = field * scalar;
    return result;
}

template <std::size_t dim>
ScalarField<dim> field_div_scalar(const ScalarField<dim>& field, double scalar)
{
    auto& mesh  = const_cast<typename ScalarField<dim>::mesh_t&>(field.mesh());
    // Initialize with 0 to avoid garbage values in ghost cells
    auto result = samurai::make_scalar_field<double>(field.name() + "_div", mesh, 0.0);
    result      = field / scalar;
    return result;
}

// Field - field operations
template <std::size_t dim>
ScalarField<dim> field_sub_field(const ScalarField<dim>& field1, const ScalarField<dim>& field2)
{
    auto& mesh  = const_cast<typename ScalarField<dim>::mesh_t&>(field1.mesh());
    // Initialize with 0 to avoid garbage values in ghost cells
    auto result = samurai::make_scalar_field<double>(field1.name() + "_sub", mesh, 0.0);
    result      = field1 - field2;
    return result;
}

template <std::size_t dim>
ScalarField<dim> field_add_field(const ScalarField<dim>& field1, const ScalarField<dim>& field2)
{
    auto& mesh  = const_cast<typename ScalarField<dim>::mesh_t&>(field1.mesh());
    // Initialize with 0 to avoid garbage values in ghost cells
    auto result = samurai::make_scalar_field<double>(field1.name() + "_add", mesh, 0.0);
    result      = field1 + field2;
    return result;
}

// Field utility operations
template <std::size_t dim>
ScalarField<dim> field_clone(const ScalarField<dim>& field)
{
    auto& mesh  = const_cast<typename ScalarField<dim>::mesh_t&>(field.mesh());
    auto result = samurai::make_scalar_field<double>(field.name() + "_clone", mesh);
    result      = field; // Deep copy via assignment operator
    return result;
}

template <std::size_t dim>
void field_copy_to(ScalarField<dim>& dest, const ScalarField<dim>& src)
{
    dest = src; // Uses copy assignment operator
}

// ============================================================
// I/O method helpers for Field classes
// ============================================================

// Helper to convert Python path to filesystem path
inline std::filesystem::path to_fs_path(const py::object& path_obj)
{
    if (path_obj.is_none())
    {
        return std::filesystem::current_path();
    }

    // Try os.PathLike protocol first (supports pathlib.Path)
    if (py::hasattr(path_obj, "__fspath__"))
    {
        auto fspath_result = path_obj.attr("__fspath__")();
        return std::filesystem::path(py::str(fspath_result));
    }

    // Fallback to string conversion
    return std::filesystem::path(py::str(path_obj));
}

// Parse unified filepath into directory and basename
struct FilePathParts
{
    std::filesystem::path directory;
    std::string basename;
};

inline FilePathParts parse_unified_filepath(const py::object& filepath_obj)
{
    std::filesystem::path filepath = to_fs_path(filepath_obj);

    // Extract directory and basename
    std::filesystem::path directory = filepath.parent_path();
    std::string basename            = filepath.stem().string();

    // If no directory, use current directory
    if (directory.empty())
    {
        directory = std::filesystem::current_path();
    }

    return {directory, basename};
}

// ============================================================
// I/O method wrappers for ScalarField
// ============================================================

template <std::size_t dim>
ScalarField<dim>& field_method_save(ScalarField<dim>& field, const py::object& filepath_obj)
{
    auto [directory, basename] = parse_unified_filepath(filepath_obj);
    samurai::save(directory, basename, field.mesh(), field);
    return field;
}

template <std::size_t dim>
ScalarField<dim>& field_method_dump(ScalarField<dim>& field, const py::object& filepath_obj)
{
    auto [directory, basename] = parse_unified_filepath(filepath_obj);
    samurai::dump(directory, basename, field.mesh(), field);
    return field;
}

template <std::size_t dim>
void field_method_load(ScalarField<dim>& field, const py::object& filepath_obj)
{
    auto [directory, basename] = parse_unified_filepath(filepath_obj);
    samurai::load(directory, basename, field.mesh(), field);
}

// ============================================================
// I/O method wrappers for VectorField
// ============================================================

template <std::size_t dim, std::size_t n_comp, bool SOA>
VectorField<dim, n_comp, SOA>& field_method_save_vector(VectorField<dim, n_comp, SOA>& field, const py::object& filepath_obj)
{
    auto [directory, basename] = parse_unified_filepath(filepath_obj);
    samurai::save(directory, basename, field.mesh(), field);
    return field;
}

template <std::size_t dim, std::size_t n_comp, bool SOA>
VectorField<dim, n_comp, SOA>& field_method_dump_vector(VectorField<dim, n_comp, SOA>& field, const py::object& filepath_obj)
{
    auto [directory, basename] = parse_unified_filepath(filepath_obj);
    samurai::dump(directory, basename, field.mesh(), field);
    return field;
}

// Helper to bind common Field methods
template <class Field, class Mesh, class... Options>
void bind_field_common_methods(py::class_<Field, Options...>& cls)
{
    using value_t = typename Field::value_type;

    // Name property
    cls.def_property(
        "name",
        [](const Field& f) -> const std::string&
        {
            return f.name();
        },
        [](Field& f, const std::string& name) -> std::string&
        {
            f.name() = name;
            return f.name();
        },
        "Field name (read/write)");

    // Mesh property (reference_internal ensures correct lifetime)
    cls.def_property_readonly(
        "mesh",
        [](const Field& f) -> const Mesh&
        {
            return f.mesh();
        },
        py::return_value_policy::reference_internal,
        "Mesh this field is defined on");

    // Dimension property
    cls.def_property_readonly(
        "dim",
        [](const Field&)
        {
            return Field::dim;
        },
        "Field dimension");

    // Size (number of cells)
    cls.def_property_readonly(
        "size",
        [](const Field& f) -> std::size_t
        {
            return f.mesh().nb_cells();
        },
        "Total number of cells in the field");

    // Fill with constant value (returns self for chaining)
    cls.def(
        "fill",
        [](Field& f, double value) -> Field&
        {
            f.fill(value);
            return f;
        },
        py::arg("value"),
        R"pbdoc(
            Fill the field with a constant value.

            Parameters
            ----------
            value : float
                Value to fill all cells with

            Returns
            -------
            Field
                Returns self for method chaining

            Examples
            --------
            >>> field.fill(0.0)  # Fill and return self
            >>> field.fill(1.0).resize()  # Chain operations
            )pbdoc");

    // Ghost cells flag
    cls.def_property(
        "ghosts_updated",
        [](const Field& f) -> bool
        {
            return f.ghosts_updated();
        },
        [](Field& f, bool value) -> bool&
        {
            f.ghosts_updated() = value;
            return f.ghosts_updated();
        },
        "Ghost cells update flag");

    // Resize field to match current mesh size
    cls.def(
        "resize",
        [](Field& f) -> Field&
        {
            f.resize();
            return f;
        },
        R"pbdoc(
            Resize the field after mesh adaptation.

            Returns
            -------
            Field
                Returns self for method chaining

            Examples
            --------
            >>> field.resize().fill(0.0)
            )pbdoc");

    // In-place assignment from another field (reuses storage)
    cls.def(
        "assign",
        [](Field& dest, const Field& src) -> Field&
        {
            std::string saved_name = dest.name(); // Save destination's original name
            dest                   = src;         // Uses C++ assignment operator, reuses dest's storage
            dest.name()            = saved_name;  // Restore original name (prevents "u_mul_add..." bug)
            return dest;
        },
        py::arg("other"),
        "In-place assignment from another field.\n\n"
        "This method reuses this field's existing storage, avoiding allocation\n"
        "and preventing stale mesh reference issues after mesh adaptation.\n\n"
        "Unlike arithmetic operators (which create new fields with captured\n"
        "mesh references), this uses the C++ assignment operator which works\n"
        "correctly even after mesh structure changes.\n\n"
        "Note: Preserves the destination field's name (e.g., 'u1' stays 'u1'\n"
        "even when assigning from an expression like 'u - dt * flux').\n\n"
        "Examples\n"
        "--------\n"
        "    >>> # WRONG: Creates new field with stale mesh reference\n"
        "    >>> u1 = u - dt * flux  # u1 has captured mesh reference\n"
        "    >>> MRadaptation(config)\n"
        "    >>> u1.resize()  # SEGFAULT: stale reference\n"
        "\n"
        "    >>> # CORRECT: Reuses existing storage\n"
        "    >>> u1.assign(u - dt * flux)\n"
        "    >>> MRadaptation(config)\n"
        "    >>> u1.resize()  # OK: u1 still references correct mesh\n\n"
        "Returns\n"
        "-------\n"
        "    Field: Reference to self (for chaining)\n\n"
        "See Also\n"
        "---------\n"
        "    resize : Resize field storage after mesh adaptation");
}

// Template function to bind ScalarField for a specific dimension
template <std::size_t dim>
void bind_scalar_field(py::module_& m, const std::string& name)
{
    using Mesh    = MRMesh<dim>;
    using Field   = ScalarField<dim>;
    using value_t = typename Field::value_type;

    // Create class with docstring
    auto cls = py::class_<Field>(m, name.c_str(), R"pbdoc(
        Scalar Field on adaptive mesh

        A scalar field defined on an adaptive mesh refinement grid.
        Provides zero-copy NumPy integration for efficient data access.

        Parameters
        ----------
        mesh : MRMesh
            Mesh to define the field on
        name : str
            Field identifier
        init : float, optional
            Initial value for all cells (default: 0.0)

        Examples
        --------
        >>> import samurai as sam
        >>> box = sam.Box2D([0., 0.], [1., 1.])
        >>> config = sam.MeshConfig2D()
        >>> config.min_level = 0
        >>> config.max_level = 2
        >>> mesh = sam.MRMesh2D(box, config)
        >>> field = sam.ScalarField2D(mesh, "u")
        >>> field.fill(1.0)

        NumPy Integration
        -----------------
        >>> import numpy as np
        >>> arr = field.numpy_view()  # Zero-copy view
        >>> arr[:] = np.sin(x) * np.cos(y)
        >>> # Modifying arr modifies field in-place

        Attributes
        ----------
        name : str
            Field name
        mesh : MRMesh
            Underlying mesh
        size : int
            Number of cells
    )pbdoc");

    // Constructor using factory function
    cls.def(py::init(
                [](Mesh& mesh, const std::string& field_name, value_t init)
                {
                    auto field = samurai::make_scalar_field<value_t>(field_name, mesh, init);
                    return field;
                }),
            py::arg("mesh"),
            py::arg("name"),
            py::arg("init") = 0.0,
            py::keep_alive<1, 2>(), // Field keeps Mesh alive
            "Create scalar field");

    // Bind common methods
    bind_field_common_methods<Field, Mesh>(cls);

    // Explicit numpy_view method (zero-copy NumPy integration)
    cls.def(
        "numpy_view",
        [](Field& f) -> py::array_t<value_t>
        {
            auto& xt = f.array();
            return py::array_t<value_t>({xt.size()},       // Shape
                                        {sizeof(value_t)}, // Strides
                                        xt.data(),         // Data pointer
                                        py::cast(f)        // Keep field alive
            );
        },
        py::return_value_policy::take_ownership,
        "Returns zero-copy NumPy view of field data");

    // ============================================================
    // NumPy-like reduction methods
    // ============================================================

    cls.def(
        "sum",
        [](const Field& f) -> double
        {
            return field_sum<dim>(f);
        },
        "Return the sum of all field values.\n\n"
        "Equivalent to np.sum(field.numpy_view()).");

    cls.def(
        "mean",
        [](const Field& f) -> double
        {
            return field_mean<dim>(f);
        },
        "Return the mean (average) of all field values.\n\n"
        "Equivalent to np.mean(field.numpy_view()).");

    cls.def(
        "max",
        [](const Field& f) -> double
        {
            return field_max<dim>(f);
        },
        "Return the maximum value in the field.\n\n"
        "Equivalent to np.max(field.numpy_view()).");

    cls.def(
        "min",
        [](const Field& f) -> double
        {
            return field_min<dim>(f);
        },
        "Return the minimum value in the field.\n\n"
        "Equivalent to np.min(field.numpy_view()).");

    // Integer-based indexing
    // Note: For CellWrapper-based indexing from for_each_cell, use field[cell.index]
    cls.def(
        "__getitem__",
        [](Field& f, std::size_t i) -> value_t
        {
            return f[i];
        },
        py::arg("index"),
        "Get field value by flat index");

    cls.def(
        "__setitem__",
        [](Field& f, std::size_t i, value_t value)
        {
            f[i] = value;
        },
        py::arg("index"),
        py::arg("value"),
        "Set field value by flat index");

    // Arithmetic operators: field +/-/* scalar
    cls.def("__sub__", &field_sub_scalar<dim>, py::arg("scalar"), "Subtract scalar from field (returns new field)");

    // __rsub__ is called for expressions like "scalar - field"
    // We need to wrap scalar_sub_field because the signature is (scalar, field) but pybind11 expects (field, scalar)
    cls.def(
        "__rsub__",
        [](const ScalarField<dim>& field, double scalar)
        {
            return scalar_sub_field(scalar, field);
        },
        py::arg("scalar"),
        "Subtract field from scalar (returns new field)");

    cls.def("__add__", &field_add_scalar<dim>, py::arg("scalar"), "Add scalar to field (returns new field)");

    cls.def("__radd__", &field_add_scalar<dim>, py::arg("scalar"), "Add scalar to field (right-hand version)");

    cls.def("__mul__", &field_mul_scalar<dim>, py::arg("scalar"), "Multiply field by scalar (returns new field)");

    cls.def("__rmul__", &field_mul_scalar<dim>, py::arg("scalar"), "Multiply field by scalar (right-hand version)");

    cls.def("__truediv__", &field_div_scalar<dim>, py::arg("scalar"), "Divide field by scalar (returns new field)");

    // In-place arithmetic operators: field +/-/*= scalar
    // These modify the field in-place and return self (for chaining)
    // They use the C++ assignment operator which reuses storage, avoiding
    // stale mesh reference issues after mesh adaptation
    //
    // IMPORTANT: To ensure ghost cells are updated correctly, we:
    // 1. Create a temporary field with the operation result
    // 2. Copy it to self (reusing storage)
    // 3. Return self
    //
    // This ensures ghost cells get the correct values from the result field,
    // not just the initial values from make_scalar_field.
    cls.def(
        "__iadd__",
        [](ScalarField<dim>& field, double scalar) -> ScalarField<dim>&
        {
            auto& mesh  = const_cast<typename ScalarField<dim>::mesh_t&>(field.mesh());
            // Create result with 0 init (ghost cells will be 0.0)
            auto result = samurai::make_scalar_field<double>(field.name() + "_temp", mesh, 0.0);
            result      = field + scalar;  // Compute result
            field       = result;         // Copy to self (reuses storage)
            field.name() = field.name().substr(0, field.name().size() - 5);  // Remove "_temp"
            return field;
        },
        py::arg("scalar"),
        "In-place addition: field += scalar\n\n"
        "Modifies this field in-place and returns self.\n"
        "This is the preferred way to update fields after mesh adaptation.");

    cls.def(
        "__isub__",
        [](ScalarField<dim>& field, double scalar) -> ScalarField<dim>&
        {
            auto& mesh  = const_cast<typename ScalarField<dim>::mesh_t&>(field.mesh());
            auto result = samurai::make_scalar_field<double>(field.name() + "_temp", mesh, 0.0);
            result      = field - scalar;
            field       = result;
            field.name() = field.name().substr(0, field.name().size() - 5);
            return field;
        },
        py::arg("scalar"),
        "In-place subtraction: field -= scalar\n\n"
        "Modifies this field in-place and returns self.");

    cls.def(
        "__imul__",
        [](ScalarField<dim>& field, double scalar) -> ScalarField<dim>&
        {
            auto& mesh  = const_cast<typename ScalarField<dim>::mesh_t&>(field.mesh());
            auto result = samurai::make_scalar_field<double>(field.name() + "_temp", mesh, 0.0);
            result      = field * scalar;
            field       = result;
            field.name() = field.name().substr(0, field.name().size() - 5);
            return field;
        },
        py::arg("scalar"),
        "In-place multiplication: field *= scalar\n\n"
        "Modifies this field in-place and returns self.");

    cls.def(
        "__itruediv__",
        [](ScalarField<dim>& field, double scalar) -> ScalarField<dim>&
        {
            auto& mesh  = const_cast<typename ScalarField<dim>::mesh_t&>(field.mesh());
            auto result = samurai::make_scalar_field<double>(field.name() + "_temp", mesh, 0.0);
            result      = field / scalar;
            field       = result;
            field.name() = field.name().substr(0, field.name().size() - 5);
            return field;
        },
        py::arg("scalar"),
        "In-place division: field /= scalar\n\n"
        "Modifies this field in-place and returns self.");

    // Field-to-field operators
    cls.def("__sub__", &field_sub_field<dim>, py::arg("other"), "Subtract another field (returns new field)");

    cls.def("__add__", &field_add_field<dim>, py::arg("other"), "Add another field (returns new field)");

    // In-place field-to-field operators: field +/-= other_field
    // These modify the field in-place and return self (for chaining)
    cls.def(
        "__iadd__",
        [](ScalarField<dim>& field, const ScalarField<dim>& other) -> ScalarField<dim>&
        {
            field = field + other;
            return field;
        },
        py::arg("other"),
        "In-place addition: field += other_field\n\n"
        "Modifies this field in-place and returns self.");

    cls.def(
        "__isub__",
        [](ScalarField<dim>& field, const ScalarField<dim>& other) -> ScalarField<dim>&
        {
            field = field - other;
            return field;
        },
        py::arg("other"),
        "In-place subtraction: field -= other_field\n\n"
        "Modifies this field in-place and returns self.");

    // Utility methods
    cls.def("clone", &field_clone<dim>, "Create a deep copy of this field");

    cls.def("copy_to", &field_copy_to<dim>, py::arg("dest"), "Copy this field's data to destination field");

    // ============================================================
    // I/O methods
    // ============================================================

    cls.def(
        "save",
        [](Field& f, const py::object& filepath_obj) -> Field&
        {
            field_method_save<dim>(f, filepath_obj);
            return f;
        },
        py::arg("filepath"),
        R"pbdoc(
            Save field to HDF5 + XDMF for Paraview visualization.

            Parameters
            ----------
            filepath : str or Path
                Output file path (e.g., 'results/solution.h5')
                The .h5 and .xdmf extensions are added automatically.

            Returns
            -------
            Field
                Returns self for method chaining

            Creates
            -------
            {directory}/{basename}.h5 - HDF5 data file
            {directory}/{basename}.xdmf - XDMF metadata file for Paraview

            Examples
            --------
            >>> field.save('solution.h5')              # Current directory
            >>> field.save('results/solution.h5')      # Subdirectory
            >>> from pathlib import Path
            >>> field.save(Path('results') / 'solution.h5')  # pathlib support
            >>> field.fill(1.0).save('solution.h5')    # Method chaining

            Notes
            -----
            - Creates both HDF5 data file and XDMF metadata file
            - Use .dump() for checkpoint/restart (HDF5 only)
            - Supports pathlib.Path objects

            See Also
            --------
            dump : Save to HDF5 only for checkpoint/restart
            load : Load field from HDF5 restart file
            samurai.open_h5py : Open HDF5 file with h5py for direct access
        )pbdoc");

    cls.def(
        "dump",
        [](Field& f, const py::object& filepath_obj) -> Field&
        {
            field_method_dump<dim>(f, filepath_obj);
            return f;
        },
        py::arg("filepath"),
        R"pbdoc(
            Dump field to HDF5 for checkpoint/restart (no XDMF metadata).

            Parameters
            ----------
            filepath : str or Path
                Output file path (e.g., 'checkpoints/restart.h5')
                The .h5 extension is added automatically.

            Returns
            -------
            Field
                Returns self for method chaining

            Creates
            -------
            {directory}/{basename}.h5 - HDF5 restart file only

            Examples
            --------
            >>> field.dump('checkpoint.h5')             # Current directory
            >>> field.dump('checkpoints/restart.h5')    # Subdirectory
            >>> field.fill(1.0).dump('checkpoint.h5')   # Method chaining

            Notes
            -----
            - Creates HDF5-only file (more efficient for checkpoints)
            - Use .save() to create XDMF metadata for Paraview visualization
            - Use .load() to restore from checkpoint file

            See Also
            --------
            save : Save to HDF5 + XDMF for visualization
            load : Load field from HDF5 restart file
        )pbdoc");

    cls.def(
        "load",
        [](Field& f, const py::object& filepath_obj)
        {
            field_method_load<dim>(f, filepath_obj);
        },
        py::arg("filepath"),
        R"pbdoc(
            Load field data from HDF5 restart file (modifies in place).

            Parameters
            ----------
            filepath : str or Path
                Input file path (e.g., 'checkpoints/restart.h5')
                The .h5 extension is added automatically.

            Examples
            --------
            >>> field.load('checkpoint.h5')             # Load from current directory
            >>> field.load('checkpoints/restart.h5')    # Load from subdirectory

            Notes
            -----
            - Modifies this field in-place (replaces data)
            - Mesh structure is also loaded from the file
            - Field name must match the name used when creating the checkpoint

            See Also
            --------
            dump : Create checkpoint file
            save : Save field with XDMF metadata
        )pbdoc");

    // String representation
    cls.def("__repr__",
            [](const Field& f)
            {
                std::ostringstream oss;
                oss << "ScalarField" << Field::dim << "D(";
                oss << "name='" << f.name() << "', ";
                oss << "size=" << f.mesh().nb_cells();
                oss << ")";
                return oss.str();
            });

    cls.def("__str__",
            [](const Field& f)
            {
                std::ostringstream oss;
                oss << "ScalarField" << Field::dim << "D";
                oss << " '" << f.name() << "'";
                oss << " [" << f.mesh().nb_cells() << " cells]";
                return oss.str();
            });
}

// Helper to bind VectorField-specific methods
template <class Field, class Mesh, class... Options>
void bind_vectorfield_methods(py::class_<Field, Options...>& cls)
{
    using value_t                       = typename Field::value_type;
    static constexpr std::size_t n_comp = Field::n_comp;

    // Number of components property
    cls.def_property_readonly(
        "n_components",
        [](const Field&)
        {
            return n_comp;
        },
        "Number of components");

    // Is SOA layout property
    cls.def_property_readonly(
        "is_soa",
        [](const Field&)
        {
            return Field::is_soa;
        },
        "True if Structure of Arrays layout, False if Array of Structures");

    // Fill with scalar value (returns self for chaining)
    cls.def(
        "fill",
        [](Field& f, double value) -> Field&
        {
            f.fill(value);
            return f;
        },
        py::arg("value"),
        R"pbdoc(
            Fill all components and cells with a constant value.

            Parameters
            ----------
            value : float
                Value to fill all components and cells with

            Returns
            -------
            Field
                Returns self for method chaining

            Examples
            --------
            >>> field.fill(0.0)  # Fill all components
            >>> field.fill(1.0).resize()  # Chain operations
            )pbdoc");

    // Fill with per-component values (returns self for chaining)
    cls.def(
        "fill",
        [](Field& f, py::list values) -> Field&
        {
            if (values.size() != n_comp)
            {
                throw std::runtime_error("Expected " + std::to_string(n_comp) + " values, got " + std::to_string(values.size()));
            }
            std::vector<value_t> vals;
            for (std::size_t i = 0; i < n_comp; ++i)
            {
                vals.push_back(values[i].cast<value_t>());
            }

            auto& xt = f.array();
            if constexpr (Field::is_soa)
            {
                // SOA: fill each component
                for (std::size_t comp = 0; comp < n_comp; ++comp)
                {
                    std::size_t n_cells = xt.shape()[1];
                    for (std::size_t i = 0; i < n_cells; ++i)
                    {
                        xt(comp, i) = vals[comp];
                    }
                }
            }
            else
            {
                // AOS: fill each cell's components
                std::size_t n_cells = xt.shape()[0];
                for (std::size_t i = 0; i < n_cells; ++i)
                {
                    for (std::size_t comp = 0; comp < n_comp; ++comp)
                    {
                        xt(i, comp) = vals[comp];
                    }
                }
            }
            return f;
        },
        py::arg("values"),
        R"pbdoc(
        Fill all cells with per-component values.

        Parameters
        ----------
        values : list of float
            Values for each component (length must match n_components)

        Returns
        -------
        Field
            Returns self for method chaining

        Examples
        --------
        >>> velocity.fill([1.0, 0.0])  # Fill 2-component field
        >>> velocity.fill([0.0, 1.0, 0.0]).resize()  # Chain operations
        )pbdoc");

    // String representation
    cls.def("__repr__",
            [](const Field& f)
            {
                std::ostringstream oss;
                oss << "VectorField" << Field::dim << "D(";
                oss << "name='" << f.name() << "', ";
                oss << "n_comp=" << n_comp << ", ";
                oss << "size=" << f.mesh().nb_cells();
                oss << ")";
                return oss.str();
            });

    cls.def("__str__",
            [](const Field& f)
            {
                std::ostringstream oss;
                oss << "VectorField" << Field::dim << "D";
                oss << " '" << f.name() << "'";
                oss << " [" << n_comp << " components]";
                oss << " [" << f.mesh().nb_cells() << " cells]";
                return oss.str();
            });

    // Explicit numpy_view method
    cls.def(
        "numpy_view",
        [](Field& f) -> py::array_t<value_t>
        {
            auto& xt = f.array();

            if constexpr (Field::is_soa)
            {
                // SOA: (n_components, n_cells)
                return py::array_t<value_t>({n_comp, static_cast<std::size_t>(xt.shape()[1])},
                                            {static_cast<std::size_t>(xt.shape()[1]) * sizeof(value_t), sizeof(value_t)},
                                            xt.data(),
                                            py::cast(f));
            }
            else
            {
                // AOS: (n_cells, n_components)
                return py::array_t<value_t>({static_cast<std::size_t>(xt.shape()[0]), n_comp},
                                            {n_comp * sizeof(value_t), sizeof(value_t)},
                                            xt.data(),
                                            py::cast(f));
            }
        },
        py::return_value_policy::take_ownership,
        "Returns zero-copy NumPy view of vector field data");

    // ============================================================
    // NumPy-like reduction methods
    // ============================================================

    cls.def(
        "sum",
        [](const Field& f) -> double
        {
            return vectorfield_sum_all<Field::dim, n_comp>(f);
        },
        "Return the sum of all field values (all components).\n\n"
        "Equivalent to np.sum(field.numpy_view()).");

    cls.def(
        "sum",
        [](const Field& f, const std::string& axis) -> py::array_t<double>
        {
            if (axis == "components")
            {
                return vectorfield_sum_by_component<Field::dim, n_comp>(f);
            }
            else
            {
                throw std::runtime_error("Invalid axis: '" + axis + "'. Use 'components' for per-component sum.");
            }
        },
        py::arg("axis"),
        "Return sum by component.\n\n"
        "Parameters\n"
        "----------\n"
        "axis : str\n"
        "    'components' - return sum for each component separately\n\n"
        "Returns\n"
        "-------\n"
        "numpy.ndarray\n"
        "    Array of shape (n_components,) with sums");

    cls.def(
        "mean",
        [](const Field& f) -> double
        {
            return vectorfield_mean<Field::dim, n_comp>(f);
        },
        "Return the mean (average) of all field values.\n\n"
        "Equivalent to np.mean(field.numpy_view()).");

    cls.def(
        "max",
        [](const Field& f) -> double
        {
            return vectorfield_max<Field::dim, n_comp>(f);
        },
        "Return the maximum value in the field.\n\n"
        "Equivalent to np.max(field.numpy_view()).");

    cls.def(
        "min",
        [](const Field& f) -> double
        {
            return vectorfield_min<Field::dim, n_comp>(f);
        },
        "Return the minimum value in the field.\n\n"
        "Equivalent to np.min(field.numpy_view()).");

    // Magnitude (Euclidean norm) for each cell
    cls.def(
        "magnitude",
        [](const Field& f) -> py::array_t<double>
        {
            return vectorfield_magnitude<Field::dim, n_comp>(f);
        },
        "Return the Euclidean magnitude (norm) for each cell.\n\n"
        "Returns a NumPy array of shape (n_cells,) with:\n"
        "    sqrt(v0^2 + v1^2 + ...) for each cell\n\n"
        "Example:\n"
        "    >>> vel = sam.field.vector(mesh, 'velocity', n_components=2)\n"
        "    >>> speed = vel.magnitude()  # Get speed at each cell");

    // Get component as scalar field (returns new field with extracted component)
    cls.def(
        "get_component",
        [](Field& f, std::size_t comp)
        {
            // Create a new scalar field with the same mesh
            auto result = samurai::make_scalar_field<value_t>(f.name() + "_comp" + std::to_string(comp), const_cast<Mesh&>(f.mesh()));

            // Copy component data
            auto& src = f.array();
            auto& dst = result.array();

            if constexpr (Field::is_soa)
            {
                // SOA: component is contiguous
                std::size_t n_cells = src.shape()[1];
                for (std::size_t i = 0; i < n_cells; ++i)
                {
                    dst[i] = src(comp, i);
                }
            }
            else
            {
                // AOS: component is strided
                std::size_t n_cells = src.shape()[0];
                for (std::size_t i = 0; i < n_cells; ++i)
                {
                    dst[i] = src(i, comp);
                }
            }

            return result;
        },
        py::arg("component"),
        "Extract a single component as a new ScalarField");

    // ============================================================
    // I/O methods for VectorField
    // ============================================================

    cls.def(
        "save",
        [](Field& f, const py::object& filepath_obj) -> Field&
        {
            field_method_save_vector<Field::dim, n_comp, Field::is_soa>(f, filepath_obj);
            return f;
        },
        py::arg("filepath"),
        R"pbdoc(
            Save vector field to HDF5 + XDMF for Paraview visualization.

            Parameters
            ----------
            filepath : str or Path
                Output file path (e.g., 'results/velocity.h5')
                The .h5 and .xdmf extensions are added automatically.

            Returns
            -------
            Field
                Returns self for method chaining

            Creates
            -------
            {directory}/{basename}.h5 - HDF5 data file
            {directory}/{basename}.xdmf - XDMF metadata file for Paraview

            Examples
            --------
            >>> velocity.save('velocity.h5')              # Current directory
            >>> velocity.save('results/velocity.h5')      # Subdirectory
            >>> velocity.fill([1.0, 0.0]).save('velocity.h5')  # Method chaining

            See Also
            --------
            dump : Save to HDF5 only for checkpoint/restart
        )pbdoc");

    cls.def(
        "dump",
        [](Field& f, const py::object& filepath_obj) -> Field&
        {
            field_method_dump_vector<Field::dim, n_comp, Field::is_soa>(f, filepath_obj);
            return f;
        },
        py::arg("filepath"),
        R"pbdoc(
            Dump vector field to HDF5 for checkpoint/restart (no XDMF metadata).

            Parameters
            ----------
            filepath : str or Path
                Output file path (e.g., 'checkpoints/restart.h5')
                The .h5 extension is added automatically.

            Returns
            -------
            Field
                Returns self for method chaining

            Examples
            --------
            >>> velocity.dump('checkpoint.h5')             # Current directory
            >>> velocity.dump('checkpoints/restart.h5')    # Subdirectory
            >>> velocity.fill([1.0, 0.0]).dump('checkpoint.h5')  # Method chaining

            See Also
            --------
            save : Save to HDF5 + XDMF for visualization
        )pbdoc");

    // Integer-based indexing for VectorField
    // __getitem__ returns a list of component values
    cls.def(
        "__getitem__",
        [](Field& f, std::size_t index) -> py::list
        {
            py::list result;
            auto& xt = f.array();

            if constexpr (Field::is_soa)
            {
                // SOA: shape is (n_components, n_cells)
                for (std::size_t comp = 0; comp < n_comp; ++comp)
                {
                    result.append(xt(comp, index));
                }
            }
            else
            {
                // AOS: shape is (n_cells, n_components)
                for (std::size_t comp = 0; comp < n_comp; ++comp)
                {
                    result.append(xt(index, comp));
                }
            }
            return result;
        },
        py::arg("index"),
        "Get all components of a cell by flat index");

    // __setitem__ accepts a list of component values
    cls.def(
        "__setitem__",
        [](Field& f, std::size_t index, py::list values)
        {
            if (values.size() != n_comp)
            {
                throw std::runtime_error("Expected " + std::to_string(n_comp) + " values, got " + std::to_string(values.size()));
            }

            std::vector<value_t> vals;
            for (std::size_t i = 0; i < n_comp; ++i)
            {
                vals.push_back(values[i].cast<value_t>());
            }

            auto& xt = f.array();
            if constexpr (Field::is_soa)
            {
                // SOA: shape is (n_components, n_cells)
                for (std::size_t comp = 0; comp < n_comp; ++comp)
                {
                    xt(comp, index) = vals[comp];
                }
            }
            else
            {
                // AOS: shape is (n_cells, n_components)
                for (std::size_t comp = 0; comp < n_comp; ++comp)
                {
                    xt(index, comp) = vals[comp];
                }
            }
        },
        py::arg("index"),
        py::arg("values"),
        "Set all components of a cell by flat index");

    // Arithmetic operators: field +/-/* scalar
    cls.def(
        "__sub__",
        [](Field& f, double scalar)
        {
            auto& mesh  = const_cast<Mesh&>(f.mesh());
            // Initialize with 0 to avoid garbage values in ghost cells
            auto result = samurai::make_vector_field<value_t, n_comp, Field::is_soa>(f.name() + "_sub", mesh, 0.0);
            result      = f - scalar;
            return result;
        },
        py::arg("scalar"),
        "Subtract scalar from field (returns new field)");

    cls.def(
        "__rsub__",
        [](Field& f, double scalar)
        {
            auto& mesh  = const_cast<Mesh&>(f.mesh());
            // Initialize with 0 to avoid garbage values in ghost cells
            auto result = samurai::make_vector_field<value_t, n_comp, Field::is_soa>("scalar_sub", mesh, 0.0);
            result      = scalar - f;
            return result;
        },
        py::arg("scalar"),
        "Subtract field from scalar (returns new field)");

    cls.def(
        "__add__",
        [](Field& f, double scalar)
        {
            auto& mesh  = const_cast<Mesh&>(f.mesh());
            // Initialize with 0 to avoid garbage values in ghost cells
            auto result = samurai::make_vector_field<value_t, n_comp, Field::is_soa>(f.name() + "_add", mesh, 0.0);
            result      = f + scalar;
            return result;
        },
        py::arg("scalar"),
        "Add scalar to field (returns new field)");

    cls.def(
        "__radd__",
        [](Field& f, double scalar)
        {
            auto& mesh  = const_cast<Mesh&>(f.mesh());
            // Initialize with 0 to avoid garbage values in ghost cells
            auto result = samurai::make_vector_field<value_t, n_comp, Field::is_soa>(f.name() + "_add", mesh, 0.0);
            result      = f + scalar;
            return result;
        },
        py::arg("scalar"),
        "Add scalar to field (right-hand version)");

    cls.def(
        "__mul__",
        [](Field& f, double scalar)
        {
            auto& mesh  = const_cast<Mesh&>(f.mesh());
            // Initialize with 0 to avoid garbage values in ghost cells
            auto result = samurai::make_vector_field<value_t, n_comp, Field::is_soa>(f.name() + "_mul", mesh, 0.0);
            result      = f * scalar;
            return result;
        },
        py::arg("scalar"),
        "Multiply field by scalar (returns new field)");

    cls.def(
        "__rmul__",
        [](Field& f, double scalar)
        {
            auto& mesh  = const_cast<Mesh&>(f.mesh());
            // Initialize with 0 to avoid garbage values in ghost cells
            auto result = samurai::make_vector_field<value_t, n_comp, Field::is_soa>(f.name() + "_mul", mesh, 0.0);
            result      = f * scalar;
            return result;
        },
        py::arg("scalar"),
        "Multiply field by scalar (right-hand version)");

    cls.def(
        "__truediv__",
        [](Field& f, double scalar)
        {
            auto& mesh  = const_cast<Mesh&>(f.mesh());
            // Initialize with 0 to avoid garbage values in ghost cells
            auto result = samurai::make_vector_field<value_t, n_comp, Field::is_soa>(f.name() + "_div", mesh, 0.0);
            result      = f / scalar;
            return result;
        },
        py::arg("scalar"),
        "Divide field by scalar (returns new field)");

    // In-place arithmetic operators: field +/-/*= scalar
    // These modify the field in-place and return self (for chaining)
    // They use the C++ assignment operator which reuses storage, avoiding
    // stale mesh reference issues after mesh adaptation
    cls.def(
        "__iadd__",
        [](Field& f, double scalar) -> Field&
        {
            f = f + scalar;
            return f;
        },
        py::arg("scalar"),
        "In-place addition: field += scalar\n\n"
        "Modifies this field in-place and returns self.\n"
        "This is the preferred way to update fields after mesh adaptation.");

    cls.def(
        "__isub__",
        [](Field& f, double scalar) -> Field&
        {
            f = f - scalar;
            return f;
        },
        py::arg("scalar"),
        "In-place subtraction: field -= scalar\n\n"
        "Modifies this field in-place and returns self.");

    cls.def(
        "__imul__",
        [](Field& f, double scalar) -> Field&
        {
            f = f * scalar;
            return f;
        },
        py::arg("scalar"),
        "In-place multiplication: field *= scalar\n\n"
        "Modifies this field in-place and returns self.");

    cls.def(
        "__itruediv__",
        [](Field& f, double scalar) -> Field&
        {
            f = f / scalar;
            return f;
        },
        py::arg("scalar"),
        "In-place division: field /= scalar\n\n"
        "Modifies this field in-place and returns self.");

    // Arithmetic operators: field +/- field
    cls.def(
        "__add__",
        [](Field& f, const Field& other)
        {
            auto& mesh  = const_cast<Mesh&>(f.mesh());
            auto result = samurai::make_vector_field<value_t, n_comp, Field::is_soa>(f.name() + "_add", mesh);
            result      = f + other;
            return result;
        },
        py::arg("other"),
        "Add two fields (returns new field)");

    cls.def(
        "__sub__",
        [](Field& f, const Field& other)
        {
            auto& mesh  = const_cast<Mesh&>(f.mesh());
            auto result = samurai::make_vector_field<value_t, n_comp, Field::is_soa>(f.name() + "_sub", mesh);
            result      = f - other;
            return result;
        },
        py::arg("other"),
        "Subtract two fields (returns new field)");

    // In-place field-to-field operators: field +/-= other_field
    // These modify the field in-place and return self (for chaining)
    cls.def(
        "__iadd__",
        [](Field& f, const Field& other) -> Field&
        {
            f = f + other;
            return f;
        },
        py::arg("other"),
        "In-place addition: field += other_field\n\n"
        "Modifies this field in-place and returns self.");

    cls.def(
        "__isub__",
        [](Field& f, const Field& other) -> Field&
        {
            f = f - other;
            return f;
        },
        py::arg("other"),
        "In-place subtraction: field -= other_field\n\n"
        "Modifies this field in-place and returns self.");
}

// Template function to bind VectorField for a specific dimension and component count
template <std::size_t dim, std::size_t n_comp, bool SOA>
void bind_vector_field(py::module_& m, const std::string& name)
{
    using Mesh    = MRMesh<dim>;
    using Field   = VectorField<dim, n_comp, SOA>;
    using value_t = typename Field::value_type;

    // Create class with docstring
    auto cls = py::class_<Field>(m, name.c_str(), R"pbdoc(
        Vector Field on adaptive mesh

        A multi-component field defined on an adaptive mesh refinement grid.
        Provides zero-copy NumPy integration for efficient data access.

        Parameters
        ----------
        mesh : MRMesh
            Mesh to define the field on
        name : str
            Field identifier
        init : float, optional
            Initial value for all components and cells (default: 0.0)

        Examples
        --------
        >>> import samurai as sam
        >>> mesh = sam.MRMesh2D(box, config)
        >>> velocity = sam.VectorField2D_2(mesh, "vel", 2)
        >>> velocity.fill([1.0, 0.0])  # Set (u, v) components

        NumPy Integration
        -----------------
        >>> arr = velocity.numpy_view()  # Shape: (n_cells, n_components)
        >>> arr[:, 0] = u_values  # Set first component
        >>> arr[:, 1] = v_values  # Set second component

        Attributes
        ----------
        name : str
            Field name
        mesh : MRMesh
            Underlying mesh
        n_components : int
            Number of components
        is_soa : bool
            Memory layout (True=Structure of Arrays, False=Array of Structures)
    )pbdoc");

    // Constructor using factory function
    cls.def(py::init(
                [](Mesh& mesh, const std::string& field_name, value_t init)
                {
                    auto field = samurai::make_vector_field<value_t, n_comp, SOA>(field_name, mesh, init);
                    return field;
                }),
            py::arg("mesh"),
            py::arg("name"),
            py::arg("init") = 0.0,
            py::keep_alive<1, 2>(), // Field keeps Mesh alive
            "Create vector field");

    // Bind common methods (name, mesh, size, etc.)
    bind_field_common_methods<Field, Mesh>(cls);

    // Bind vector-specific methods
    bind_vectorfield_methods<Field, Mesh>(cls);
}

// Module initialization function for Field bindings
void init_field_bindings(py::module_& m)
{
    // ============================================================
    // BREAKING CHANGE (v0.30.0): Explicit Field classes removed from public API
    // Users must use the factory: sam.field.scalar(mesh, name, init=0.0)
    //
    // NOTE: We still register the Field types with pybind11 (as _ScalarField1D, etc.)
    // because the factory function needs to return these types. The underscore prefix
    // indicates they are internal implementation details, not public API.
    // ============================================================

    // ============================================================
    // Create field submodule for organized API access
    // ============================================================
    py::module_ field = m.def_submodule("field",
                                        "Field classes for Samurai AMR simulations\n\n"
                                        "Factory Functions:\n"
                                        "  scalar(mesh, name, init=0.0) - Create ScalarField (dim inferred from mesh)\n"
                                        "  vector(mesh, name, n_components=2, init=0.0) - Create VectorField\n\n"
                                        "Examples:\n"
                                        "    >>> import samurai_python as sam\n"
                                        "    >>> mesh = sam.mesh.make(box, min_level=4, max_level=8)\n"
                                        "    >>> u = sam.field.scalar(mesh, \"u\")\n"
                                        "    >>> vel = sam.field.vector(mesh, \"vel\", n_components=2)\n");

    // Register Field types (internal, with _ prefix) for factory function return types
    bind_scalar_field<1>(field, "_ScalarField1D");
    bind_scalar_field<2>(field, "_ScalarField2D");
    bind_scalar_field<3>(field, "_ScalarField3D");

    // Register VectorField types (internal, with _ prefix) for factory function return types
    bind_vector_field<1, 2, false>(field, "_VectorField1D_2");
    bind_vector_field<1, 3, false>(field, "_VectorField1D_3");
    bind_vector_field<2, 2, false>(field, "_VectorField2D_2");
    bind_vector_field<2, 3, false>(field, "_VectorField2D_3");
    bind_vector_field<3, 2, false>(field, "_VectorField3D_2");
    bind_vector_field<3, 3, false>(field, "_VectorField3D_3");

    // ============================================================
    // Factory functions for creating fields
    // ============================================================

    // ----- Scalar field factories (dimension inferred from mesh) -----
    field.def(
        "scalar",
        [](MRMesh<1>& mesh, const std::string& name, double init)
        {
            return samurai::make_scalar_field<double>(name, mesh, init);
        },
        py::arg("mesh"),
        py::arg("name"),
        py::arg("init") = 0.0,
        R"pbdoc(
        Create a 1D scalar field.

        Parameters
        ----------
        mesh : MRMesh1D
            The mesh to define the field on
        name : str
            Field identifier
        init : float, optional
            Initial value for all cells (default: 0.0)

        Returns
        -------
        ScalarField1D
            The created scalar field

        Examples
        --------
        >>> import samurai_python as sam
        >>> mesh = sam.MRMesh1D(box, config)
        >>> u = sam.field.scalar(mesh, "u")
        >>> u = sam.field.scalar(mesh, "u", init=1.0)
        )pbdoc");

    field.def(
        "scalar",
        [](MRMesh<2>& mesh, const std::string& name, double init)
        {
            return samurai::make_scalar_field<double>(name, mesh, init);
        },
        py::arg("mesh"),
        py::arg("name"),
        py::arg("init") = 0.0,
        "Create a 2D scalar field.\n\n"
        "See 1D version for detailed documentation.");

    field.def(
        "scalar",
        [](MRMesh<3>& mesh, const std::string& name, double init)
        {
            return samurai::make_scalar_field<double>(name, mesh, init);
        },
        py::arg("mesh"),
        py::arg("name"),
        py::arg("init") = 0.0,
        "Create a 3D scalar field.\n\n"
        "See 1D version for detailed documentation.");

    // ----- Vector field factories (dimension inferred from mesh) -----
    field.def(
        "vector",
        [](MRMesh<1>& mesh, const std::string& name, std::size_t n_components, double init) -> py::object
        {
            if (n_components == 2)
            {
                auto field_obj = samurai::make_vector_field<double, 2, false>(name, mesh, init);
                return py::cast(std::move(field_obj));
            }
            else if (n_components == 3)
            {
                auto field_obj = samurai::make_vector_field<double, 3, false>(name, mesh, init);
                return py::cast(std::move(field_obj));
            }
            else
            {
                throw std::runtime_error("n_components must be 2 or 3, got: " + std::to_string(n_components));
            }
        },
        py::arg("mesh"),
        py::arg("name"),
        py::arg("n_components") = 2,
        py::arg("init")         = 0.0,
        R"pbdoc(
        Create a 1D vector field.

        Parameters
        ----------
        mesh : MRMesh1D
            The mesh to define the field on
        name : str
            Field identifier
        n_components : int, optional
            Number of components (2 or 3, default: 2)
        init : float, optional
            Initial value for all cells (default: 0.0)

        Returns
        -------
        VectorField1D_2 or VectorField1D_3
            The created vector field

        Examples
        --------
        >>> import samurai_python as sam
        >>> mesh = sam.MRMesh1D(box, config)
        >>> vel = sam.field.vector(mesh, "velocity", n_components=2)
        >>> vel = sam.field.vector(mesh, "velocity", n_components=3, init=1.0)
        )pbdoc");

    field.def(
        "vector",
        [](MRMesh<2>& mesh, const std::string& name, std::size_t n_components, double init) -> py::object
        {
            if (n_components == 2)
            {
                auto field_obj = samurai::make_vector_field<double, 2, false>(name, mesh, init);
                return py::cast(std::move(field_obj));
            }
            else if (n_components == 3)
            {
                auto field_obj = samurai::make_vector_field<double, 3, false>(name, mesh, init);
                return py::cast(std::move(field_obj));
            }
            else
            {
                throw std::runtime_error("n_components must be 2 or 3, got: " + std::to_string(n_components));
            }
        },
        py::arg("mesh"),
        py::arg("name"),
        py::arg("n_components") = 2,
        py::arg("init")         = 0.0,
        "Create a 2D vector field.\n\n"
        "See 1D version for detailed documentation.");

    field.def(
        "vector",
        [](MRMesh<3>& mesh, const std::string& name, std::size_t n_components, double init) -> py::object
        {
            if (n_components == 2)
            {
                auto field_obj = samurai::make_vector_field<double, 2, false>(name, mesh, init);
                return py::cast(std::move(field_obj));
            }
            else if (n_components == 3)
            {
                auto field_obj = samurai::make_vector_field<double, 3, false>(name, mesh, init);
                return py::cast(std::move(field_obj));
            }
            else
            {
                throw std::runtime_error("n_components must be 2 or 3, got: " + std::to_string(n_components));
            }
        },
        py::arg("mesh"),
        py::arg("name"),
        py::arg("n_components") = 2,
        py::arg("init")         = 0.0,
        "Create a 3D vector field.\n\n"
        "See 1D version for detailed documentation.");

    // ============================================================
    // NumPy-style field creation helpers - Scalar Fields
    // ============================================================

    // === zeros() - create scalar field initialized to 0.0 ===
    field.def(
        "zeros",
        [](MRMesh<1>& mesh, const std::string& name)
        {
            return samurai::make_scalar_field<double>(name, mesh, 0.0);
        },
        py::arg("mesh"),
        py::arg("name") = "zeros",
        R"pbdoc(
        Create a scalar field initialized to zeros.

        Equivalent to sam.field.scalar(mesh, name, init=0.0).

        Parameters
        ----------
        mesh : MRMesh1D
            Mesh to define field on
        name : str, optional
            Field name (default: "zeros")

        Returns
        -------
        ScalarField1D
            Field with all values set to 0.0

        Examples
        --------
        >>> u = sam.field.zeros(mesh, "u")
        )pbdoc");

    field.def(
        "zeros",
        [](MRMesh<2>& mesh, const std::string& name)
        {
            return samurai::make_scalar_field<double>(name, mesh, 0.0);
        },
        py::arg("mesh"),
        py::arg("name") = "zeros",
        "Create a 2D scalar field initialized to zeros");

    field.def(
        "zeros",
        [](MRMesh<3>& mesh, const std::string& name)
        {
            return samurai::make_scalar_field<double>(name, mesh, 0.0);
        },
        py::arg("mesh"),
        py::arg("name") = "zeros",
        "Create a 3D scalar field initialized to zeros");

    // === ones() - create scalar field initialized to 1.0 ===
    field.def(
        "ones",
        [](MRMesh<1>& mesh, const std::string& name)
        {
            return samurai::make_scalar_field<double>(name, mesh, 1.0);
        },
        py::arg("mesh"),
        py::arg("name") = "ones",
        "Create a 1D scalar field initialized to ones");

    field.def(
        "ones",
        [](MRMesh<2>& mesh, const std::string& name)
        {
            return samurai::make_scalar_field<double>(name, mesh, 1.0);
        },
        py::arg("mesh"),
        py::arg("name") = "ones",
        "Create a 2D scalar field initialized to ones");

    field.def(
        "ones",
        [](MRMesh<3>& mesh, const std::string& name)
        {
            return samurai::make_scalar_field<double>(name, mesh, 1.0);
        },
        py::arg("mesh"),
        py::arg("name") = "ones",
        "Create a 3D scalar field initialized to ones");

    // === full() - create scalar field filled with specified value ===
    field.def(
        "full",
        [](MRMesh<1>& mesh, double fill_value, const std::string& name)
        {
            return samurai::make_scalar_field<double>(name, mesh, fill_value);
        },
        py::arg("mesh"),
        py::arg("fill_value"),
        py::arg("name") = "full",
        "Create a 1D scalar field filled with specified value");

    field.def(
        "full",
        [](MRMesh<2>& mesh, double fill_value, const std::string& name)
        {
            return samurai::make_scalar_field<double>(name, mesh, fill_value);
        },
        py::arg("mesh"),
        py::arg("fill_value"),
        py::arg("name") = "full",
        "Create a 2D scalar field filled with specified value");

    field.def(
        "full",
        [](MRMesh<3>& mesh, double fill_value, const std::string& name)
        {
            return samurai::make_scalar_field<double>(name, mesh, fill_value);
        },
        py::arg("mesh"),
        py::arg("fill_value"),
        py::arg("name") = "full",
        "Create a 3D scalar field filled with specified value");

    // === zeros_like() - create scalar field like another but with zeros ===
    field.def(
        "zeros_like",
        [](const ScalarField<1>& other, const std::string& name)
        {
            auto& mesh = const_cast<MRMesh<1>&>(other.mesh());
            return samurai::make_scalar_field<double>(name, mesh, 0.0);
        },
        py::arg("other"),
        py::arg("name") = "zeros_like",
        "Create a 1D scalar field like another but initialized to 0.0");

    field.def(
        "zeros_like",
        [](const ScalarField<2>& other, const std::string& name)
        {
            auto& mesh = const_cast<MRMesh<2>&>(other.mesh());
            return samurai::make_scalar_field<double>(name, mesh, 0.0);
        },
        py::arg("other"),
        py::arg("name") = "zeros_like",
        "Create a 2D scalar field like another but initialized to 0.0");

    field.def(
        "zeros_like",
        [](const ScalarField<3>& other, const std::string& name)
        {
            auto& mesh = const_cast<MRMesh<3>&>(other.mesh());
            return samurai::make_scalar_field<double>(name, mesh, 0.0);
        },
        py::arg("other"),
        py::arg("name") = "zeros_like",
        "Create a 3D scalar field like another but initialized to 0.0");

    // === ones_like() - create scalar field like another but with ones ===
    field.def(
        "ones_like",
        [](const ScalarField<1>& other, const std::string& name)
        {
            auto& mesh = const_cast<MRMesh<1>&>(other.mesh());
            return samurai::make_scalar_field<double>(name, mesh, 1.0);
        },
        py::arg("other"),
        py::arg("name") = "ones_like",
        "Create a 1D scalar field like another but initialized to 1.0");

    field.def(
        "ones_like",
        [](const ScalarField<2>& other, const std::string& name)
        {
            auto& mesh = const_cast<MRMesh<2>&>(other.mesh());
            return samurai::make_scalar_field<double>(name, mesh, 1.0);
        },
        py::arg("other"),
        py::arg("name") = "ones_like",
        "Create a 2D scalar field like another but initialized to 1.0");

    field.def(
        "ones_like",
        [](const ScalarField<3>& other, const std::string& name)
        {
            auto& mesh = const_cast<MRMesh<3>&>(other.mesh());
            return samurai::make_scalar_field<double>(name, mesh, 1.0);
        },
        py::arg("other"),
        py::arg("name") = "ones_like",
        "Create a 3D scalar field like another but initialized to 1.0");

    // === full_like() - create scalar field like another with specified value ===
    field.def(
        "full_like",
        [](const ScalarField<1>& other, double fill_value, const std::string& name)
        {
            auto& mesh = const_cast<MRMesh<1>&>(other.mesh());
            return samurai::make_scalar_field<double>(name, mesh, fill_value);
        },
        py::arg("other"),
        py::arg("fill_value"),
        py::arg("name") = "full_like",
        "Create a 1D scalar field like another with specified value");

    field.def(
        "full_like",
        [](const ScalarField<2>& other, double fill_value, const std::string& name)
        {
            auto& mesh = const_cast<MRMesh<2>&>(other.mesh());
            return samurai::make_scalar_field<double>(name, mesh, fill_value);
        },
        py::arg("other"),
        py::arg("fill_value"),
        py::arg("name") = "full_like",
        "Create a 2D scalar field like another with specified value");

    field.def(
        "full_like",
        [](const ScalarField<3>& other, double fill_value, const std::string& name)
        {
            auto& mesh = const_cast<MRMesh<3>&>(other.mesh());
            return samurai::make_scalar_field<double>(name, mesh, fill_value);
        },
        py::arg("other"),
        py::arg("fill_value"),
        py::arg("name") = "full_like",
        "Create a 3D scalar field like another with specified value");

    // ============================================================
    // NumPy-style field creation helpers - Vector Fields
    // ============================================================

    // Helper lambda to create vector field (returns py::object for type erasure)
    auto make_zeros_vector_1d = [](MRMesh<1>& mesh, const std::string& name, std::size_t n_components) -> py::object
    {
        if (n_components == 2)
        {
            auto field = samurai::make_vector_field<double, 2, false>(name, mesh, 0.0);
            return py::cast(std::move(field));
        }
        else if (n_components == 3)
        {
            auto field = samurai::make_vector_field<double, 3, false>(name, mesh, 0.0);
            return py::cast(std::move(field));
        }
        else
        {
            throw std::runtime_error("n_components must be 2 or 3, got: " + std::to_string(n_components));
        }
    };

    auto make_zeros_vector_2d = [](MRMesh<2>& mesh, const std::string& name, std::size_t n_components) -> py::object
    {
        if (n_components == 2)
        {
            auto field = samurai::make_vector_field<double, 2, false>(name, mesh, 0.0);
            return py::cast(std::move(field));
        }
        else if (n_components == 3)
        {
            auto field = samurai::make_vector_field<double, 3, false>(name, mesh, 0.0);
            return py::cast(std::move(field));
        }
        else
        {
            throw std::runtime_error("n_components must be 2 or 3, got: " + std::to_string(n_components));
        }
    };

    auto make_zeros_vector_3d = [](MRMesh<3>& mesh, const std::string& name, std::size_t n_components) -> py::object
    {
        if (n_components == 2)
        {
            auto field = samurai::make_vector_field<double, 2, false>(name, mesh, 0.0);
            return py::cast(std::move(field));
        }
        else if (n_components == 3)
        {
            auto field = samurai::make_vector_field<double, 3, false>(name, mesh, 0.0);
            return py::cast(std::move(field));
        }
        else
        {
            throw std::runtime_error("n_components must be 2 or 3, got: " + std::to_string(n_components));
        }
    };

    // === zeros_vector() - create vector field with zeros ===
    field.def("zeros_vector",
              make_zeros_vector_1d,
              py::arg("mesh"),
              py::arg("name")         = "vel",
              py::arg("n_components") = 2,
              R"pbdoc(
        Create a vector field filled with zeros.

        Parameters
        ----------
        mesh : MRMesh
            Mesh to define field on
        name : str, optional
            Field name (default: "vel")
        n_components : int, optional
            Number of components (2 or 3, default: 2)

        Returns
        -------
        VectorField
            Vector field with all components set to 0.0

        Examples
        --------
        >>> vel = sam.field.zeros_vector(mesh, n_components=2)
        >>> B = sam.field.zeros_vector(mesh, "B", n_components=3)
        )pbdoc");

    field.def("zeros_vector",
              make_zeros_vector_2d,
              py::arg("mesh"),
              py::arg("name")         = "vel",
              py::arg("n_components") = 2,
              "Create a 2D vector field filled with zeros");

    field.def("zeros_vector",
              make_zeros_vector_3d,
              py::arg("mesh"),
              py::arg("name")         = "vel",
              py::arg("n_components") = 2,
              "Create a 3D vector field filled with zeros");

    // === ones_vector() - create vector field with ones ===
    auto make_ones_vector_1d = [](MRMesh<1>& mesh, const std::string& name, std::size_t n_components) -> py::object
    {
        if (n_components == 2)
        {
            auto field = samurai::make_vector_field<double, 2, false>(name, mesh, 1.0);
            return py::cast(std::move(field));
        }
        else if (n_components == 3)
        {
            auto field = samurai::make_vector_field<double, 3, false>(name, mesh, 1.0);
            return py::cast(std::move(field));
        }
        else
        {
            throw std::runtime_error("n_components must be 2 or 3, got: " + std::to_string(n_components));
        }
    };

    auto make_ones_vector_2d = [](MRMesh<2>& mesh, const std::string& name, std::size_t n_components) -> py::object
    {
        if (n_components == 2)
        {
            auto field = samurai::make_vector_field<double, 2, false>(name, mesh, 1.0);
            return py::cast(std::move(field));
        }
        else if (n_components == 3)
        {
            auto field = samurai::make_vector_field<double, 3, false>(name, mesh, 1.0);
            return py::cast(std::move(field));
        }
        else
        {
            throw std::runtime_error("n_components must be 2 or 3, got: " + std::to_string(n_components));
        }
    };

    auto make_ones_vector_3d = [](MRMesh<3>& mesh, const std::string& name, std::size_t n_components) -> py::object
    {
        if (n_components == 2)
        {
            auto field = samurai::make_vector_field<double, 2, false>(name, mesh, 1.0);
            return py::cast(std::move(field));
        }
        else if (n_components == 3)
        {
            auto field = samurai::make_vector_field<double, 3, false>(name, mesh, 1.0);
            return py::cast(std::move(field));
        }
        else
        {
            throw std::runtime_error("n_components must be 2 or 3, got: " + std::to_string(n_components));
        }
    };

    field.def("ones_vector",
              make_ones_vector_1d,
              py::arg("mesh"),
              py::arg("name")         = "vel",
              py::arg("n_components") = 2,
              "Create a 1D vector field filled with ones");

    field.def("ones_vector",
              make_ones_vector_2d,
              py::arg("mesh"),
              py::arg("name")         = "vel",
              py::arg("n_components") = 2,
              "Create a 2D vector field filled with ones");

    field.def("ones_vector",
              make_ones_vector_3d,
              py::arg("mesh"),
              py::arg("name")         = "vel",
              py::arg("n_components") = 2,
              "Create a 3D vector field filled with ones");

    // === full_vector() - create vector field filled with specified value ===
    auto make_full_vector_1d = [](MRMesh<1>& mesh, double fill_value, const std::string& name, std::size_t n_components) -> py::object
    {
        if (n_components == 2)
        {
            auto field = samurai::make_vector_field<double, 2, false>(name, mesh, fill_value);
            return py::cast(std::move(field));
        }
        else if (n_components == 3)
        {
            auto field = samurai::make_vector_field<double, 3, false>(name, mesh, fill_value);
            return py::cast(std::move(field));
        }
        else
        {
            throw std::runtime_error("n_components must be 2 or 3, got: " + std::to_string(n_components));
        }
    };

    auto make_full_vector_2d = [](MRMesh<2>& mesh, double fill_value, const std::string& name, std::size_t n_components) -> py::object
    {
        if (n_components == 2)
        {
            auto field = samurai::make_vector_field<double, 2, false>(name, mesh, fill_value);
            return py::cast(std::move(field));
        }
        else if (n_components == 3)
        {
            auto field = samurai::make_vector_field<double, 3, false>(name, mesh, fill_value);
            return py::cast(std::move(field));
        }
        else
        {
            throw std::runtime_error("n_components must be 2 or 3, got: " + std::to_string(n_components));
        }
    };

    auto make_full_vector_3d = [](MRMesh<3>& mesh, double fill_value, const std::string& name, std::size_t n_components) -> py::object
    {
        if (n_components == 2)
        {
            auto field = samurai::make_vector_field<double, 2, false>(name, mesh, fill_value);
            return py::cast(std::move(field));
        }
        else if (n_components == 3)
        {
            auto field = samurai::make_vector_field<double, 3, false>(name, mesh, fill_value);
            return py::cast(std::move(field));
        }
        else
        {
            throw std::runtime_error("n_components must be 2 or 3, got: " + std::to_string(n_components));
        }
    };

    field.def("full_vector",
              make_full_vector_1d,
              py::arg("mesh"),
              py::arg("fill_value"),
              py::arg("name")         = "vel",
              py::arg("n_components") = 2,
              "Create a 1D vector field filled with specified value");

    field.def("full_vector",
              make_full_vector_2d,
              py::arg("mesh"),
              py::arg("fill_value"),
              py::arg("name")         = "vel",
              py::arg("n_components") = 2,
              "Create a 2D vector field filled with specified value");

    field.def("full_vector",
              make_full_vector_3d,
              py::arg("mesh"),
              py::arg("fill_value"),
              py::arg("name")         = "vel",
              py::arg("n_components") = 2,
              "Create a 3D vector field filled with specified value");

    // === zeros_like_vector() - create vector field like another but with zeros ===
    // Use type erasure pattern to handle different VectorField types
    field.def(
        "zeros_like_vector",
        [](const py::object& other, const std::string& name) -> py::object
        {
            // For VectorField2D_2
            if (py::isinstance<VectorField<2, 2, false>>(other))
            {
                auto& field_obj = other.cast<VectorField<2, 2, false>&>();
                auto& mesh      = const_cast<MRMesh<2>&>(field_obj.mesh());
                auto new_field  = samurai::make_vector_field<double, 2, false>(name, mesh, 0.0);
                return py::cast(std::move(new_field));
            }
            // For VectorField2D_3
            else if (py::isinstance<VectorField<2, 3, false>>(other))
            {
                auto& field_obj = other.cast<VectorField<2, 3, false>&>();
                auto& mesh      = const_cast<MRMesh<2>&>(field_obj.mesh());
                auto new_field  = samurai::make_vector_field<double, 3, false>(name, mesh, 0.0);
                return py::cast(std::move(new_field));
            }
            // For VectorField3D_2
            else if (py::isinstance<VectorField<3, 2, false>>(other))
            {
                auto& field_obj = other.cast<VectorField<3, 2, false>&>();
                auto& mesh      = const_cast<MRMesh<3>&>(field_obj.mesh());
                auto new_field  = samurai::make_vector_field<double, 2, false>(name, mesh, 0.0);
                return py::cast(std::move(new_field));
            }
            // For VectorField3D_3
            else if (py::isinstance<VectorField<3, 3, false>>(other))
            {
                auto& field_obj = other.cast<VectorField<3, 3, false>&>();
                auto& mesh      = const_cast<MRMesh<3>&>(field_obj.mesh());
                auto new_field  = samurai::make_vector_field<double, 3, false>(name, mesh, 0.0);
                return py::cast(std::move(new_field));
            }
            // For VectorField1D_2
            else if (py::isinstance<VectorField<1, 2, false>>(other))
            {
                auto& field_obj = other.cast<VectorField<1, 2, false>&>();
                auto& mesh      = const_cast<MRMesh<1>&>(field_obj.mesh());
                auto new_field  = samurai::make_vector_field<double, 2, false>(name, mesh, 0.0);
                return py::cast(std::move(new_field));
            }
            // For VectorField1D_3
            else if (py::isinstance<VectorField<1, 3, false>>(other))
            {
                auto& field_obj = other.cast<VectorField<1, 3, false>&>();
                auto& mesh      = const_cast<MRMesh<1>&>(field_obj.mesh());
                auto new_field  = samurai::make_vector_field<double, 3, false>(name, mesh, 0.0);
                return py::cast(std::move(new_field));
            }
            else
            {
                throw std::runtime_error("zeros_like_vector: argument must be a VectorField");
            }
        },
        py::arg("other"),
        py::arg("name") = "vel2",
        R"pbdoc(
        Create a vector field like another but filled with zeros.

        Parameters
        ----------
        other : VectorField
            Template field to copy mesh and component count from
        name : str, optional
            Name for the new field (default: "vel2")

        Returns
        -------
        VectorField
            New vector field with same mesh and components as other, but with zeros

        Examples
        --------
        >>> vel = sam.field.zeros_vector(mesh, n_components=2)
        >>> vel2 = sam.field.zeros_like_vector(vel, "vel2")
        )pbdoc");

    // ============================================================
    // Array swap utilities for efficient time stepping
    // ============================================================
    // Note: These are needed because the internal array() method is not exposed to Python

    m.def(
        "swap_field_arrays_1d",
        [](ScalarField<1>& f1, ScalarField<1>& f2)
        {
            std::swap(f1.array(), f2.array());
            bool f1_ghosts      = f1.ghosts_updated();
            f1.ghosts_updated() = f2.ghosts_updated();
            f2.ghosts_updated() = f1_ghosts;
        },
        py::arg("f1"),
        py::arg("f2"),
        "Swap underlying data arrays of two 1D fields (efficient for time stepping)");

    m.def(
        "swap_field_arrays_2d",
        [](ScalarField<2>& f1, ScalarField<2>& f2)
        {
            std::swap(f1.array(), f2.array());
            bool f1_ghosts      = f1.ghosts_updated();
            f1.ghosts_updated() = f2.ghosts_updated();
            f2.ghosts_updated() = f1_ghosts;
        },
        py::arg("f1"),
        py::arg("f2"),
        "Swap underlying data arrays of two 2D fields (efficient for time stepping)");

    m.def(
        "swap_field_arrays_3d",
        [](ScalarField<3>& f1, ScalarField<3>& f2)
        {
            std::swap(f1.array(), f2.array());
            bool f1_ghosts      = f1.ghosts_updated();
            f1.ghosts_updated() = f2.ghosts_updated();
            f2.ghosts_updated() = f1_ghosts;
        },
        py::arg("f1"),
        py::arg("f2"),
        "Swap underlying data arrays of two 3D fields (efficient for time stepping)");
}
