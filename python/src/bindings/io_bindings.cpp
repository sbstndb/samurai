// Samurai Python Bindings - HDF5 I/O
//
// Bindings for save(), dump(), and load() functions for fields and meshes
// to enable Paraview visualization and checkpoint/restart functionality

#include <filesystem>
#include <fstream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <samurai/field.hpp>
#include <samurai/io/hdf5.hpp>
#include <samurai/io/restart.hpp>
#include <samurai/mesh_config.hpp>
#include <samurai/mr/mesh.hpp>

namespace py = pybind11;

// ============================================================
// Type aliases matching field_bindings.cpp pattern
// ============================================================

using default_interval = samurai::Interval<double, std::size_t>;

template <std::size_t dim>
using MRMesh = samurai::MRMesh<samurai::complete_mesh_config<samurai::mesh_config<dim>, samurai::MRMeshId>>;

template <std::size_t dim>
using ScalarField = samurai::ScalarField<MRMesh<dim>, double>;

template <std::size_t dim, std::size_t n_comp, bool SOA = false>
using VectorField = samurai::VectorField<MRMesh<dim>, double, n_comp, SOA>;

// Specific VectorField types for Burgers equation (n_comp == dim)
using VectorField2D_2 = VectorField<2, 2, false>;
using VectorField3D_3 = VectorField<3, 3, false>;

// ============================================================
// Helper to convert Python path/string to fs::path
// Supports pathlib.Path objects and PathLike protocol
// ============================================================

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

// ============================================================
// Helper to extract mesh from field (for single field case)
// ============================================================

template <std::size_t dim>
const MRMesh<dim>& extract_mesh(const ScalarField<dim>& field)
{
    return field.mesh();
}

// ============================================================
// Unified filepath parsing helper
// ============================================================

// Parse unified filepath into directory and filename
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
    std::string basename = filepath.stem().string();

    // If no directory, use current directory
    if (directory.empty())
    {
        directory = std::filesystem::current_path();
    }

    return {directory, basename};
}

// ============================================================
// Unified save() function wrappers - 1D
// ============================================================

// Save single field (1D) - unified filepath
void save_1d(const py::object& filepath_obj, const ScalarField<1>& field)
{
    auto [directory, basename] = parse_unified_filepath(filepath_obj);
    samurai::save(directory, basename, field.mesh(), field);
}

// Save two fields (1D) - unified filepath
void save_1d_2fields(const py::object& filepath_obj, const ScalarField<1>& field1, const ScalarField<1>& field2)
{
    auto [directory, basename] = parse_unified_filepath(filepath_obj);
    samurai::save(directory, basename, field1.mesh(), field1, field2);
}

// Save three fields (1D) - unified filepath
void save_1d_3fields(const py::object& filepath_obj,
                     const ScalarField<1>& field1,
                     const ScalarField<1>& field2,
                     const ScalarField<1>& field3)
{
    auto [directory, basename] = parse_unified_filepath(filepath_obj);
    samurai::save(directory, basename, field1.mesh(), field1, field2, field3);
}

// ============================================================
// Unified save() function wrappers - 2D
// ============================================================

void save_2d(const py::object& filepath_obj, const ScalarField<2>& field)
{
    auto [directory, basename] = parse_unified_filepath(filepath_obj);
    samurai::save(directory, basename, field.mesh(), field);
}

void save_2d_2fields(const py::object& filepath_obj, const ScalarField<2>& field1, const ScalarField<2>& field2)
{
    auto [directory, basename] = parse_unified_filepath(filepath_obj);
    samurai::save(directory, basename, field1.mesh(), field1, field2);
}

void save_2d_3fields(const py::object& filepath_obj,
                     const ScalarField<2>& field1,
                     const ScalarField<2>& field2,
                     const ScalarField<2>& field3)
{
    auto [directory, basename] = parse_unified_filepath(filepath_obj);
    samurai::save(directory, basename, field1.mesh(), field1, field2, field3);
}

// ============================================================
// Unified save() function wrappers - 3D
// ============================================================

void save_3d(const py::object& filepath_obj, const ScalarField<3>& field)
{
    auto [directory, basename] = parse_unified_filepath(filepath_obj);
    samurai::save(directory, basename, field.mesh(), field);
}

void save_3d_2fields(const py::object& filepath_obj, const ScalarField<3>& field1, const ScalarField<3>& field2)
{
    auto [directory, basename] = parse_unified_filepath(filepath_obj);
    samurai::save(directory, basename, field1.mesh(), field1, field2);
}

void save_3d_3fields(const py::object& filepath_obj,
                     const ScalarField<3>& field1,
                     const ScalarField<3>& field2,
                     const ScalarField<3>& field3)
{
    auto [directory, basename] = parse_unified_filepath(filepath_obj);
    samurai::save(directory, basename, field1.mesh(), field1, field2, field3);
}

// ============================================================
// Unified save() function wrappers - VectorField (2D and 3D)
// ============================================================

void save_2d_vector(const py::object& filepath_obj, const VectorField2D_2& field)
{
    auto [directory, basename] = parse_unified_filepath(filepath_obj);
    samurai::save(directory, basename, field.mesh(), field);
}

void save_3d_vector(const py::object& filepath_obj, const VectorField3D_3& field)
{
    auto [directory, basename] = parse_unified_filepath(filepath_obj);
    samurai::save(directory, basename, field.mesh(), field);
}

// ============================================================
// Unified dump() function wrappers - 1D
// ============================================================

void dump_1d(const py::object& filepath_obj, const ScalarField<1>& field)
{
    auto [directory, basename] = parse_unified_filepath(filepath_obj);
    samurai::dump(directory, basename, field.mesh(), field);
}

// ============================================================
// Unified dump() function wrappers - 2D
// ============================================================

void dump_2d(const py::object& filepath_obj, const ScalarField<2>& field)
{
    auto [directory, basename] = parse_unified_filepath(filepath_obj);
    samurai::dump(directory, basename, field.mesh(), field);
}

// ============================================================
// Unified dump() function wrappers - 3D
// ============================================================

void dump_3d(const py::object& filepath_obj, const ScalarField<3>& field)
{
    auto [directory, basename] = parse_unified_filepath(filepath_obj);
    samurai::dump(directory, basename, field.mesh(), field);
}

// ============================================================
// Unified dump() function wrappers - VectorField (2D and 3D)
// ============================================================

void dump_2d_vector(const py::object& filepath_obj, const VectorField2D_2& field)
{
    auto [directory, basename] = parse_unified_filepath(filepath_obj);
    samurai::dump(directory, basename, field.mesh(), field);
}

void dump_3d_vector(const py::object& filepath_obj, const VectorField3D_3& field)
{
    auto [directory, basename] = parse_unified_filepath(filepath_obj);
    samurai::dump(directory, basename, field.mesh(), field);
}

// ============================================================
// Unified load() function wrappers - 1D
// ============================================================

void load_1d(const py::object& filepath_obj, ScalarField<1>& field)
{
    auto [directory, basename] = parse_unified_filepath(filepath_obj);
    samurai::load(directory, basename, field.mesh(), field);
}

// ============================================================
// Unified load() function wrappers - 2D
// ============================================================

void load_2d(const py::object& filepath_obj, ScalarField<2>& field)
{
    auto [directory, basename] = parse_unified_filepath(filepath_obj);
    samurai::load(directory, basename, field.mesh(), field);
}

// ============================================================
// Unified load() function wrappers - 3D
// ============================================================

void load_3d(const py::object& filepath_obj, ScalarField<3>& field)
{
    auto [directory, basename] = parse_unified_filepath(filepath_obj);
    samurai::load(directory, basename, field.mesh(), field);
}

// ============================================================
// open_h5py() helper function - Open HDF5 files with h5py
// ============================================================

py::object open_h5py_wrapper(const py::object& filename_obj, const std::string& mode)
{
    // Convert filename to string, add .h5 extension if needed
    std::string filename = py::str(filename_obj);
    if (filename.size() < 3 || filename.substr(filename.size() - 3) != ".h5")
    {
        filename = filename + ".h5";
    }

    // Import h5py
    auto h5py = py::module_::import("h5py");
    auto File = h5py.attr("File");

    // Return h5py.File object
    return File(filename, mode);
}

// ============================================================
// Field method wrappers for save()
// ============================================================

template <std::size_t dim>
void field_method_save(const ScalarField<dim>& field, const py::object& filepath_obj)
{
    auto [directory, basename] = parse_unified_filepath(filepath_obj);
    samurai::save(directory, basename, field.mesh(), field);
}

template <std::size_t dim>
void field_method_save_vector(const VectorField2D_2& field, const py::object& filepath_obj)
{
    auto [directory, basename] = parse_unified_filepath(filepath_obj);
    samurai::save(directory, basename, field.mesh(), field);
}

template <std::size_t dim>
void field_method_save_vector3d(const VectorField3D_3& field, const py::object& filepath_obj)
{
    auto [directory, basename] = parse_unified_filepath(filepath_obj);
    samurai::save(directory, basename, field.mesh(), field);
}

// ============================================================
// Field method wrappers for dump()
// ============================================================

template <std::size_t dim>
void field_method_dump(const ScalarField<dim>& field, const py::object& filepath_obj)
{
    auto [directory, basename] = parse_unified_filepath(filepath_obj);
    samurai::dump(directory, basename, field.mesh(), field);
}

template <std::size_t dim>
void field_method_dump_vector(const VectorField2D_2& field, const py::object& filepath_obj)
{
    auto [directory, basename] = parse_unified_filepath(filepath_obj);
    samurai::dump(directory, basename, field.mesh(), field);
}

template <std::size_t dim>
void field_method_dump_vector3d(const VectorField3D_3& field, const py::object& filepath_obj)
{
    auto [directory, basename] = parse_unified_filepath(filepath_obj);
    samurai::dump(directory, basename, field.mesh(), field);
}

// ============================================================
// Field method wrappers for load()
// ============================================================

template <std::size_t dim>
void field_method_load(ScalarField<dim>& field, const py::object& filepath_obj)
{
    auto [directory, basename] = parse_unified_filepath(filepath_obj);
    samurai::load(directory, basename, field.mesh(), field);
}

// ============================================================
// Module initialization
// ============================================================

void init_io_bindings(py::module_& m)
{
    // ============================================================
    // save() function bindings - unified filepath API
    // ============================================================

    // 1D save() - single field
    m.def("save",
          &save_1d,
          py::arg("filepath"),
          py::arg("field"),
          R"pbdoc(
            Save 1D field mesh and data to HDF5 + XDMF for Paraview visualization.

            Parameters
            ----------
            filepath : str or Path
                Unified file path (e.g., "results/solution.h5")
                The directory and filename are extracted from this path.
            field : ScalarField1D
                Field to save

            Creates
            -------
            {directory}/{basename}.h5 - HDF5 data file
            {directory}/{basename}.xdmf - XDMF metadata file for Paraview

            Examples
            --------
            >>> import samurai_python as sam
            >>> samurai.save("results/solution.h5", field)
            >>> samurai.save("solution.h5", field)  # Current directory
            >>> samurai.save(Path("results/solution.h5"), field)
        )pbdoc");

    // 1D save() - two fields
    m.def("save",
          &save_1d_2fields,
          py::arg("filepath"),
          py::arg("field1"),
          py::arg("field2"),
          "Save 1D mesh and two fields to HDF5 + XDMF. "
          "Parameters: filepath (str/Path), field1 (ScalarField1D), field2 (ScalarField1D).");

    // 1D save() - three fields
    m.def("save",
          &save_1d_3fields,
          py::arg("filepath"),
          py::arg("field1"),
          py::arg("field2"),
          py::arg("field3"),
          "Save 1D mesh and three fields to HDF5 + XDMF. "
          "Parameters: filepath (str/Path), field1, field2, field3 (ScalarField1D).");

    // 2D save() - single field
    m.def("save",
          &save_2d,
          py::arg("filepath"),
          py::arg("field"),
          R"pbdoc(
            Save 2D field mesh and data to HDF5 + XDMF for Paraview visualization.

            Parameters
            ----------
            filepath : str or Path
                Unified file path (e.g., "results/solution.h5")
            field : ScalarField2D
                Field to save

            Examples
            --------
            >>> samurai.save("results/solution.h5", field)
        )pbdoc");

    // 2D save() - two fields
    m.def("save",
          &save_2d_2fields,
          py::arg("filepath"),
          py::arg("field1"),
          py::arg("field2"),
          "Save 2D mesh and two fields to HDF5 + XDMF. "
          "Parameters: filepath (str/Path), field1 (ScalarField2D), field2 (ScalarField2D).");

    // 2D save() - three fields
    m.def("save",
          &save_2d_3fields,
          py::arg("filepath"),
          py::arg("field1"),
          py::arg("field2"),
          py::arg("field3"),
          "Save 2D mesh and three fields to HDF5 + XDMF. "
          "Parameters: filepath (str/Path), field1, field2, field3 (ScalarField2D).");

    // 3D save() - single field
    m.def("save",
          &save_3d,
          py::arg("filepath"),
          py::arg("field"),
          R"pbdoc(
            Save 3D field mesh and data to HDF5 + XDMF for Paraview visualization.

            Parameters
            ----------
            filepath : str or Path
                Unified file path (e.g., "results/solution.h5")
            field : ScalarField3D
                Field to save

            Examples
            --------
            >>> samurai.save("results/solution.h5", field)
        )pbdoc");

    // 3D save() - two fields
    m.def("save",
          &save_3d_2fields,
          py::arg("filepath"),
          py::arg("field1"),
          py::arg("field2"),
          "Save 3D mesh and two fields to HDF5 + XDMF. "
          "Parameters: filepath (str/Path), field1 (ScalarField3D), field2 (ScalarField3D).");

    // 3D save() - three fields
    m.def("save",
          &save_3d_3fields,
          py::arg("filepath"),
          py::arg("field1"),
          py::arg("field2"),
          py::arg("field3"),
          "Save 3D mesh and three fields to HDF5 + XDMF. "
          "Parameters: filepath (str/Path), field1, field2, field3 (ScalarField3D).");

    // ============================================================
    // dump() function bindings - unified filepath API
    // ============================================================

    // 1D dump()
    m.def("dump",
          &dump_1d,
          py::arg("filepath"),
          py::arg("field"),
          R"pbdoc(
            Dump 1D field mesh and data to HDF5 for checkpoint/restart.

            Creates HDF5-only file (no XDMF metadata) for efficient
            checkpointing and restarting simulations.

            Parameters
            ----------
            filepath : str or Path
                Unified file path (e.g., "checkpoints/solution.h5")
            field : ScalarField1D
                Field to save

            Creates
            -------
            {directory}/{basename}.h5 - HDF5 restart file

            Examples
            --------
            >>> import samurai_python as sam
            >>> samurai.dump("checkpoints/solution.h5", field)
        )pbdoc");

    // 2D dump()
    m.def("dump",
          &dump_2d,
          py::arg("filepath"),
          py::arg("field"),
          "Dump 2D field mesh and data to HDF5 for checkpoint/restart. "
          "Parameters: filepath (str/Path), field (ScalarField2D).");

    // 3D dump()
    m.def("dump",
          &dump_3d,
          py::arg("filepath"),
          py::arg("field"),
          "Dump 3D field mesh and data to HDF5 for checkpoint/restart. "
          "Parameters: filepath (str/Path), field (ScalarField3D).");

    // ============================================================
    // VectorField save() bindings - unified filepath API
    // ============================================================

    m.def("save",
          &save_2d_vector,
          py::arg("filepath"),
          py::arg("field"),
          "Save 2D vector field (2 components) mesh and data to HDF5 + XDMF. "
          "Parameters: filepath (str/Path), field (VectorField2D).");

    m.def("save",
          &save_3d_vector,
          py::arg("filepath"),
          py::arg("field"),
          "Save 3D vector field (3 components) mesh and data to HDF5 + XDMF. "
          "Parameters: filepath (str/Path), field (VectorField3D).");

    // ============================================================
    // VectorField dump() bindings - unified filepath API
    // ============================================================

    m.def("dump",
          &dump_2d_vector,
          py::arg("filepath"),
          py::arg("field"),
          "Dump 2D vector field (2 components) mesh and data to HDF5 for checkpoint/restart. "
          "Parameters: filepath (str/Path), field (VectorField2D).");

    m.def("dump",
          &dump_3d_vector,
          py::arg("filepath"),
          py::arg("field"),
          "Dump 3D vector field (3 components) mesh and data to HDF5 for checkpoint/restart. "
          "Parameters: filepath (str/Path), field (VectorField3D).");

    // ============================================================
    // load() function bindings - unified filepath API
    // ============================================================

    // 1D load()
    m.def("load",
          &load_1d,
          py::arg("filepath"),
          py::arg("field"),
          R"pbdoc(
            Load 1D field mesh and data from HDF5 restart file.

            Parameters
            ----------
            filepath : str or Path
                Unified file path (e.g., "checkpoints/solution.h5")
            field : ScalarField1D
                Field object to load data into (will be modified)

            Reads
            ------
            {directory}/{basename}.h5 - HDF5 restart file

            Note
            ----
            The mesh and field objects will have their data replaced
            with the contents of the restart file. The field name
            must match the name used when creating the restart file.

            Examples
            --------
            >>> import samurai_python as sam
            >>> samurai.load("checkpoints/solution.h5", field)
        )pbdoc");

    // 2D load()
    m.def("load",
          &load_2d,
          py::arg("filepath"),
          py::arg("field"),
          "Load 2D field mesh and data from HDF5 restart file. "
          "Parameters: filepath (str/Path), field (ScalarField2D).");

    // 3D load()
    m.def("load",
          &load_3d,
          py::arg("filepath"),
          py::arg("field"),
          "Load 3D field mesh and data from HDF5 restart file. "
          "Parameters: filepath (str/Path), field (ScalarField3D).");

    // ============================================================
    // open_h5py() function binding
    // ============================================================

    m.def("open_h5py",
          &open_h5py_wrapper,
          py::arg("filename"),
          py::arg("mode") = "r",
          R"pbdoc(
            Open HDF5 file created by Samurai using h5py.

            This function opens an HDF5 file created by Samurai's save() or dump()
            functions and returns an h5py.File object for direct data access.

            Parameters
            ----------
            filename : str or Path
                File path to open (with or without .h5 extension)
            mode : str, default: 'r'
                File access mode:
                - 'r': Read-only (default)
                - 'r+': Read and write
                - 'w': Write (truncate existing file)

            Returns
            -------
            h5py.File
                h5py File object for direct HDF5 data access

            Examples
            --------
            >>> import samurai_python as sam
            >>> # Save a field
            >>> field.save("results/solution.h5")
            >>> # Open with h5py
            >>> with samurai.open_h5py("results/solution.h5") as f:
            ...     data = f["mesh/fields/u"][:]
            ...     points = f["mesh/points"][:]
            ...     print(f"Field min: {data.min()}, max: {data.max()}")

            Notes
            -----
            Requires h5py to be installed (pip install h5py).

            The HDF5 file structure is:
            /mesh/points - Cell coordinates (N x 3)
            /mesh/connectivity - Cell connectivity (N_cells x 2^dim)
            /mesh/fields/{field_name} - Field data

            See Also
            --------
            Field.save : Save field to HDF5
            Field.load : Load field from HDF5
        )pbdoc");
}
