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
// save() function wrappers - 1D
// ============================================================

// Save with path, filename, and single field (1D)
void save_1d_path_field(const py::object& path_obj, const std::string& filename, const ScalarField<1>& field)
{
    auto path = to_fs_path(path_obj);
    samurai::save(path, filename, field.mesh(), field);
}

// Save with path, filename, and two fields (1D)
void save_1d_path_fields(const py::object& path_obj, const std::string& filename, const ScalarField<1>& field1, const ScalarField<1>& field2)
{
    auto path = to_fs_path(path_obj);
    samurai::save(path, filename, field1.mesh(), field1, field2);
}

// Save with path, filename, and three fields (1D)
void save_1d_path_fields3(const py::object& path_obj,
                          const std::string& filename,
                          const ScalarField<1>& field1,
                          const ScalarField<1>& field2,
                          const ScalarField<1>& field3)
{
    auto path = to_fs_path(path_obj);
    samurai::save(path, filename, field1.mesh(), field1, field2, field3);
}

// Save with filename only (current directory) - 1D, single field
void save_1d_file_field(const std::string& filename, const ScalarField<1>& field)
{
    samurai::save(filename, field.mesh(), field);
}

// Save with filename only - 1D, two fields
void save_1d_file_fields(const std::string& filename, const ScalarField<1>& field1, const ScalarField<1>& field2)
{
    samurai::save(filename, field1.mesh(), field1, field2);
}

// Save with filename only - 1D, three fields
void save_1d_file_fields3(const std::string& filename, const ScalarField<1>& field1, const ScalarField<1>& field2, const ScalarField<1>& field3)
{
    samurai::save(filename, field1.mesh(), field1, field2, field3);
}

// ============================================================
// save() function wrappers - 2D
// ============================================================

void save_2d_path_field(const py::object& path_obj, const std::string& filename, const ScalarField<2>& field)
{
    auto path = to_fs_path(path_obj);
    samurai::save(path, filename, field.mesh(), field);
}

void save_2d_path_fields(const py::object& path_obj, const std::string& filename, const ScalarField<2>& field1, const ScalarField<2>& field2)
{
    auto path = to_fs_path(path_obj);
    samurai::save(path, filename, field1.mesh(), field1, field2);
}

void save_2d_path_fields3(const py::object& path_obj,
                          const std::string& filename,
                          const ScalarField<2>& field1,
                          const ScalarField<2>& field2,
                          const ScalarField<2>& field3)
{
    auto path = to_fs_path(path_obj);
    samurai::save(path, filename, field1.mesh(), field1, field2, field3);
}

void save_2d_file_field(const std::string& filename, const ScalarField<2>& field)
{
    samurai::save(filename, field.mesh(), field);
}

void save_2d_file_fields(const std::string& filename, const ScalarField<2>& field1, const ScalarField<2>& field2)
{
    samurai::save(filename, field1.mesh(), field1, field2);
}

void save_2d_file_fields3(const std::string& filename, const ScalarField<2>& field1, const ScalarField<2>& field2, const ScalarField<2>& field3)
{
    samurai::save(filename, field1.mesh(), field1, field2, field3);
}

// ============================================================
// save() function wrappers - 3D
// ============================================================

void save_3d_path_field(const py::object& path_obj, const std::string& filename, const ScalarField<3>& field)
{
    auto path = to_fs_path(path_obj);
    samurai::save(path, filename, field.mesh(), field);
}

void save_3d_path_fields(const py::object& path_obj, const std::string& filename, const ScalarField<3>& field1, const ScalarField<3>& field2)
{
    auto path = to_fs_path(path_obj);
    samurai::save(path, filename, field1.mesh(), field1, field2);
}

void save_3d_path_fields3(const py::object& path_obj,
                          const std::string& filename,
                          const ScalarField<3>& field1,
                          const ScalarField<3>& field2,
                          const ScalarField<3>& field3)
{
    auto path = to_fs_path(path_obj);
    samurai::save(path, filename, field1.mesh(), field1, field2, field3);
}

void save_3d_file_field(const std::string& filename, const ScalarField<3>& field)
{
    samurai::save(filename, field.mesh(), field);
}

void save_3d_file_fields(const std::string& filename, const ScalarField<3>& field1, const ScalarField<3>& field2)
{
    samurai::save(filename, field1.mesh(), field1, field2);
}

void save_3d_file_fields3(const std::string& filename, const ScalarField<3>& field1, const ScalarField<3>& field2, const ScalarField<3>& field3)
{
    samurai::save(filename, field1.mesh(), field1, field2, field3);
}

// ============================================================
// save() function wrappers - VectorField (2D and 3D)
// ============================================================

void save_2d_path_field_vector(const py::object& path_obj, const std::string& filename, const VectorField2D_2& field)
{
    auto path = to_fs_path(path_obj);
    samurai::save(path, filename, field.mesh(), field);
}

void save_2d_file_field_vector(const std::string& filename, const VectorField2D_2& field)
{
    samurai::save(filename, field.mesh(), field);
}

void save_3d_path_field_vector(const py::object& path_obj, const std::string& filename, const VectorField3D_3& field)
{
    auto path = to_fs_path(path_obj);
    samurai::save(path, filename, field.mesh(), field);
}

void save_3d_file_field_vector(const std::string& filename, const VectorField3D_3& field)
{
    samurai::save(filename, field.mesh(), field);
}

// ============================================================
// dump() function wrappers - 1D
// ============================================================

void dump_1d_path_field(const py::object& path_obj, const std::string& filename, const ScalarField<1>& field)
{
    auto path = to_fs_path(path_obj);
    samurai::dump(path, filename, field.mesh(), field);
}

void dump_1d_file_field(const std::string& filename, const ScalarField<1>& field)
{
    samurai::dump(filename, field.mesh(), field);
}

// ============================================================
// dump() function wrappers - 2D
// ============================================================

void dump_2d_path_field(const py::object& path_obj, const std::string& filename, const ScalarField<2>& field)
{
    auto path = to_fs_path(path_obj);
    samurai::dump(path, filename, field.mesh(), field);
}

void dump_2d_file_field(const std::string& filename, const ScalarField<2>& field)
{
    samurai::dump(filename, field.mesh(), field);
}

// ============================================================
// dump() function wrappers - 3D
// ============================================================

void dump_3d_path_field(const py::object& path_obj, const std::string& filename, const ScalarField<3>& field)
{
    auto path = to_fs_path(path_obj);
    samurai::dump(path, filename, field.mesh(), field);
}

void dump_3d_file_field(const std::string& filename, const ScalarField<3>& field)
{
    samurai::dump(filename, field.mesh(), field);
}

// ============================================================
// dump() function wrappers - VectorField (2D and 3D)
// ============================================================

void dump_2d_path_field_vector(const py::object& path_obj, const std::string& filename, const VectorField2D_2& field)
{
    auto path = to_fs_path(path_obj);
    samurai::dump(path, filename, field.mesh(), field);
}

void dump_2d_file_field_vector(const std::string& filename, const VectorField2D_2& field)
{
    samurai::dump(filename, field.mesh(), field);
}

void dump_3d_path_field_vector(const py::object& path_obj, const std::string& filename, const VectorField3D_3& field)
{
    auto path = to_fs_path(path_obj);
    samurai::dump(path, filename, field.mesh(), field);
}

void dump_3d_file_field_vector(const std::string& filename, const VectorField3D_3& field)
{
    samurai::dump(filename, field.mesh(), field);
}

// ============================================================
// load() function wrappers - 1D
// ============================================================

void load_1d_path(const py::object& path_obj, const std::string& filename, ScalarField<1>& field)
{
    auto path = to_fs_path(path_obj);
    samurai::load(path, filename, field.mesh(), field);
}

void load_1d_file(const std::string& filename, ScalarField<1>& field)
{
    samurai::load(filename, field.mesh(), field);
}

// ============================================================
// load() function wrappers - 2D
// ============================================================

void load_2d_path(const py::object& path_obj, const std::string& filename, ScalarField<2>& field)
{
    auto path = to_fs_path(path_obj);
    samurai::load(path, filename, field.mesh(), field);
}

void load_2d_file(const std::string& filename, ScalarField<2>& field)
{
    samurai::load(filename, field.mesh(), field);
}

// ============================================================
// load() function wrappers - 3D
// ============================================================

void load_3d_path(const py::object& path_obj, const std::string& filename, ScalarField<3>& field)
{
    auto path = to_fs_path(path_obj);
    samurai::load(path, filename, field.mesh(), field);
}

void load_3d_file(const std::string& filename, ScalarField<3>& field)
{
    samurai::load(filename, field.mesh(), field);
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
// Field method helpers - Unified path parsing
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
    // save() function bindings
    // ============================================================

    // 1D save() - with path and filename
    m.def("save",
          &save_1d_path_field,
          py::arg("path"),
          py::arg("filename"),
          py::arg("field"),
          R"pbdoc(
            Save 1D field mesh and data to HDF5 + XDMF for Paraview visualization.

            Parameters
            ----------
            path : str or Path
                Output directory path (or None for current directory)
            filename : str
                Base filename (without .h5/.xdmf extension)
            field : ScalarField1D
                Field to save

            Creates
            -------
            {path}/{filename}.h5 - HDF5 data file
            {path}/{filename}.xdmf - XDMF metadata file for Paraview

            Examples
            --------
            >>> import samurai_python as sam
            >>> samurai.save("results", "solution", field)
            >>> # Or with None for current directory
            >>> samurai.save(None, "solution", field)
        )pbdoc");

    m.def("save",
          &save_1d_path_fields,
          py::arg("path"),
          py::arg("filename"),
          py::arg("field1"),
          py::arg("field2"),
          "Save 1D mesh and two fields to HDF5 + XDMF");

    m.def("save",
          &save_1d_path_fields3,
          py::arg("path"),
          py::arg("filename"),
          py::arg("field1"),
          py::arg("field2"),
          py::arg("field3"),
          "Save 1D mesh and three fields to HDF5 + XDMF");

    // 1D save() - filename only (current directory)
    m.def("save", &save_1d_file_field, py::arg("filename"), py::arg("field"), "Save 1D field to HDF5 + XDMF (current directory)");

    m.def("save",
          &save_1d_file_fields,
          py::arg("filename"),
          py::arg("field1"),
          py::arg("field2"),
          "Save 1D mesh and two fields to HDF5 + XDMF (current directory)");

    m.def("save",
          &save_1d_file_fields3,
          py::arg("filename"),
          py::arg("field1"),
          py::arg("field2"),
          py::arg("field3"),
          "Save 1D mesh and three fields to HDF5 + XDMF (current directory)");

    // 2D save() - with path and filename
    m.def("save",
          &save_2d_path_field,
          py::arg("path"),
          py::arg("filename"),
          py::arg("field"),
          "Save 2D field mesh and data to HDF5 + XDMF for Paraview visualization");

    m.def("save",
          &save_2d_path_fields,
          py::arg("path"),
          py::arg("filename"),
          py::arg("field1"),
          py::arg("field2"),
          "Save 2D mesh and two fields to HDF5 + XDMF");

    m.def("save",
          &save_2d_path_fields3,
          py::arg("path"),
          py::arg("filename"),
          py::arg("field1"),
          py::arg("field2"),
          py::arg("field3"),
          "Save 2D mesh and three fields to HDF5 + XDMF");

    // 2D save() - filename only
    m.def("save", &save_2d_file_field, py::arg("filename"), py::arg("field"), "Save 2D field to HDF5 + XDMF (current directory)");

    m.def("save",
          &save_2d_file_fields,
          py::arg("filename"),
          py::arg("field1"),
          py::arg("field2"),
          "Save 2D mesh and two fields to HDF5 + XDMF (current directory)");

    m.def("save",
          &save_2d_file_fields3,
          py::arg("filename"),
          py::arg("field1"),
          py::arg("field2"),
          py::arg("field3"),
          "Save 2D mesh and three fields to HDF5 + XDMF (current directory)");

    // 3D save() - with path and filename
    m.def("save",
          &save_3d_path_field,
          py::arg("path"),
          py::arg("filename"),
          py::arg("field"),
          "Save 3D field mesh and data to HDF5 + XDMF for Paraview visualization");

    m.def("save",
          &save_3d_path_fields,
          py::arg("path"),
          py::arg("filename"),
          py::arg("field1"),
          py::arg("field2"),
          "Save 3D mesh and two fields to HDF5 + XDMF");

    m.def("save",
          &save_3d_path_fields3,
          py::arg("path"),
          py::arg("filename"),
          py::arg("field1"),
          py::arg("field2"),
          py::arg("field3"),
          "Save 3D mesh and three fields to HDF5 + XDMF");

    // 3D save() - filename only
    m.def("save", &save_3d_file_field, py::arg("filename"), py::arg("field"), "Save 3D field to HDF5 + XDMF (current directory)");

    m.def("save",
          &save_3d_file_fields,
          py::arg("filename"),
          py::arg("field1"),
          py::arg("field2"),
          "Save 3D mesh and two fields to HDF5 + XDMF (current directory)");

    m.def("save",
          &save_3d_file_fields3,
          py::arg("filename"),
          py::arg("field1"),
          py::arg("field2"),
          py::arg("field3"),
          "Save 3D mesh and three fields to HDF5 + XDMF (current directory)");

    // ============================================================
    // dump() function bindings (checkpoint/restart format)
    // ============================================================

    // 1D dump()
    m.def("dump",
          &dump_1d_path_field,
          py::arg("path"),
          py::arg("filename"),
          py::arg("field"),
          R"pbdoc(
            Dump 1D field mesh and data to HDF5 for checkpoint/restart.

            Creates HDF5-only file (no XDMF metadata) for efficient
            checkpointing and restarting simulations.

            Parameters
            ----------
            path : str or Path
                Output directory path (or None for current directory)
            filename : str
                Base filename (without .h5 extension)
            field : ScalarField1D
                Field to save

            Creates
            -------
            {path}/{filename}.h5 - HDF5 restart file
        )pbdoc");

    m.def("dump", &dump_1d_file_field, py::arg("filename"), py::arg("field"), "Dump 1D field to HDF5 restart file (current directory)");

    // 2D dump()
    m.def("dump",
          &dump_2d_path_field,
          py::arg("path"),
          py::arg("filename"),
          py::arg("field"),
          "Dump 2D field mesh and data to HDF5 for checkpoint/restart");

    m.def("dump", &dump_2d_file_field, py::arg("filename"), py::arg("field"), "Dump 2D field to HDF5 restart file (current directory)");

    // 3D dump()
    m.def("dump",
          &dump_3d_path_field,
          py::arg("path"),
          py::arg("filename"),
          py::arg("field"),
          "Dump 3D field mesh and data to HDF5 for checkpoint/restart");

    m.def("dump", &dump_3d_file_field, py::arg("filename"), py::arg("field"), "Dump 3D field to HDF5 restart file (current directory)");

    // ============================================================
    // VectorField save() bindings
    // ============================================================

    m.def("save",
          &save_2d_path_field_vector,
          py::arg("path"),
          py::arg("filename"),
          py::arg("field"),
          "Save 2D vector field (2 components) mesh and data to HDF5 + XDMF");

    m.def("save", &save_2d_file_field_vector, py::arg("filename"), py::arg("field"), "Save 2D vector field to HDF5 + XDMF (current directory)");

    m.def("save",
          &save_3d_path_field_vector,
          py::arg("path"),
          py::arg("filename"),
          py::arg("field"),
          "Save 3D vector field (3 components) mesh and data to HDF5 + XDMF");

    m.def("save", &save_3d_file_field_vector, py::arg("filename"), py::arg("field"), "Save 3D vector field to HDF5 + XDMF (current directory)");

    // ============================================================
    // VectorField dump() bindings
    // ============================================================

    m.def("dump",
          &dump_2d_path_field_vector,
          py::arg("path"),
          py::arg("filename"),
          py::arg("field"),
          "Dump 2D vector field (2 components) mesh and data to HDF5 for checkpoint/restart");

    m.def("dump", &dump_2d_file_field_vector, py::arg("filename"), py::arg("field"), "Dump 2D vector field to HDF5 restart file (current directory)");

    m.def("dump",
          &dump_3d_path_field_vector,
          py::arg("path"),
          py::arg("filename"),
          py::arg("field"),
          "Dump 3D vector field (3 components) mesh and data to HDF5 for checkpoint/restart");

    m.def("dump", &dump_3d_file_field_vector, py::arg("filename"), py::arg("field"), "Dump 3D vector field to HDF5 restart file (current directory)");

    // ============================================================
    // load() function bindings (checkpoint/restart)
    // ============================================================

    // 1D load()
    m.def("load",
          &load_1d_path,
          py::arg("path"),
          py::arg("filename"),
          py::arg("field"),
          R"pbdoc(
            Load 1D field mesh and data from HDF5 restart file.

            Parameters
            ----------
            path : str or Path
                Directory containing the restart file
            filename : str
                Base filename (without .h5 extension)
            field : ScalarField1D
                Field object to load data into (will be modified)

            Reads
            ------
            {path}/{filename}.h5 - HDF5 restart file

            Note
            ----
            The mesh and field objects will have their data replaced
            with the contents of the restart file. The field name
            must match the name used when creating the restart file.
        )pbdoc");

    m.def("load", &load_1d_file, py::arg("filename"), py::arg("field"), "Load 1D field from HDF5 restart file (current directory)");

    // 2D load()
    m.def("load", &load_2d_path, py::arg("path"), py::arg("filename"), py::arg("field"), "Load 2D field mesh and data from HDF5 restart file");

    m.def("load", &load_2d_file, py::arg("filename"), py::arg("field"), "Load 2D field from HDF5 restart file (current directory)");

    // 3D load()
    m.def("load", &load_3d_path, py::arg("path"), py::arg("filename"), py::arg("field"), "Load 3D field mesh and data from HDF5 restart file");

    m.def("load", &load_3d_file, py::arg("filename"), py::arg("field"), "Load 3D field from HDF5 restart file (current directory)");

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
