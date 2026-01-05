// Field bindings for Python - Simplified proof of concept

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <samurai/uniform_mesh.hpp>
#include <samurai/box.hpp>
#include <vector>
#include <string>

namespace py = pybind11;

// ============================================================================
// Simplified Cell wrapper for Python
// ============================================================================
template <std::size_t dim>
struct PyCell {
    std::size_t level;
    std::vector<std::size_t> indices;
    std::vector<double> center;
    std::vector<double> corner;
    std::size_t index;

    PyCell() : level(0), index(0) {
        indices.resize(dim, 0);
        center.resize(dim, 0.0);
        corner.resize(dim, 0.0);
    }

    PyCell(std::size_t l, std::size_t idx) : level(l), index(idx) {
        indices.resize(dim, 0);
        center.resize(dim, 0.0);
        corner.resize(dim, 0.0);
    }
};

// ============================================================================
// Simplified ScalarField wrapper for Python
// ============================================================================
template <std::size_t dim, typename value_t = double>
class PyScalarField {
public:
    using mesh_t = samurai::UniformMesh<samurai::UniformConfig<dim>>;

    PyScalarField(mesh_t& mesh, std::string name)
        : m_mesh(mesh), m_name(name), m_data(mesh.nb_cells(samurai::UniformMeshId::cells)) {
    }

    // Element access by PyCell
    value_t get(const PyCell<dim>& cell) const {
        return m_data[cell.index];
    }

    void set(const PyCell<dim>& cell, value_t value) {
        m_data[cell.index] = value;
    }

    // Element access by linear index
    value_t get_index(std::size_t i) const {
        return m_data[i];
    }

    void set_index(std::size_t i, value_t value) {
        m_data[i] = value;
    }

    // Fill all values
    void fill(value_t value) {
        std::fill(m_data.begin(), m_data.end(), value);
    }

    // Get size
    std::size_t size() const {
        return m_data.size();
    }

    // Get name
    const std::string& name() const {
        return m_name;
    }

    void set_name(const std::string& name) {
        m_name = name;
    }

    // Get data as numpy array (copy)
    py::array_t<value_t> array() {
        py::array_t<value_t> result(static_cast<pybind11::ssize_t>(m_data.size()));
        auto buf = result.mutable_unchecked();
        for (std::size_t i = 0; i < m_data.size(); ++i) {
            buf(i) = m_data[i];
        }
        return result;
    }

    // Set data from numpy array
    void set_array(py::array_t<value_t> arr) {
        py::buffer_info buf = arr.request();
        if (static_cast<std::size_t>(buf.size) != m_data.size()) {
            throw std::runtime_error("Array size must match field size");
        }
        auto* ptr = static_cast<value_t*>(buf.ptr);
        std::copy(ptr, ptr + m_data.size(), m_data.begin());
    }

    // Create simplified cells for iteration (linear indexing)
    py::list cells() const {
        py::list result;
        for (std::size_t i = 0; i < m_data.size(); ++i) {
            PyCell<dim> cell(0, i); // Simplified: all cells at level 0 for now

            // Compute coordinates based on linear index
            if constexpr (dim == 2) {
                // Assume square grid for simplicity
                std::size_t n = static_cast<std::size_t>(std::sqrt(m_data.size()));
                std::size_t j = i / n;
                std::size_t ii = i % n;
                cell.indices = {ii, j};

                double h = 1.0 / n;
                cell.center = {(static_cast<double>(ii) + 0.5) * h, (static_cast<double>(j) + 0.5) * h};
                cell.corner = {static_cast<double>(ii) * h, static_cast<double>(j) * h};
            } else if constexpr (dim == 1) {
                cell.indices = {i};
                double h = 1.0 / m_data.size();
                cell.center = {(static_cast<double>(i) + 0.5) * h};
                cell.corner = {static_cast<double>(i) * h};
            }

            result.append(py::cast(cell));
        }
        return result;
    }

    // String representation
    std::string repr() const {
        return "ScalarField(name='" + m_name + "', size=" + std::to_string(m_data.size()) + ")";
    }

private:
    mesh_t& m_mesh;
    std::string m_name;
    std::vector<value_t> m_data;
};

// ============================================================================
// Module definition
// ============================================================================
PYBIND11_MODULE(samurai_fields, m) {
    m.doc() = "Samurai Field Python Bindings - Simplified Field Implementation";

    // Version info
    m.attr("__version__") = "0.1.0-dev";

    // ============================================================================
    // BIND MESH TYPES
    // ============================================================================
    using Mesh1D = samurai::UniformMesh<samurai::UniformConfig<1>>;
    using Mesh2D = samurai::UniformMesh<samurai::UniformConfig<2>>;

    py::class_<Mesh1D>(m, "UniformMesh1D")
        .def("nb_cells", [](const Mesh1D& mesh) {
            return mesh.nb_cells(samurai::UniformMeshId::cells);
        }, "Number of cells in the mesh")
        .def("__repr__", [](const Mesh1D& mesh) {
            return "UniformMesh1D(cells=" + std::to_string(mesh.nb_cells(samurai::UniformMeshId::cells)) + ")";
        });

    py::class_<Mesh2D>(m, "UniformMesh2D")
        .def("nb_cells", [](const Mesh2D& mesh) {
            return mesh.nb_cells(samurai::UniformMeshId::cells);
        }, "Number of cells in the mesh")
        .def("__repr__", [](const Mesh2D& mesh) {
            return "UniformMesh2D(cells=" + std::to_string(mesh.nb_cells(samurai::UniformMeshId::cells)) + ")";
        });

    // ============================================================================
    // BIND PYCELL FOR 1D AND 2D
    // ============================================================================
    py::class_<PyCell<1>>(m, "Cell1D")
        .def(py::init<>())
        .def_readwrite("level", &PyCell<1>::level)
        .def_readwrite("indices", &PyCell<1>::indices)
        .def_readwrite("center", &PyCell<1>::center)
        .def_readwrite("corner", &PyCell<1>::corner)
        .def_readwrite("index", &PyCell<1>::index)
        .def("__repr__", [](const PyCell<1>& c) {
            return "Cell1D(level=" + std::to_string(c.level) + ", index=" + std::to_string(c.index) + ")";
        });

    py::class_<PyCell<2>>(m, "Cell2D")
        .def(py::init<>())
        .def_readwrite("level", &PyCell<2>::level)
        .def_readwrite("indices", &PyCell<2>::indices)
        .def_readwrite("center", &PyCell<2>::center)
        .def_readwrite("corner", &PyCell<2>::corner)
        .def_readwrite("index", &PyCell<2>::index)
        .def("__repr__", [](const PyCell<2>& c) {
            return "Cell2D(level=" + std::to_string(c.level) + ", index=" + std::to_string(c.index) + ")";
        });

    // ============================================================================
    // BIND SCALAR FIELD
    // ============================================================================

    // 1D ScalarField
    py::class_<PyScalarField<1, double>>(m, "ScalarField1D")
        .def(py::init<Mesh1D&, std::string>(),
             py::arg("mesh"), py::arg("name"))
        .def("get", &PyScalarField<1, double>::get,
             "Get value at cell", py::arg("cell"))
        .def("set", &PyScalarField<1, double>::set,
             "Set value at cell", py::arg("cell"), py::arg("value"))
        .def("get_index", &PyScalarField<1, double>::get_index,
             "Get value by linear index", py::arg("index"))
        .def("set_index", &PyScalarField<1, double>::set_index,
             "Set value by linear index", py::arg("index"), py::arg("value"))
        .def("fill", &PyScalarField<1, double>::fill,
             "Fill all cells with value", py::arg("value"))
        .def("size", &PyScalarField<1, double>::size,
             "Number of cells in field")
        .def_property("name",
             &PyScalarField<1, double>::name,
             &PyScalarField<1, double>::set_name,
             "Field name")
        .def("array", &PyScalarField<1, double>::array,
             "Get data as numpy array")
        .def("set_array", &PyScalarField<1, double>::set_array,
             "Set data from numpy array", py::arg("array"))
        .def("cells", &PyScalarField<1, double>::cells,
             "Get list of cells for iteration")
        .def("__repr__", &PyScalarField<1, double>::repr);

    // 2D ScalarField
    py::class_<PyScalarField<2, double>>(m, "ScalarField2D")
        .def(py::init<Mesh2D&, std::string>(),
             py::arg("mesh"), py::arg("name"))
        .def("get", &PyScalarField<2, double>::get,
             "Get value at cell", py::arg("cell"))
        .def("set", &PyScalarField<2, double>::set,
             "Set value at cell", py::arg("cell"), py::arg("value"))
        .def("get_index", &PyScalarField<2, double>::get_index,
             "Get value by linear index", py::arg("index"))
        .def("set_index", &PyScalarField<2, double>::set_index,
             "Set value by linear index", py::arg("index"), py::arg("value"))
        .def("fill", &PyScalarField<2, double>::fill,
             "Fill all cells with value", py::arg("value"))
        .def("size", &PyScalarField<2, double>::size,
             "Number of cells in field")
        .def_property("name",
             &PyScalarField<2, double>::name,
             &PyScalarField<2, double>::set_name,
             "Field name")
        .def("array", &PyScalarField<2, double>::array,
             "Get data as numpy array")
        .def("set_array", &PyScalarField<2, double>::set_array,
             "Set data from numpy array", py::arg("array"))
        .def("cells", &PyScalarField<2, double>::cells,
             "Get list of cells for iteration")
        .def("__repr__", &PyScalarField<2, double>::repr);

    // ============================================================================
    // HELPER FUNCTIONS
    // ============================================================================

    // Create uniform mesh from box and level
    m.def("make_uniform_mesh_1d",
          [](py::array_t<double> min_arr, py::array_t<double> max_arr, std::size_t level) {
              auto min_buf = min_arr.request();
              auto max_buf = max_arr.request();
              if (min_buf.size != 1 || max_buf.size != 1) {
                  throw std::runtime_error("Box must be 1D");
              }
              auto* min_ptr = static_cast<double*>(min_buf.ptr);
              auto* max_ptr = static_cast<double*>(max_buf.ptr);
              samurai::Box<double, 1> box({min_ptr[0]}, {max_ptr[0]});
              return Mesh1D(box, level);
          },
          py::arg("min"), py::arg("max"), py::arg("level"),
          "Create a 1D uniform mesh");

    m.def("make_uniform_mesh_2d",
          [](py::array_t<double> min_arr, py::array_t<double> max_arr, std::size_t level) {
              auto min_buf = min_arr.request();
              auto max_buf = max_arr.request();
              if (min_buf.size != 2 || max_buf.size != 2) {
                  throw std::runtime_error("Box must be 2D");
              }
              auto* min_ptr = static_cast<double*>(min_buf.ptr);
              auto* max_ptr = static_cast<double*>(max_buf.ptr);
              samurai::Box<double, 2> box({min_ptr[0], min_ptr[1]}, {max_ptr[0], max_ptr[1]});
              return Mesh2D(box, level);
          },
          py::arg("min"), py::arg("max"), py::arg("level"),
          "Create a 2D uniform mesh");

    // Create scalar field
    m.def("make_scalar_field",
          [](Mesh2D& mesh, const std::string& name, double init_value = 0.0) {
              auto field = PyScalarField<2, double>(mesh, name);
              field.fill(init_value);
              return field;
          },
          py::arg("mesh"), py::arg("name"), py::arg("init_value") = 0.0,
          "Create a 2D scalar field");
}
