// VectorField bindings for Python - Simplified implementation

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <samurai/uniform_mesh.hpp>
#include <samurai/box.hpp>
#include <vector>
#include <string>

namespace py = pybind11;

// ============================================================================
// Simplified Cell wrapper for Python (reused from field_bindings)
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
// Simplified VectorField wrapper for Python
// ============================================================================
template <std::size_t dim, typename value_t = double, std::size_t n_comp = 2>
class PyVectorField {
public:
    using mesh_t = samurai::UniformMesh<samurai::UniformConfig<dim>>;

    PyVectorField(mesh_t& mesh, std::string name)
        : m_mesh(mesh), m_name(name), m_data(mesh.nb_cells(samurai::UniformMeshId::cells) * n_comp) {
    }

    // Get all components at a cell
    std::vector<value_t> get(const PyCell<dim>& cell) const {
        std::vector<value_t> result(n_comp);
        for (std::size_t c = 0; c < n_comp; ++c) {
            result[c] = m_data[cell.index * n_comp + c];
        }
        return result;
    }

    // Set all components at a cell
    void set(const PyCell<dim>& cell, const std::vector<value_t>& values) {
        if (values.size() != n_comp) {
            throw std::runtime_error("Expected " + std::to_string(n_comp) + " components");
        }
        for (std::size_t c = 0; c < n_comp; ++c) {
            m_data[cell.index * n_comp + c] = values[c];
        }
    }

    // Get all components by linear index
    std::vector<value_t> get_index(std::size_t i) const {
        std::vector<value_t> result(n_comp);
        for (std::size_t c = 0; c < n_comp; ++c) {
            result[c] = m_data[i * n_comp + c];
        }
        return result;
    }

    // Set all components by linear index
    void set_index(std::size_t i, const std::vector<value_t>& values) {
        if (values.size() != n_comp) {
            throw std::runtime_error("Expected " + std::to_string(n_comp) + " components");
        }
        for (std::size_t c = 0; c < n_comp; ++c) {
            m_data[i * n_comp + c] = values[c];
        }
    }

    // Get single component by linear index
    value_t get_component(std::size_t i, std::size_t comp) const {
        if (comp >= n_comp) {
            throw std::runtime_error("Component index out of range");
        }
        return m_data[i * n_comp + comp];
    }

    // Set single component by linear index
    void set_component(std::size_t i, std::size_t comp, value_t value) {
        if (comp >= n_comp) {
            throw std::runtime_error("Component index out of range");
        }
        m_data[i * n_comp + comp] = value;
    }

    // Fill all components with value
    void fill(value_t value) {
        std::fill(m_data.begin(), m_data.end(), value);
    }

    // Fill specific component
    void fill_component(std::size_t comp, value_t value) {
        if (comp >= n_comp) {
            throw std::runtime_error("Component index out of range");
        }
        for (std::size_t i = 0; i < nb_cells(); ++i) {
            m_data[i * n_comp + comp] = value;
        }
    }

    // Get number of cells
    std::size_t nb_cells() const {
        return m_data.size() / n_comp;
    }

    // Get size (total number of values)
    std::size_t size() const {
        return m_data.size();
    }

    // Get number of components
    std::size_t n_components() const {
        return n_comp;
    }

    // Get name
    const std::string& name() const {
        return m_name;
    }

    void set_name(const std::string& name) {
        m_name = name;
    }

    // Get data as numpy array (2D: n_cells x n_comp)
    py::array_t<value_t> array() {
        std::size_t nc = nb_cells();
        py::array_t<value_t> result({static_cast<pybind11::ssize_t>(nc),
                                     static_cast<pybind11::ssize_t>(n_comp)});
        auto buf = result.mutable_unchecked();
        for (std::size_t i = 0; i < nc; ++i) {
            for (std::size_t c = 0; c < n_comp; ++c) {
                buf(i, c) = m_data[i * n_comp + c];
            }
        }
        return result;
    }

    // Set data from numpy array
    void set_array(py::array_t<value_t> arr) {
        py::buffer_info buf = arr.request();

        if (buf.ndim == 1) {
            // 1D array: treat as flattened (n_cells * n_comp)
            if (static_cast<std::size_t>(buf.size) != m_data.size()) {
                throw std::runtime_error("Array size must match field size (n_cells * n_comp)");
            }
            auto* ptr = static_cast<value_t*>(buf.ptr);
            std::copy(ptr, ptr + m_data.size(), m_data.begin());
        } else if (buf.ndim == 2) {
            // 2D array: n_cells x n_comp
            auto* ptr = static_cast<value_t*>(buf.ptr);
            std::size_t nc = nb_cells();
            if (static_cast<std::size_t>(buf.shape[0]) != nc ||
                static_cast<std::size_t>(buf.shape[1]) != n_comp) {
                throw std::runtime_error("Array shape must be (n_cells, n_comp)");
            }
            for (std::size_t i = 0; i < nc; ++i) {
                for (std::size_t c = 0; c < n_comp; ++c) {
                    m_data[i * n_comp + c] = ptr[i * buf.shape[1] + c];
                }
            }
        } else {
            throw std::runtime_error("Array must be 1D or 2D");
        }
    }

    // Create simplified cells for iteration
    py::list cells() const {
        py::list result;
        std::size_t nc = nb_cells();
        for (std::size_t i = 0; i < nc; ++i) {
            PyCell<dim> cell(0, i);

            if constexpr (dim == 2) {
                std::size_t n = static_cast<std::size_t>(std::sqrt(nc));
                std::size_t j = i / n;
                std::size_t ii = i % n;
                cell.indices = {ii, j};

                double h = 1.0 / n;
                cell.center = {(static_cast<double>(ii) + 0.5) * h, (static_cast<double>(j) + 0.5) * h};
                cell.corner = {static_cast<double>(ii) * h, static_cast<double>(j) * h};
            } else if constexpr (dim == 1) {
                cell.indices = {i};
                double h = 1.0 / nc;
                cell.center = {(static_cast<double>(i) + 0.5) * h};
                cell.corner = {static_cast<double>(i) * h};
            }

            result.append(py::cast(cell));
        }
        return result;
    }

    // String representation
    std::string repr() const {
        return "VectorField(name='" + m_name + "', n_comp=" + std::to_string(n_comp) +
               ", size=" + std::to_string(nb_cells()) + " cells)";
    }

private:
    mesh_t& m_mesh;
    std::string m_name;
    std::vector<value_t> m_data;
};

// ============================================================================
// Module definition
// ============================================================================
PYBIND11_MODULE(samurai_vector_fields, m) {
    m.doc() = "Samurai VectorField Python Bindings";

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
    // BIND PYCELL
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
    // BIND VECTOR FIELD 2 COMPONENTS
    // ============================================================================

    // 1D VectorField with 2 components
    py::class_<PyVectorField<1, double, 2>>(m, "VectorField1D_2")
        .def(py::init<Mesh1D&, std::string>(),
             py::arg("mesh"), py::arg("name"))
        .def("get", &PyVectorField<1, double, 2>::get,
             "Get all components at cell", py::arg("cell"))
        .def("set", &PyVectorField<1, double, 2>::set,
             "Set all components at cell", py::arg("cell"), py::arg("values"))
        .def("get_index", &PyVectorField<1, double, 2>::get_index,
             "Get all components by linear index", py::arg("index"))
        .def("set_index", &PyVectorField<1, double, 2>::set_index,
             "Set all components by linear index", py::arg("index"), py::arg("values"))
        .def("get_component", &PyVectorField<1, double, 2>::get_component,
             "Get single component", py::arg("index"), py::arg("comp"))
        .def("set_component", &PyVectorField<1, double, 2>::set_component,
             "Set single component", py::arg("index"), py::arg("comp"), py::arg("value"))
        .def("fill", &PyVectorField<1, double, 2>::fill,
             "Fill all components with value", py::arg("value"))
        .def("fill_component", &PyVectorField<1, double, 2>::fill_component,
             "Fill specific component", py::arg("comp"), py::arg("value"))
        .def("nb_cells", &PyVectorField<1, double, 2>::nb_cells,
             "Number of cells")
        .def("size", &PyVectorField<1, double, 2>::size,
             "Total number of values")
        .def("n_components", &PyVectorField<1, double, 2>::n_components,
             "Number of components per cell")
        .def_property("name",
             &PyVectorField<1, double, 2>::name,
             &PyVectorField<1, double, 2>::set_name,
             "Field name")
        .def("array", &PyVectorField<1, double, 2>::array,
             "Get data as numpy array (n_cells, n_comp)")
        .def("set_array", &PyVectorField<1, double, 2>::set_array,
             "Set data from numpy array", py::arg("array"))
        .def("cells", &PyVectorField<1, double, 2>::cells,
             "Get list of cells for iteration")
        .def("__repr__", &PyVectorField<1, double, 2>::repr);

    // 2D VectorField with 2 components
    py::class_<PyVectorField<2, double, 2>>(m, "VectorField2D_2")
        .def(py::init<Mesh2D&, std::string>(),
             py::arg("mesh"), py::arg("name"))
        .def("get", &PyVectorField<2, double, 2>::get,
             "Get all components at cell", py::arg("cell"))
        .def("set", &PyVectorField<2, double, 2>::set,
             "Set all components at cell", py::arg("cell"), py::arg("values"))
        .def("get_index", &PyVectorField<2, double, 2>::get_index,
             "Get all components by linear index", py::arg("index"))
        .def("set_index", &PyVectorField<2, double, 2>::set_index,
             "Set all components by linear index", py::arg("index"), py::arg("values"))
        .def("get_component", &PyVectorField<2, double, 2>::get_component,
             "Get single component", py::arg("index"), py::arg("comp"))
        .def("set_component", &PyVectorField<2, double, 2>::set_component,
             "Set single component", py::arg("index"), py::arg("comp"), py::arg("value"))
        .def("fill", &PyVectorField<2, double, 2>::fill,
             "Fill all components with value", py::arg("value"))
        .def("fill_component", &PyVectorField<2, double, 2>::fill_component,
             "Fill specific component", py::arg("comp"), py::arg("value"))
        .def("nb_cells", &PyVectorField<2, double, 2>::nb_cells,
             "Number of cells")
        .def("size", &PyVectorField<2, double, 2>::size,
             "Total number of values")
        .def("n_components", &PyVectorField<2, double, 2>::n_components,
             "Number of components per cell")
        .def_property("name",
             &PyVectorField<2, double, 2>::name,
             &PyVectorField<2, double, 2>::set_name,
             "Field name")
        .def("array", &PyVectorField<2, double, 2>::array,
             "Get data as numpy array (n_cells, n_comp)")
        .def("set_array", &PyVectorField<2, double, 2>::set_array,
             "Set data from numpy array", py::arg("array"))
        .def("cells", &PyVectorField<2, double, 2>::cells,
             "Get list of cells for iteration")
        .def("__repr__", &PyVectorField<2, double, 2>::repr);

    // ============================================================================
    // BIND VECTOR FIELD 3 COMPONENTS
    // ============================================================================

    // 2D VectorField with 3 components
    py::class_<PyVectorField<2, double, 3>>(m, "VectorField2D_3")
        .def(py::init<Mesh2D&, std::string>(),
             py::arg("mesh"), py::arg("name"))
        .def("get", &PyVectorField<2, double, 3>::get,
             "Get all components at cell", py::arg("cell"))
        .def("set", &PyVectorField<2, double, 3>::set,
             "Set all components at cell", py::arg("cell"), py::arg("values"))
        .def("get_index", &PyVectorField<2, double, 3>::get_index,
             "Get all components by linear index", py::arg("index"))
        .def("set_index", &PyVectorField<2, double, 3>::set_index,
             "Set all components by linear index", py::arg("index"), py::arg("values"))
        .def("get_component", &PyVectorField<2, double, 3>::get_component,
             "Get single component", py::arg("index"), py::arg("comp"))
        .def("set_component", &PyVectorField<2, double, 3>::set_component,
             "Set single component", py::arg("index"), py::arg("comp"), py::arg("value"))
        .def("fill", &PyVectorField<2, double, 3>::fill,
             "Fill all components with value", py::arg("value"))
        .def("fill_component", &PyVectorField<2, double, 3>::fill_component,
             "Fill specific component", py::arg("comp"), py::arg("value"))
        .def("nb_cells", &PyVectorField<2, double, 3>::nb_cells,
             "Number of cells")
        .def("size", &PyVectorField<2, double, 3>::size,
             "Total number of values")
        .def("n_components", &PyVectorField<2, double, 3>::n_components,
             "Number of components per cell")
        .def_property("name",
             &PyVectorField<2, double, 3>::name,
             &PyVectorField<2, double, 3>::set_name,
             "Field name")
        .def("array", &PyVectorField<2, double, 3>::array,
             "Get data as numpy array (n_cells, n_comp)")
        .def("set_array", &PyVectorField<2, double, 3>::set_array,
             "Set data from numpy array", py::arg("array"))
        .def("cells", &PyVectorField<2, double, 3>::cells,
             "Get list of cells for iteration")
        .def("__repr__", &PyVectorField<2, double, 3>::repr);

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

    // Create vector field factory functions
    m.def("make_vector_field_2",
          [](Mesh2D& mesh, const std::string& name, double init_value = 0.0) {
              auto field = PyVectorField<2, double, 2>(mesh, name);
              field.fill(init_value);
              return field;
          },
          py::arg("mesh"), py::arg("name"), py::arg("init_value") = 0.0,
          "Create a 2D vector field with 2 components");

    m.def("make_vector_field_3",
          [](Mesh2D& mesh, const std::string& name, double init_value = 0.0) {
              auto field = PyVectorField<2, double, 3>(mesh, name);
              field.fill(init_value);
              return field;
          },
          py::arg("mesh"), py::arg("name"), py::arg("init_value") = 0.0,
          "Create a 2D vector field with 3 components");
}
