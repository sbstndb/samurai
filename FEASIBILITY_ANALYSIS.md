# Analyse de Faisabilité : Bindings Python pour Samurai V2

**Date** : 2025-01-05
**Worktree** : `/home/sbstndbs/sbstndbs/samurai-worktrees/python-bindings`

---

## 1. Diagnostic : Pourquoi c'est COMPLEXE mais POSSIBLE

### 1.1 Les Défis

| Défi | Description | Impact |
|------|-------------|--------|
| **Templates lourds** | `template <class D, class Config> class Mesh_base` | Nécessite des instanciations concrètes |
| **Dimension compile-time** | `std::size_t dim` comme template param | 3 versions séparées (1D, 2D, 3D) |
| **Configuration complexe** | `interval_t`, `mesh_id_t`, `max_refinement_level` | Doit être fixée à l'avance |
| **xTensor backend** | Stockage via `xt::xtensor` | Intégration NumPy non-triviale |

### 1.2 Les Opportunités

| Opportunité | Description | Bénéfice |
|-------------|-------------|----------|
| **Header-only** | Pas de problèmes ABI | Instable? Non, recompilé à chaque fois |
| **Factory functions** | `make_scalar_field<T>(...)` | Cache la complexité des templates |
| **Types concrets existants** | `UniformConfig<1>`, `Interval<int, int>` | Peuvent être bindés directement |
| **CRTP pattern** | `Mesh_base<D, Config>` | Classes finales instanciables |

---

## 2. Analyse des Types Cibles

### 2.1 Classes Simples (High ROI, Low Effort)

```cpp
// Ces classes ont une interface simple, peu de dépendances
Box<double, 1>       // Géométrie 1D
Box<double, 2>       // Géométrie 2D
Interval<int, int>   // Intervalle d'indices
Cell<1, interval_t>  // Cellule 1D
Cell<2, interval_t>  // Cellule 2D
```

**Complexité pybind11** : ⭐ (très simple)

### 2.2 Classes Moyennes (Medium ROI, Medium Effort)

```cpp
// Configurations avec templates mais parameters fixes
UniformConfig<1>
UniformConfig<2>
UniformConfig<3>

// Maillages uniformes (pas d'adaptation)
UniformMesh<UniformConfig<1>>
UniformMesh<UniformConfig<2>>
UniformMesh<UniformConfig<3>>
```

**Complexité pybind11** : ⭐⭐ (nécessite des typedefs)

### 2.3 Classes Complexes (High ROI, High Effort)

```cpp
// Champs avec buffer protocol pour NumPy
ScalarField<UniformMesh<UniformConfig<2>>, double>
VectorField<UniformMesh<UniformConfig<2>>, double, 2>

// Mesh adaptatif (MRMesh) - TARD
MRMesh<MRConfig<2>>
```

**Complexité pybind11** : ⭐⭐⭐ (nécessite understanding profond)

---

## 3. Stratégie d'Instanciation Concrète

Au lieu d'exposer les templates, on expose des **types concrets** :

```cpp
// En C++ (header-only)
using samurai_interval_t = samurai::Interval<int, long long int>;

// Pour chaque dimension
using UniformMesh1D = samurai::UniformMesh<samurai::UniformConfig<1>>;
using UniformMesh2D = samurai::UniformMesh<samurai::UniformConfig<2>>;
using UniformMesh3D = samurai::UniformMesh<samurai::UniformConfig<3>>;

// Fields
using ScalarField1D = samurai::ScalarField<UniformMesh1D, double>;
using ScalarField2D = samurai::ScalarField<UniformMesh2D, double>;
using ScalarField3D = samurai::ScalarField<UniformMesh3D, double>;
```

---

## 4. Plan Progressif de Démonstration

### Phase 0 : Proof of Concept (1 semaine)

**Objectif** : Démontrer qu'on peut exposer UN type simple

```cpp
// src/python_bindings/box_bindings.cpp

#include <pybind11/pybind11.h>
#include <samurai/box.hpp>

PYBIND11_MODULE(samurai_core, m) {
    py::class_<samurai::Box<double, 1>>(m, "Box1D")
        .def(py::init<const xt::xtensor_fixed<double, xt::xshape<1>>&,
                      const xt::xtensor_fixed<double, xt::xshape<1>>&>())
        .def_property_readonly("min", &samurai::Box<double, 1>::min_corner)
        .def_property_readonly("max", &samurai::Box<double, 1>::max_corner)
        .def_property_readonly("length", &samurai::Box<double, 1>::length)
        .def("__repr__", [](const samurai::Box<double, 1>& b) {
            return "Box1D(min=" + std::to_string(b.min_corner()[0]) +
                   ", max=" + std::to_string(b.max_corner()[0]) + ")";
        });
}
```

**Critère de succès** : `python -c "import samurai_core; b = samurai_core.Box1D([0.], [1.]); print(b)"`

---

### Phase 1 : Types Fondamentaux (1-2 semaines)

**Objectif** : Exposer les briques de base

| Classe C++ | Classe Python | Priorité |
|------------|---------------|----------|
| `Box<double, 1>` | `Box1D` | P0 |
| `Box<double, 2>` | `Box2D` | P0 |
| `Interval<int, int>` | `Interval` | P1 |
| `Cell<1, interval_t>` | `Cell1D` | P1 |
| `Cell<2, interval_t>` | `Cell2D` | P1 |

**Dépendances** : Aucune (types fondamentaux)

**Tests** : pytest démontrant création et accès aux propriétés

---

### Phase 2 : Mesh Uniforme (2-3 semaines)

**Objectif** : Créer et manipuler un maillage

```python
# Exemple cible
config = samurai.UniformConfig1D(level=5)
box = samurai.Box1D([0.], [1.])
mesh = samurai.UniformMesh1D(box, 5)

print(f"Cells: {mesh.nb_cells()}")  # 32
```

**C++ à binder** :

```cpp
using Config1D = samurai::UniformConfig<1>;
using UniformMesh1D = samurai::UniformMesh<Config1D>;

py::class_<UniformMesh1D>(m, "UniformMesh1D")
    .def(py::init<const samurai::Box<double, 1>&, std::size_t>())
    .def("nb_cells", &UniformMesh1D::nb_cells)
    .def_property_readonly("origin_point", &UniformMesh1D::origin_point)
    .def_property_readonly("scaling_factor", &UniformMesh1D::scaling_factor);
```

**Dépendances** : Phase 1 (Box, Interval)

---

### Phase 3 : Iteration et Cellules (2 semaines)

**Objectif** : Itérer sur les cellules du maillage

```python
for cell in mesh.for_each_cell():
    print(f"Center: {cell.center}, Level: {cell.level}")
```

**Challenge** : Wrapper les lambdas C++ → Python

```cpp
// C++ side
m.def("for_each_cell", [](UniformMesh1D& mesh, py::function func) {
    samurai::for_each_cell(mesh, [&](auto& cell) {
        func(cell);  // Convert cell to Python
    });
});
```

**Dépendances** : Phase 2

---

### Phase 4 : Champs avec NumPy Zero-Copy (3-4 semaines)

**Objectif** : Accès direct aux données via NumPy

```python
u = samurai.ScalarField1D("u", mesh)
u.fill(1.0)

# ZERO-COPY
u_arr = u.numpy_view()
print(u_arr.shape)  # (32,)
u_arr[:] = np.sin(x_coords)
print(u[0])  # Modifié via NumPy!
```

**Défi clé** : Buffer protocol

```cpp
py::class_<ScalarField1D>(m, "ScalarField1D", py::buffer_protocol())
    .def_buffer([](ScalarField1D& f) -> py::buffer_info {
        return py::buffer_info(
            f.array().data(),                          // Pointer
            sizeof(double),                            // Size of scalar
            py::format_descriptor<double>::format(),    // Format
            1,                                          // Dimensions
            {f.array().size()},                        // Shape
            {sizeof(double)}                           // Strides
        );
    })
    .def("numpy_view", [](ScalarField1D& f) {
        return py::array_t<double>(
            {f.array().size()},
            {sizeof(double)},
            f.array().data(),
            py::cast(f)  // Keep alive!
        );
    }, py::return_value_policy::take_ownership);
```

**Dépendances** : Phase 2

---

### Phase 5 : Factory Functions (1 semaine)

**Objectif** : Simplifier la création d'objets

```python
# Au lieu de
# config = samurai.MeshConfig2D()
# config.min_level = 2
# config.max_level = 4

# On peut faire
u = samurai.make_scalar_field_2d("u", mesh, lambda x: np.sin(x[0]))
```

**Dépendances** : Phase 4

---

### Phase 6 : Opérateurs et Algorithmes (3 semaines)

**Objectif** : Exposer des algorithmes utiles

```python
# Adaptation de maillage (si MRMesh)
samurai.adapt(mesh, 0.01, criterion)

# Conditions aux limites
u.attach_bc(lambda dir, cell: 0., samurai.Direction.Left)
```

**Dépendances** : Phase 5

---

## 5. Sous-ensemble de Fonctionnalités pour Démonstration

### Scope Minimal (MVP)

```python
# Ce qu'on veut pouvoir faire en Python dès le début :
import samurai
import numpy as np

# 1. Créer un maillage
box = samurai.Box1D([0.], [1.])
mesh = samurai.UniformMesh1D(box, level=5)

# 2. Créer un champ
u = samurai.ScalarField1D("solution", mesh)

# 3. Le remplir avec une fonction Python
for cell in mesh.for_each_cell():
    x = cell.center[0]
    u[cell] = np.sin(2 * np.pi * x)

# 4. Accéder zero-copy aux données
u_arr = u.numpy_view()
u_arr *= 2

# 5. Sauvegarder
samurai.save("solution", u)
```

### Hors Scope Initial

- ❌ MRMesh (maillage adaptatif)
- ❌ VectorField
- ❌ Opérateurs différentiels
- ❌ Schémas numériques (finite volume, etc.)
- ❌ MPI/Parallélisme

---

## 6. Risques et Mitigations

| Risque | Probabilité | Impact | Mitigation |
|--------|-------------|--------|------------|
| xTensor incompatible avec pybind11 | Moyenne | Élevé | Utiliser xtensor-python ou conversion manuelle |
| Performance dégradée | Faible | Moyen | Profilage systématique, zero-copy |
- Compilation trop lente | Élevée | Faible | Modules séparés, precompiled headers |
| Template explosion | Moyenne | Moyen | Limiter les instanciations (1D,2D,3D) |

---

## 7. Ressources Nécessaires

### Dépendances

```cmake
# CMakeLists.txt additionnel
find_package(pybind11 2.10 REQUIRED)
find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development)

# Optionnel: xtensor-python
find_package(xtensor-python)  # Pour meilleure intégration
```

### Structure de Répertoire Proposée

```
src/python_bindings/
├── CMakeLists.txt
├── samurai_module.cpp          # Module principal
├── box_bindings.hpp/cpp        # Phase 1
├── interval_bindings.hpp/cpp   # Phase 1
├── cell_bindings.hpp/cpp       # Phase 1
├── mesh_bindings.hpp/cpp       # Phase 2
├── field_bindings.hpp/cpp      # Phase 4
└── algorithm_bindings.hpp/cpp  # Phase 6

python/
├── tests/
│   ├── test_box.py
│   ├── test_mesh.py
│   └── test_field.py
└── examples/
    └── demo_1d_heat.py
```

---

## 8. Critères de Succès par Phase

### Phase 0-1 (PoC + Types de base)
- [ ] Box1D/Box2D créables depuis Python
- [ ] Accès aux propriétés (min, max, length)
- [ ] Tests pytest passent

### Phase 2 (Mesh)
- [ ] UniformMesh1D/2D fonctionnels
- [ ] nb_cells() retourne la bonne valeur
- [ ] Affichage du maillage

### Phase 3 (Itération)
- [ ] for_each_cell fonctionne
- [ ] Cell wrappers exposent center, level, indices

### Phase 4 (NumPy)
- [ ] numpy_view() retourne un array
- [ ] Modification NumPy = modification Field
- [ ] np.shares_memory() == True

### Phase 5-6 (Complet)
- [ ] make_scalar_field simplifie l'API
- [ ] Sauvegarde HDF5 fonctionnelle
- [ ] Performance < 5% overhead vs C++

---

## 9. Conclusion

### Faisabilité : **CONFIRMÉE** avec conditions

✅ **Possible** car :
- Types concrets existent (`UniformConfig<dim>`)
- Header-only = pas d'ABI issues
- pybind11 gère bien les xtensor

⚠️ **Sous conditions** :
- Fixer les template parameters (dim, value_t, interval_t)
- Commencer par UniformMesh (pas MRMesh au début)
- Accepter plusieurs versions (1D, 2D, 3D)

### Recommandation

Commencer par **Phase 0-1** (PoC) pour valider la faisabilité technique, puis évoluer vers **Phase 2-4** pour une MVP fonctionnelle.

**Timeline estimée MVP** : 8-10 semaines pour une version utilisable avec UniformMesh + ScalarField + NumPy integration.
