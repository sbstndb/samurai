# Bindings Python Samurai - Résumé Exécutif

## 📁 Worktree Créé

```bash
/worktree: /home/sbstndbs/sbstndbs/samurai-worktrees/python-bindings
/branch: python-bindings
```

---

## 🎯 Verdict de Faisabilité

### ✅ POSSIBLE (avec restrictions)

**Pourquoi c'est possible** :
1. **Types concrets existent** : `UniformConfig<1>`, `UniformMesh<UniformConfig<1>>`
2. **Header-only** = pas de problèmes de compatibilité binaire
3. **pybind11** gère bien xtensor via `xtensor-python`

**Pourquoi c'est complexe** :
1. Samurai est **très templaté** (`template <class D, class Config> class Mesh_base`)
2. La dimension est un **paramètre template** (pas runtime)
3. xtensor comme backend nécessite une intégration spéciale avec NumPy

---

## 📋 Sous-ensemble Fonctionnel (MVP)

### Ce qu'on VA binder (Phase 1-4)

| Fonctionnalité | C++ | Python | Complexité |
|----------------|-----|--------|------------|
| Boîte géométrique | `Box<double, 1/2>` | `Box1D`, `Box2D` | ⭐ |
| Intervalles | `Interval<int, int>` | `Interval` | ⭐ |
| Maillage uniforme | `UniformMesh<UniformConfig<1>>` | `UniformMesh1D` | ⭐⭐ |
| Itération cellules | `for_each_cell` | `for_each_cell()` | ⭐⭐ |
| Champ scalaire | `ScalarField<Mesh, double>` | `ScalarField1D` | ⭐⭐⭐ |
| Accès NumPy zero-copy | xtensor | `numpy_view()` | ⭐⭐⭐ |

### Ce qu'on NE bindera PAS (initialement)

- ❌ Maillage adaptatif (`MRMesh`)
- ❌ Champs vectoriels
- ❌ Schémas numériques (finite volume, etc.)
- ❌ MPI/Parallélisme
- ❌ Opérateurs différentiels complexes

---

## 🚀 Plan Progressif

```
Phase 0: Infrastructure (Jours 1)
   └─> CMakeLists.txt, build system

Phase 1: Types simples (Jours 2-5)
   └─> Box1D/2D, Interval, Cell1D/2D
   └─> ✅ Démontre que pybind11 fonctionne

Phase 2: Mesh uniforme (Jours 6-12)
   └─> UniformMesh1D/2D
   └─> ✅ Démontre l'instanciation de classes complexes

Phase 3: Itération (Jours 13-16)
   └─> for_each_cell(), lambdas C++ → Python
   └─> ✅ Démontre l'interopérabilité callable

Phase 4: Champs + NumPy (Jours 17-28)
   └─> ScalarField1D/2D avec buffer protocol
   └─> ✅ Démontre la performance (<5% overhead)

Phase 5: Factory functions (Jours 29-33)
   └─> make_scalar_field(), API simplifiée
   └─> ✅ Démontre l'ergonomie Python

Phase 6: I/O HDF5 (Jours 34-40)
   └─> save(), load()
   └─> ✅ Intégration écosystème complet
```

---

## 📝 Exemple d'Usage Cible (MVP)

```python
import samurai_core
import numpy as np

# 1. Créer un maillage
box = samurai_core.Box1D([0.], [1.])
mesh = samurai_core.UniformMesh1D(box, level=5)

# 2. Créer un champ
u = samurai_core.ScalarField1D("solution", mesh)

# 3. Le remplir avec une fonction Python
for cell in mesh.for_each_cell():
    x = cell.center[0]
    u[cell] = np.sin(2 * np.pi * x)

# 4. Accès zero-copy aux données
u_arr = u.numpy_view()
u_arr *= 2

# 5. Sauvegarder
samurai_core.save(".", "solution", u)
```

---

## 🔧 Stratégie Technique

### 1. Instanciation Concrète (pas de templates génériques)

```cpp
// Au lieu de binder des templates...
template <std::size_t dim> class Mesh...

// On binder des types concrets
using UniformMesh1D = UniformMesh<UniformConfig<1>>;
using UniformMesh2D = UniformMesh<UniformConfig<2>>;

py::class_<UniformMesh1D>(m, "UniformMesh1D") { ... };
py::class_<UniformMesh2D>(m, "UniformMesh2D") { ... };
```

### 2. Fixer les Template Parameters

```cpp
namespace samurai::python {
    // Fixés pour Python
    using default_interval = Interval<int, long long int>;

    template <std::size_t dim>
    using Config = UniformConfig<dim, 1, default_interval>;

    template <std::size_t dim>
    using UniformMesh = samurai::UniformMesh<Config<dim>>;
}
```

### 3. Zero-Copy NumPy Integration

```cpp
.def("numpy_view", [](Field& f) {
    return py::array_t<double>(
        {f.array().size()},       // shape
        {sizeof(double)},          // strides
        f.array().data(),          // data pointer
        py::cast(f)                // keep alive!
    );
})
```

---

## 📊 Timeline Estimée

| Phase | Durée | Livrable |
|-------|-------|----------|
| 0 | 1 jour | Build fonctionnel |
| 1 | 4 jours | Tests Box/Interval passent |
| 2 | 7 jours | Mesh fonctionnel |
| 3 | 4 jours | Itération cellules |
| 4 | 12 jours | Zero-copy NumPy |
| 5 | 5 jours | Factory functions |
| 6 | 7 jours | I/O HDF5 |
| **Total** | **40 jours** (~8 semaines) | **MVP complète** |

---

## ⚠️ Risques Identifiés

| Risque | Probabilité | Mitigation |
|--------|-------------|------------|
| xTensor incompatible | Moyenne | Utiliser xtensor-python |
| Performance dégradée | Faible | Profilage, zero-copy |
| Compilation lente | Élevée | Modules séparés |

---

## 🎁 Livrables

### Documentation
- `FEASIBILITY_ANALYSIS.md` - Analyse détaillée faisabilité
- `IMPLEMENTATION_PLAN.md` - Plan d'implémentation pas-à-pas
- `README_PYTHON_BINDINGS.md` - Ce fichier

### Code (à implémenter)
- `src/python_bindings/` - Bindings C++
- `python/tests/` - Tests pytest
- `python/examples/` - Exemples d'usage

---

## 🚦 Prochaine Étape

Commencer **Phase 0** : Création de l'infrastructure de build

```bash
cd /home/sbstndbs/sbstndbs/samurai-worktrees/python-bindings
mkdir -p src/python_bindings python/tests python/examples build-python
```

Voulez-vous que je commence l'implémentation de la Phase 0 ?
