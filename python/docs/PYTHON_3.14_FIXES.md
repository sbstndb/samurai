# Fixes pour Python 3.14 - À appliquer dans le dépôt

## Fichiers modifiés à commit

### 1. `python/pyproject.toml`

**Problème** : Options scikit-build-core obsolètes

**Changements** :
```toml
# AVANT (incorrect)
[tool.scikit-build]
cmake_minimum-version = "3.16"
cmake-source-dir = "."
build-dir = "build/{wheel_tag}"
wheel.py-api = "py3"
cmake.args = ["-GNinja"]
cmake.build-type = "Release"
jobs = 6

[tool.scikit-build.metadata]
version-provider = "scikit_build_core.metadata.regex"

# APRÈS (correct)
[tool.scikit-build]
minimum-version = "build-system"
cmake.source-dir = "."
build-dir = "build/{wheel_tag}"
wheel.py-api = "py3"
cmake.args = ["-GNinja"]
cmake.build-type = "Release"
jobs = 6

[tool.scikit-build.metadata.version]
provider = "scikit_build_core.metadata.regex"
input = "../version.txt"
regex = "(?P<version>.+)"
```

### 2. `python/CMakeLists.txt`

**Problème** : pybind11 du système utilisé au lieu de conda

**Changements** : Ajouter après le `find_package(pybind11)` :
```cmake
# Force use of conda pybind11 if available
if(DEFINED ENV{CONDA_PREFIX})
    set(pybind11_INCLUDE_DIR "$ENV{CONDA_PREFIX}/include" CACHE PATH "Pybind11 include dir from conda")
    message(STATUS "Using conda pybind11 from: ${pybind11_INCLUDE_DIR}")
endif()
```

### 3. `python/__init__.py` (dans build, pas dans source)

**Problème** : Utilisait ctypes.PyDLL qui ne fonctionne pas avec pybind11

**Changements** : Remplacer tout le chargement ctypes par importlib
```python
# AVANT (ne fonctionne pas)
import ctypes
_so = ctypes.PyDLL(_so_path)
PyInit_samurai_python = getattr(_so, "PyInit_samurai_python")
_compiled_module = PyInit_samurai_python()

# APRÈS (correct)
import importlib.util
spec = importlib.util.spec_from_file_location("samurai_python", _so_path)
_compiled_module = importlib.util.module_from_spec(spec)
sys.modules["samurai_python_compiled"] = _compiled_module
spec.loader.exec_module(_compiled_module)

# Copier __version__ explicitement
if hasattr(_compiled_module, "__version__"):
    __version__ = _compiled_module.__version__
```

**Note** : Ce fichier est généré lors du build, il faut modifier le source dans `python/src/samurai_python/__init__.py.in` ou créer un script de post-build.

### 4. `python/conftest.py` (NOUVEAU FICHIER)

**Problème** : Pas de configuration pytest pour trouver le module compilé

**Créer** : `python/conftest.py`
```python
import sys
import os

# Add build directory to Python path for pytest
build_python_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'build_py314', 'python'))
if os.path.exists(build_python_dir) and build_python_dir not in sys.path:
    sys.path.insert(0, build_python_dir)
```

### 5. `conda/environment-py314.yml` (NOUVEAU FICHIER)

**Problème** : Dépendances manquantes

**Créer** : `conda/environment-py314.yml`
```yaml
name: samurai-py314
channels:
  - conda-forge
dependencies:
  # Python - version fixée
  - python=3.14.2

  # Outils de build
  - cmake=4.2.1
  - ninja=1.13.2
  - cxx-compiler=1.11.0

  # Dépendences principales Samurai
  - xtensor=0.27.1
  - highfive=3.2.0
  - fmt=12.1.0
  - pugixml=1.15
  - cli11=2.6.0

  # Python dependencies
  - numpy
  - pip
  - pybind11  # Dernière version automatiquement

  # Tests
  - pytest=9.0.2
  - pytest-cov
  - h5py=3.15.1

  # Optionnel: PETSc pour solveurs
  - petsc=3.24.3
  - pkg-config
```

### 6. `conda/mpi-environment-py314.yml` (NOUVEAU FICHIER)

**Créer** pour MPI avec Python 3.14
```yaml
name: samurai-mpi-py314
channels:
  - conda-forge
dependencies:
  - python=3.14.2
  - cmake=4.2.1
  - ninja=1.13.2
  - cxx-compiler=1.11.0
  - xtensor=0.27.1
  - highfive=3.2.0
  - fmt=12.1.0
  - pugixml=1.15
  - cli11=2.6.0
  - numpy
  - pip
  - pybind11
  - pytest=9.0.2
  - pytest-cov
  - h5py=3.15.1
  - mpich=4.3.2
  - libboost-mpi=1.89.0
  - libboost-devel
  - libboost-headers
  - hdf5=*mpi_*
  - petsc=3.24.3
  - pkg-config
```

## Commandes pour la prochaine compilation

```bash
# 1. Créer l'environnement
conda env create -f conda/environment-py314.yml
conda activate samurai-py314

# 2. Configurer
cmake -S . -B build_py314 -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON_BINDINGS=ON -GNinja

# 3. Compiler avec 6 jobs
cmake --build build_py314 --target samurai_python --parallel 6

# 4. Tester
cd python
pytest tests/test_basic.py -v
```

## Résumé des problèmes root cause

1. **scikit-build-core** : API changée entre versions, options obsolètes dans pyproject.toml
2. **CMake + conda** : find_package trouve le système avant conda, contournement nécessaire
3. **ctypes vs importlib** : ctypes.PyDLL ne fonctionne pas pour charger pybind11 modules
4. **Tests** : Pas de configuration pour trouver le module compilé
5. **Dependencies** : numpy, pip manquaient dans l'environnement

## Note sur Python 3.14

Le module fonctionne mais le warning GIL est normal. Pour une support complet du no-GIL, pybind11 devra être mis à jour dans une future version (3.1+ probablement).
