# Plan de Migration: Séparation des Bindings Python Samurai

**Date**: 10 janvier 2026
**Objectif**: Rendre le dossier `python/` complètement autonome tout en restant dans le repo `samurai_pybind11`

---

## Choix Architecturaux (Basés sur vos réponses)

| Aspect | Choix | Description |
|--------|-------|-------------|
| **Dépendance C++** | `FindPackage` | Samurai doit être installé au système (via conda/apt/cmake install) |
| **Emplacement** | `/python` dans `samurai_pybind11` | Dossier autonome dans le repo existant |
| **Distribution** | Conda uniquement | Paquets conda-forge pour HPC scientifique |
| **Headers** | Installés uniquement | Utilisation de `find_package(samurai CONFIG REQUIRED)` |

---

## Architecture Cible

```
samurai_pybind11/
├── include/samurai/           # C++ library (installed via cmake)
├── CMakeLists.txt             # Main C++ library build
├── python/                    # ← AUTONOME PYTHON PACKAGE
│   ├── CMakeLists.txt         # ← Standalone CMake (find_package samurai)
│   ├── pyproject.toml         # ← Python package metadata
│   ├── conda-recipe/          # ← Conda build recipe
│   ├── src/
│   │   ├── bindings/          # C++ bindings (13 files)
│   │   └── samurai_python/    # Pure Python utilities
│   ├── tests/
│   ├── examples/
│   └── README.md              # Installation spécifique
└── (reste du projet samurai)
```

---

## Implications des Choix

### ✅ Avantages

1. **Séparation propre des responsabilités**
   - Le dossier `python/` peut être construit indépendamment
   - Le C++ samurai peut être utilisé sans Python
   - Les bindings Python peuvent être versionnés indépendamment

2. **Installation standard conda**
   - `conda install -c conda-forge samurai` (C++ library)
   - `conda install -c conda-forge samurai-python` (bindings)
   - Gestion des dépendances via conda

3. **Développement facilité**
   - Pas de duplication de code (headers partagés)
   - Build système standard (CMake + conda-build)

### ⚠️ Contraintes à Gérer

1. **Ordre d'installation requis**
   ```
   conda install samurai          # D'abord le C++
   conda install samurai-python   # Ensuite les bindings
   ```

2. **Développement local plus complexe**
   - Nécessite `cmake --install` ou `conda develop` pour tester
   - Plus possible de `cd build && make` directement

3. **Compatibilité des versions**
   - Le bindings doivent spécifier la version exacte de samurai requise
   - Risque de mismatch si versions différentes

---

## Plan de Migration par Étapes

### Phase 1: Restructuration du Dossier `python/`

**Objectif**: Rendre `python/` constructible de manière autonome

#### Étape 1.1: Créer un `CMakeLists.txt` autonome pour Python

```cmake
# python/CMakeLists.txt (nouveau)
cmake_minimum_required(VERSION 3.20)
project(samurai_python LANGUAGES CXX)

# Trouver Python et pybind11
find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

# Trouver samurai installé
find_package(samurai CONFIG REQUIRED)

# Créer le module Python
pybind11_add_module(samurai_python
    src/bindings/main.cpp
    src/bindings/box_bindings.cpp
    src/bindings/mesh_config_bindings.cpp
    src/bindings/mesh_bindings.cpp
    src/bindings/field_bindings.cpp
    src/bindings/interval_bindings.cpp
    src/bindings/algorithm_bindings.cpp
    src/bindings/operator_bindings.cpp
    src/bindings/bc_bindings.cpp
    src/bindings/mra_config_bindings.cpp
    src/bindings/adapt_bindings.cpp
    src/bindings/io_bindings.cpp
    src/bindings/domain_builder_bindings.cpp
)

# Lier contre samurai installé
target_link_libraries(samurai_python
    PRIVATE
        samurai::samurai
)

# Configuration d'installation
install(TARGETS samurai_python
    LIBRARY DESTINATION ${Python_SITEARCH}
)

install(DIRECTORY src/samurai_python/
    DESTINATION ${Python_SITEROOT}/samurai_python
    FILES_MATCHING PATTERN "*.py"
)
```

#### Étape 1.2: Créer `pyproject.toml` pour le packaging

```toml
# python/pyproject.toml
[build-system]
requires = ["scikit-build-core>=0.10", "pybind11>=2.13", "cmake>=3.20"]
build-backend = "scikit_build_core.build"

[project]
name = "samurai-python"
version = "0.30.0"
description = "Python bindings for Samurai AMR library"
readme = "README.md"
requires-python = ">=3.9"
dependencies = [
    "numpy>=1.20",
    "h5py>=3.0",
]

[project.urls]
Homepage = "https://github.com/hpc-maths/samurai"
Documentation = "https://samurai.readthedocs.io"

[tool.scikit-build]
cmake.args = ["-DSAMURAI_PYTHON_STANDALONE=ON"]
wheel.py-api = "py3"
```

#### Étape 1.3: Créer la recette conda

```bash
# python/conda-recipe/
mkdir -p python/conda-recipe
```

```yaml
# python/conda-recipe/meta.yaml
package:
  name: samurai-python
  version: "0.30.0"

source:
  path: ../  # Pour le développement local

build:
  number: 0
  script: {{ PYTHON }} -m pip install . --no-build-isolation -vv

requirements:
  build:
    - {{ compiler('cxx') }}
    - cmake >=3.20
    - scikit-build-core
    - pybind11 >=2.13
  host:
    - python >=3.9
    - pip
    - samurai >=0.30.0  # Dépendance C++
  run:
    - python >=3.9
    - samurai >=0.30.0
    - numpy >=1.20
    - h5py >=3.0
    - >=0  # xtensor sera tiré par samurai

test:
  imports:
    - samurai
  commands:
    - python -c "import samurai; print(samurai.__version__)"

about:
  home: https://github.com/hpc-maths/samurai
  license: BSD-3-Clause
  summary: Python bindings for Samurai AMR library
```

---

### Phase 2: Modifications du CMake Principal

**Objectif**: Permettre l'installation de samurai pour consommation externe

#### Étape 2.1: Ajouter l'installation dans `CMakeLists.txt` principal

```cmake
# À la fin du CMakeLists.txt principal
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Installer les headers
install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp"
)

# Créer et installer le config
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/samuraiConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/samuraiConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/samurai
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/samuraiConfigVersion.cmake"
    VERSION ${SAMURAI_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/samuraiConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/samuraiConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/samurai
)

# Installer la cible samurai
install(TARGETS samurai
    EXPORT samuraiTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT samuraiTargets
    FILE samuraiTargets.cmake
    NAMESPACE samurai::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/samurai
)
```

#### Étape 2.2: Créer le fichier de config template

```cmake
# cmake/samuraiConfig.cmake.in (nouveau)
@PACKAGE_INIT@

include(CMakeFindDependencyMacro)

# Dépendances
find_dependency(xtensor CONFIG REQUIRED)
find_dependency(HighFive CONFIG REQUIRED)
find_dependency(Eigen3 CONFIG)
find_dependency(Boost COMPONENTS serialization mpi)

# Inclure les targets exportées
include("${CMAKE_CURRENT_LIST_DIR}/samuraiTargets.cmake")

check_required_components(samurai)
```

#### Étape 2.3: Modifier la condition Python dans le CMake principal

```cmake
# CMakeLists.txt principal (modification)
option(BUILD_PYTHON_BINDINGS "Build Python bindings (DEPRECATED - use python/ dir)" OFF)

if(BUILD_PYTHON_BINDINGS)
    message(WARNING "BUILD_PYTHON_BINDINGS is deprecated. "
                    "Build from python/ directory instead for standalone package.")
    add_subdirectory(python)
endif()
```

---

### Phase 3: Workflow de Développement

#### Option A: Développement avec installation locale

```bash
# 1. Construire et installer samurai
cd /path/to/samurai_pybind11
cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_INSTALL=ON
cmake --build build --target install

# 2. Construire les bindings Python
cd python
cmake -B build -DCMAKE_PREFIX_PATH=/path/to/install
cmake --build build
cp build/samurai_python*.so src/samurai_python/

# 3. Tester
python -c "import samurai; print(samurai.__version__)"
```

#### Option B: Développement avec conda

```bash
# 1. Créer l'environnement avec samurai C++
conda create -n samurai-dev samurai -c conda-forge
conda activate samurai-dev

# 2. Installer les bindings en mode développement
cd python
pip install -e . --no-build-isolation

# ou avec conda develop
conda develop .
```

#### Option C: Build conda complet

```bash
# 1. Builder samurai C++
conda build conda/recipe

# 2. Builder samurai-python
cd python
conda build conda-recipe

# 3. Tester l'installation locale
conda install --use-local samurai-python
```

---

### Phase 4: Tests et CI/CD

#### Étape 4.1: Créer un workflow CI pour Python

```yaml
# .github/workflows/python-bindings.yml
name: Python Bindings

on:
  push:
    paths: ['python/**']

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Install samurai C++
        run: |
          cmake -B build -DCMAKE_BUILD_TYPE=Release
          cmake --build build --target install

      - name: Build Python bindings
        working-directory: python
        run: |
          cmake -B build -DCMAKE_PREFIX_PATH=/usr/local
          cmake --build build

      - name: Test
        working-directory: python
        run: pytest tests/
```

#### Étape 4.2: Tests locaux

```bash
# Dans python/
pytest tests/ -v
```

---

### Phase 5: Documentation

#### Étape 5.1: README pour `python/`

```markdown
# Samurai Python Bindings

Python bindings for the Samurai AMR library.

## Installation

### Via Conda (Recommended)

```bash
conda install -c conda-forge samurai
conda install -c conda-forge samurai-python
```

### From Source

**Prerequisites**: Samurai C++ library must be installed and findable by CMake.

```bash
# Install samurai C++ first
cd /path/to/samurai
cmake -B build -DCMAKE_BUILD_TYPE=Release
sudo cmake --build build --target install

# Then build Python bindings
cd python
pip install .
```

## Development

```bash
# Install in editable mode
pip install -e . --no-build-isolation

# Run tests
pytest tests/
```

## Examples

See `examples/` directory.
```

---

## Checklist de Migration

- [ ] Créer `python/CMakeLists.txt` autonome avec `find_package(samurai)`
- [ ] Créer `python/pyproject.toml` pour scikit-build-core
- [ ] Créer `python/conda-recipe/meta.yaml`
- [ ] Modifier CMake principal pour installer samurai correctement
- [ ] Créer `cmake/samuraiConfig.cmake.in`
- [ ] Créer `python/README.md` avec instructions d'installation
- [ ] Mettre à jour `.github/workflows/` pour tester les bindings séparément
- [ ] Tester le build complet: samurai C++ → install → bindings Python
- [ ] Documenter le workflow de développement
- [ ] Mettre à jour la documentation principale

---

## Risques et Mitigations

| Risque | Mitigation |
|--------|------------|
| **Dépendance circulaire** | Le C++ samurai ne dépend plus de python/ |
| **Version mismatch** | Spécifier version exacte dans conda recipe |
| **Installation complexe** | Fournir paquets conda pré-compilés |
| **Développement plus lent** | Script de build/develop pour simplifier |
| **Tests cassés** | Adapter `conftest.py` pour trouver le module compilé |

---

## Questions Restantes

1. **Gestion des versions communes**: Faut-il synchroniser les versions de samurai et samurai-python ?

2. **CI/CD**: Faut-il des releases séparées ou synchronisées ?

3. **Backward compatibility**: Que faire de l'option `BUILD_PYTHON_BINDINGS` dans le CMake principal ?

4. **conda-forge**: Le paquet sera-t-il maintenu dans conda-forge ou dans un channel séparé ?

---

## Prochaine Étape

Confirmez-vous ce plan ? Je peux alors :
1. Créer les fichiers de configuration (CMakeLists.txt, pyproject.toml, meta.yaml)
2. Modifier le CMake principal pour l'installation
3. Créer les scripts de build/développement
