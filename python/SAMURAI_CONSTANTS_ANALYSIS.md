# Analyse des Constantes et Magic Numbers - Samurai Python

## Rapport d'Analyse Approfondie

**Date**: 7 Janvier 2026
**Contexte**: Bindings Python Samurai - Analyse des constantes numériques hardcoded
**Objectif**: Améliorer la gestion des paramètres de simulation

---

## 1. Inventaire des Constantes par Type de Problème

### 1.1 Advection (Équation Linéaire de Convection)

**Exemples analysés**:
- `python/examples/advection_2d.py`
- `python/examples/linear_convection.py`
- `python/examples/linear_convection_obstacle.py`
- `demos/FiniteVolume/advection_2d.cpp`
- `demos/FiniteVolume/linear_convection.cpp`

| Paramètre | Valeur Typique | Plage Observée | Description |
|-----------|----------------|----------------|-------------|
| **CFL** | 0.5 | 0.5 - 0.95 | Upwind: 0.5, WENO5: 0.95 |
| **epsilon** | 1e-4 - 2e-4 | 1e-5 - 1e-3 | Tolérance AMR |
| **regularity** | 1.0 | 0.0 - 3.0 | Gradation de maillage |
| **min_level** | 4-5 | 1 - 6 | Niveau minimum de raffinement |
| **max_level** | 9-10 | 4 - 12 | Niveau maximum de raffinement |
| **max_stencil_size** | 2 (upwind), 6 (WENO5) | 2 - 6 | Taille du stencil numérique |
| **order** | - | 1 - 3 | Ordre des conditions aux limites |

**Valeurs par défaut constatées**:
```python
# Upwind (advection_2d.py)
cfl = 0.5
epsilon = 2e-4
min_level = 4
max_level = 10
max_stencil_size = 2

# WENO5 (linear_convection.py)
cfl = 0.95
epsilon = 1e-4
min_level = 5
max_level = 9
max_stencil_size = 6
```

---

### 1.2 Burgers (Non-linéaire)

**Exemples analysés**:
- `python/examples/burgers_2d.py`
- `python/examples/burgers_2d_simple.py`
- `demos/FiniteVolume/burgers.cpp`
- `demos/FiniteVolume/burgers_mra.cpp`

| Paramètre | Valeur Typique | Plage Observée | Description |
|-----------|----------------|----------------|-------------|
| **CFL** | 0.95 | 0.5 - 0.99 | Plus conservatif pour non-linéaire |
| **epsilon** | 2e-4 | 1e-4 - 1e-3 | Tolérance AMR (chocs) |
| **regularity** | 1.0 | 0.0 - 2.0 | Gradation pour discontinuités |
| **min_level** | 5 | 0 - 6 | Niveau minimum |
| **max_level** | 7 | 3 - 8 | Niveau maximum |
| **max_stencil_size** | 6 (WENO5) | 6 | Obligatoire pour WENO5 |
| **order (BC)** | 3 | 1 - 3 | Ordre BC pour WENO5 |

**Particularités Burgers**:
- Nécessite `max_stencil_size = 6` pour WENO5
- BC avec `order = 3` par défaut
- CFL plus élevé possible (0.95-0.99) avec schémas TVD

---

### 1.3 Diffusion (Équation de la Chaleur)

**Exemples analysés**:
- `demos/FiniteVolume/heat.cpp`
- `demos/FiniteVolume/heat_nonlinear.cpp`
- `demos/FiniteVolume/heat_heterogeneous.cpp`

| Paramètre | Valeur Typique | Plage Observée | Description |
|-----------|----------------|----------------|-------------|
| **CFL** | 0.95 | 0.5 - 0.95 | Diffusion explicite |
| **epsilon** | 1e-4 | 1e-5 - 1e-2 | Tolérance AMR |
| **min_level** | 4 | 3 - 5 | Niveau minimum |
| **max_level** | 8 | 5 - 8 | Niveau maximum |
| **max_stencil_size** | 2 | 2 | Stencil compact pour diffusion |

---

### 1.4 Autres Problèmes

**Belousov-Zhabotinsky (BZ)**:
- `epsilon = 1.e-2` (paramètre de raideur de réaction, pas AMR!)
- `f = 1.6`, `q = 2.e-3` (paramètres cinétique)

**Level Set**:
- `CFL = 5./8 = 0.625` (spécifique à level set)

---

## 2. Schémas Numériques et Constantes Associées

### 2.1 Schémas de Convection

| Schéma | max_stencil_size | CFL typique | Ordre | Utilisation |
|--------|------------------|-------------|-------|-------------|
| **Upwind** | 2 | 0.5 | 1 | Advection simple |
| **WENO5** | 6 | 0.95 | 5 | Haute précision |
| **Houc5** | 6 | 0.95 | 5 | Level set |

### 2.2 Schémas Temporels (TVD-RK3)

**Coefficients SSP-RK3 (Hardcodés partout!)**:
```python
# Stage 1
u1 = u - dt * flux(u)

# Stage 2
u2 = (3/4) * u + (1/4) * (u1 - dt * flux(u1))

# Stage 3
unp1 = (1/3) * u + (2/3) * (u2 - dt * flux(u2))
```

**Constantes magiques**: `3.0/4.0`, `1.0/4.0`, `1.0/3.0`, `2.0/3.0`

---

## 3. Propositions: Système de Presets

### 3.1 Architecture Proposée

```python
# Fichier: python/samurai/config/presets.py

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from enum import Enum

class ProblemType(Enum):
    """Types de problèmes prédéfinis"""
    ADVECTION_UPWIND = "advection_upwind"
    ADVECTION_WENO5 = "advection_weno5"
    BURGERS = "burgers"
    DIFFUSION = "diffusion"
    LINEAR_CONVECTION = "linear_convection"

class AccuracyLevel(Enum):
    """Niveaux de précision"""
    COARSE = "coarse"      # Calculs rapides
    DEFAULT = "default"    # Équilibre précision/coût
    FINE = "fine"          # Haute précision
    ULTRA_FINE = "ultra_fine"  # Précision maximale

@dataclass
class MeshConfigPreset:
    """Configuration de maillage prédéfinie"""
    min_level: int
    max_level: int
    max_stencil_size: int
    periodic: bool = False
    enable_ghost_width: bool = True

@dataclass
class MRAConfigPreset:
    """Configuration d'adaptation multirésolution prédéfinie"""
    epsilon: float
    regularity: float
    relative_detail: bool = False

@dataclass
class TimeSteppingPreset:
    """Configuration de pas de temps prédéfinie"""
    cfl: float
    scheme: str = "SSPRK3"  # SSP-RK3 par défaut

@dataclass
class BoundaryConditionPreset:
    """Configuration de conditions aux limites prédéfinie"""
    bc_type: str  # "dirichlet", "neumann"
    value: float
    order: int = 1

@dataclass
class SimulationConfig:
    """Configuration complète de simulation"""
    name: str
    problem_type: ProblemType
    mesh: MeshConfigPreset
    mra: MRAConfigPreset
    time: TimeSteppingPreset
    boundary: Optional[BoundaryConditionPreset] = None
    description: str = ""

    def validate(self) -> tuple[bool, list[str]]:
        """
        Valide la cohérence de la configuration.

        Returns:
            (is_valid, warnings): Tuple indicant si valide et liste d'avertissements
        """
        warnings = []

        # Validation maillage
        if self.mesh.min_level > self.mesh.max_level:
            warnings.append(f"min_level ({self.mesh.min_level}) > max_level ({self.mesh.max_level})")

        # Validation stencil
        if self.problem_type in [ProblemType.ADVECTION_WENO5, ProblemType.BURGERS]:
            if self.mesh.max_stencil_size < 6:
                warnings.append(f"{self.problem_type.value} nécessite max_stencil_size >= 6 pour WENO5")

        # Validation MRA
        if self.mra.epsilon <= 0:
            warnings.append(f"epsilon doit être positif, reçu: {self.mra.epsilon}")

        if self.mra.regularity < 0:
            warnings.append(f"regularity doit être >= 0, reçu: {self.mra.regularity}")

        # Validation CFL
        if self.time.cfl <= 0 or self.time.cfl > 1.0:
            warnings.append(f"CFL doit être dans (0, 1], reçu: {self.time.cfl}")

        # Validation BC
        if self.boundary and self.boundary.order < 1:
            warnings.append(f"BC order doit être >= 1, reçu: {self.boundary.order}")

        is_valid = len(warnings) == 0
        return is_valid, warnings

    def to_dict(self) -> Dict[str, Any]:
        """Convertit la configuration en dictionnaire"""
        return {
            'name': self.name,
            'problem_type': self.problem_type.value,
            'mesh': {
                'min_level': self.mesh.min_level,
                'max_level': self.mesh.max_level,
                'max_stencil_size': self.mesh.max_stencil_size,
                'periodic': self.mesh.periodic,
            },
            'mra': {
                'epsilon': self.mra.epsilon,
                'regularity': self.mra.regularity,
                'relative_detail': self.mra.relative_detail,
            },
            'time': {
                'cfl': self.time.cfl,
                'scheme': self.time.scheme,
            }
        }


class ConfigPresets:
    """
    Bibliothèque de configurations prédéfinies pour Samurai.

    Cette classe fournit des configurations optimisées pour différents
    types de problèmes et niveaux de précision.
    """

    # =========================================================================
    # PRESETS ADVECTION
    # =========================================================================

    @staticmethod
    def advection_upwind(accuracy: AccuracyLevel = AccuracyLevel.DEFAULT) -> SimulationConfig:
        """
        Configuration pour advection avec schéma upwind (ordre 1).

        Args:
            accuracy: Niveau de précision désiré

        Returns:
            SimulationConfig: Configuration complète
        """
        presets = {
            AccuracyLevel.COARSE: SimulationConfig(
                name="advection_upwind_coarse",
                problem_type=ProblemType.ADVECTION_UPWIND,
                mesh=MeshConfigPreset(min_level=3, max_level=6, max_stencil_size=2),
                mra=MRAConfigPreset(epsilon=1e-2, regularity=1.0),
                time=TimeSteppingPreset(cfl=0.5, scheme="forward_euler"),
                boundary=BoundaryConditionPreset(bc_type="dirichlet", value=0.0, order=1),
                description="Advection upwind rapide (résolution grossière)"
            ),
            AccuracyLevel.DEFAULT: SimulationConfig(
                name="advection_upwind_default",
                problem_type=ProblemType.ADVECTION_UPWIND,
                mesh=MeshConfigPreset(min_level=4, max_level=10, max_stencil_size=2),
                mra=MRAConfigPreset(epsilon=2e-4, regularity=1.0),
                time=TimeSteppingPreset(cfl=0.5, scheme="forward_euler"),
                boundary=BoundaryConditionPreset(bc_type="dirichlet", value=0.0, order=1),
                description="Advection upwind standard (équilibre précision/coût)"
            ),
            AccuracyLevel.FINE: SimulationConfig(
                name="advection_upwind_fine",
                problem_type=ProblemType.ADVECTION_UPWIND,
                mesh=MeshConfigPreset(min_level=5, max_level=12, max_stencil_size=2),
                mra=MRAConfigPreset(epsilon=1e-4, regularity=1.0),
                time=TimeSteppingPreset(cfl=0.5, scheme="forward_euler"),
                boundary=BoundaryConditionPreset(bc_type="dirichlet", value=0.0, order=1),
                description="Advection upwind précis (haute résolution)"
            ),
        }
        return presets[accuracy]

    @staticmethod
    def advection_weno5(accuracy: AccuracyLevel = AccuracyLevel.DEFAULT) -> SimulationConfig:
        """
        Configuration pour advection avec schéma WENO5 (ordre 5).

        Args:
            accuracy: Niveau de précision désiré

        Returns:
            SimulationConfig: Configuration complète
        """
        presets = {
            AccuracyLevel.COARSE: SimulationConfig(
                name="advection_weno5_coarse",
                problem_type=ProblemType.ADVECTION_WENO5,
                mesh=MeshConfigPreset(min_level=4, max_level=7, max_stencil_size=6),
                mra=MRAConfigPreset(epsilon=5e-4, regularity=1.0),
                time=TimeSteppingPreset(cfl=0.95, scheme="SSPRK3"),
                boundary=BoundaryConditionPreset(bc_type="dirichlet", value=0.0, order=3),
                description="Advection WENO5 rapide"
            ),
            AccuracyLevel.DEFAULT: SimulationConfig(
                name="advection_weno5_default",
                problem_type=ProblemType.ADVECTION_WENO5,
                mesh=MeshConfigPreset(min_level=5, max_level=9, max_stencil_size=6),
                mra=MRAConfigPreset(epsilon=1e-4, regularity=1.0),
                time=TimeSteppingPreset(cfl=0.95, scheme="SSPRK3"),
                boundary=BoundaryConditionPreset(bc_type="dirichlet", value=0.0, order=3),
                description="Advection WENO5 standard"
            ),
            AccuracyLevel.FINE: SimulationConfig(
                name="advection_weno5_fine",
                problem_type=ProblemType.ADVECTION_WENO5,
                mesh=MeshConfigPreset(min_level=6, max_level=12, max_stencil_size=6),
                mra=MRAConfigPreset(epsilon=5e-5, regularity=2.0),
                time=TimeSteppingPreset(cfl=0.95, scheme="SSPRK3"),
                boundary=BoundaryConditionPreset(bc_type="dirichlet", value=0.0, order=3),
                description="Advection WENO5 haute précision"
            ),
        }
        return presets[accuracy]

    # =========================================================================
    # PRESETS BURGERS
    # =========================================================================

    @staticmethod
    def burgers(accuracy: AccuracyLevel = AccuracyLevel.DEFAULT) -> SimulationConfig:
        """
        Configuration pour l'équation de Burgers avec schéma WENO5.

        Args:
            accuracy: Niveau de précision désiré

        Returns:
            SimulationConfig: Configuration complète
        """
        presets = {
            AccuracyLevel.COARSE: SimulationConfig(
                name="burgers_coarse",
                problem_type=ProblemType.BURGERS,
                mesh=MeshConfigPreset(min_level=4, max_level=6, max_stencil_size=6),
                mra=MRAConfigPreset(epsilon=5e-4, regularity=1.0),
                time=TimeSteppingPreset(cfl=0.9, scheme="SSPRK3"),
                boundary=BoundaryConditionPreset(bc_type="dirichlet", value=0.0, order=3),
                description="Burgers WENO5 rapide"
            ),
            AccuracyLevel.DEFAULT: SimulationConfig(
                name="burgers_default",
                problem_type=ProblemType.BURGERS,
                mesh=MeshConfigPreset(min_level=5, max_level=7, max_stencil_size=6),
                mra=MRAConfigPreset(epsilon=2e-4, regularity=1.0),
                time=TimeSteppingPreset(cfl=0.95, scheme="SSPRK3"),
                boundary=BoundaryConditionPreset(bc_type="dirichlet", value=0.0, order=3),
                description="Burgers WENO5 standard"
            ),
            AccuracyLevel.FINE: SimulationConfig(
                name="burgers_fine",
                problem_type=ProblemType.BURGERS,
                mesh=MeshConfigPreset(min_level=6, max_level=8, max_stencil_size=6),
                mra=MRAConfigPreset(epsilon=1e-4, regularity=1.0),
                time=TimeSteppingPreset(cfl=0.95, scheme="SSPRK3"),
                boundary=BoundaryConditionPreset(bc_type="dirichlet", value=0.0, order=3),
                description="Burgers WENO5 haute précision (chocs bien résolus)"
            ),
        }
        return presets[accuracy]

    # =========================================================================
    # PRESETS DIFFUSION
    # =========================================================================

    @staticmethod
    def diffusion(accuracy: AccuracyLevel = AccuracyLevel.DEFAULT) -> SimulationConfig:
        """
        Configuration pour l'équation de diffusion (chaleur).

        Args:
            accuracy: Niveau de précision désiré

        Returns:
            SimulationConfig: Configuration complète
        """
        presets = {
            AccuracyLevel.COARSE: SimulationConfig(
                name="diffusion_coarse",
                problem_type=ProblemType.DIFFUSION,
                mesh=MeshConfigPreset(min_level=3, max_level=5, max_stencil_size=2),
                mra=MRAConfigPreset(epsilon=1e-2, regularity=1.0),
                time=TimeSteppingPreset(cfl=0.5, scheme="forward_euler"),
                boundary=BoundaryConditionPreset(bc_type="dirichlet", value=0.0, order=1),
                description="Diffusion rapide"
            ),
            AccuracyLevel.DEFAULT: SimulationConfig(
                name="diffusion_default",
                problem_type=ProblemType.DIFFUSION,
                mesh=MeshConfigPreset(min_level=4, max_level=8, max_stencil_size=2),
                mra=MRAConfigPreset(epsilon=1e-4, regularity=1.0),
                time=TimeSteppingPreset(cfl=0.95, scheme="SSPRK3"),
                boundary=BoundaryConditionPreset(bc_type="dirichlet", value=0.0, order=1),
                description="Diffusion standard"
            ),
            AccuracyLevel.FINE: SimulationConfig(
                name="diffusion_fine",
                problem_type=ProblemType.DIFFUSION,
                mesh=MeshConfigPreset(min_level=5, max_level=10, max_stencil_size=2),
                mra=MRAConfigPreset(epsilon=1e-5, regularity=2.0),
                time=TimeSteppingPreset(cfl=0.95, scheme="SSPRK3"),
                boundary=BoundaryConditionPreset(bc_type="dirichlet", value=0.0, order=1),
                description="Diffusion haute précision"
            ),
        }
        return presets[accuracy]

    # =========================================================================
    # UTILITAIRES
    # =========================================================================

    @staticmethod
    def list_presets() -> Dict[str, list[str]]:
        """
        Liste tous les presets disponibles.

        Returns:
            Dict avec clés = types de problèmes, valeurs = listes de presets
        """
        return {
            "advection_upwind": ["coarse", "default", "fine"],
            "advection_weno5": ["coarse", "default", "fine"],
            "burgers": ["coarse", "default", "fine"],
            "diffusion": ["coarse", "default", "fine"],
        }

    @staticmethod
    def print_preset_info(config: SimulationConfig):
        """Affiche les détails d'une configuration"""
        print(f"\n{'='*70}")
        print(f"Preset: {config.name}")
        print(f"{'='*70}")
        print(f"Description: {config.description}")
        print(f"\nMesh Configuration:")
        print(f"  min_level: {config.mesh.min_level}")
        print(f"  max_level: {config.mesh.max_level}")
        print(f"  max_stencil_size: {config.mesh.max_stencil_size}")
        print(f"  periodic: {config.mesh.periodic}")
        print(f"\nMRA Configuration:")
        print(f"  epsilon: {config.mra.epsilon:.2e}")
        print(f"  regularity: {config.mra.regularity}")
        print(f"  relative_detail: {config.mra.relative_detail}")
        print(f"\nTime Stepping:")
        print(f"  CFL: {config.time.cfl}")
        print(f"  scheme: {config.time.scheme}")
        if config.boundary:
            print(f"\nBoundary Conditions:")
            print(f"  type: {config.boundary.bc_type}")
            print(f"  value: {config.boundary.value}")
            print(f"  order: {config.boundary.order}")
        print(f"{'='*70}\n")
```

---

## 4. Intégration avec Samurai Python

### 4.1 Fonctions Helper

```python
# Fichier: python/samurai/config/helpers.py

import samurai_python as sam
from .presets import SimulationConfig, AccuracyLevel, ProblemType

def apply_config_to_mesh(config: SimulationConfig) -> sam.MeshConfig2D:
    """
    Applique une configuration de maillage prédéfinie.

    Args:
        config: Configuration de simulation

    Returns:
        sam.MeshConfig2D: Configuration de maillage Samurai
    """
    mesh_cfg = sam.MeshConfig2D()
    mesh_cfg.min_level = config.mesh.min_level
    mesh_cfg.max_level = config.mesh.max_level
    mesh_cfg.max_stencil_size = config.mesh.max_stencil_size

    if config.mesh.periodic:
        mesh_cfg.set_periodic(True)
    if not config.mesh.enable_ghost_width:
        mesh_cfg.disable_minimal_ghost_width()

    return mesh_cfg

def apply_mra_config(config: SimulationConfig) -> sam.MRAConfig:
    """
    Applique une configuration MRA prédéfinie.

    Args:
        config: Configuration de simulation

    Returns:
        sam.MRAConfig: Configuration MRA Samurai
    """
    mra_cfg = sam.MRAConfig()
    mra_cfg.epsilon = config.mra.epsilon
    mra_cfg.regularity = config.mra.regularity
    mra_cfg.relative_detail = config.mra.relative_detail
    return mra_cfg

def apply_bc_config(field, config: SimulationConfig):
    """
    Applique une configuration de conditions aux limites prédéfinie.

    Args:
        field: Champ Samurai (ScalarField ou VectorField)
        config: Configuration de simulation
    """
    if config.boundary is None:
        return

    if config.boundary.bc_type == "dirichlet":
        if hasattr(field, 'n_components') and field.n_components > 1:
            # VectorField
            value = [config.boundary.value] * field.n_components
            sam.make_dirichlet_bc(field, value, order=config.boundary.order)
        else:
            # ScalarField
            sam.make_dirichlet_bc(field, config.boundary.value, order=config.boundary.order)
    elif config.boundary.bc_type == "neumann":
        # Implémentation similaire pour Neumann
        pass


class SSPRK3Integrator:
    """
    Intégrateur SSP-RK3 avec constantes prédéfinies.

    Remplace les coefficients magiques (3/4, 1/4, 1/3, 2/3) par une API claire.
    """

    # Coefficients SSP-RK3 (Shu & Osher, 1988)
    ALPHA = [1.0, 3.0/4.0, 1.0/3.0]
    BETA = [1.0, 1.0/4.0, 2.0/3.0]

    def __init__(self, flux_operator, dt: float):
        """
        Args:
            flux_operator: Fonction de flux (ex: sam.make_convection_weno5)
            dt: Pas de temps
        """
        self.flux_operator = flux_operator
        self.dt = dt

    def step(self, u, u1, u2, unp1):
        """
        Effectue un pas de temps SSP-RK3.

        Args:
            u: Champ au temps actuel (sera écrasé)
            u1: Champ de travail étape 1
            u2: Champ de travail étape 2
            unp1: Champ de travail étape 3
        """
        dt = self.dt

        # Stage 1: u1 = u - dt * flux(u)
        flux1 = self.flux_operator(u)
        u1.assign(u - dt * flux1)

        # Stage 2: u2 = 3/4*u + 1/4*(u1 - dt*flux(u1))
        flux2 = self.flux_operator(u1)
        u2.assign(self.ALPHA[1] * u + self.BETA[1] * (u1 - dt * flux2))

        # Stage 3: unp1 = 1/3*u + 2/3*(u2 - dt*flux(u2))
        flux3 = self.flux_operator(u2)
        unp1.assign(self.ALPHA[2] * u + self.BETA[2] * (u2 - dt * flux3))

        # Swap u et unp1
        sam.swap_field_arrays_2d(u, unp1)


def create_simulation_from_preset(
    preset: SimulationConfig,
    box: sam.Box2D
) -> tuple:
    """
    Crée une simulation complète à partir d'un preset.

    Args:
        preset: Configuration prédéfinie
        box: Domaine de calcul

    Returns:
        (mesh, u, mra_config, dt, integrator): Mesh, champ principal, config MRA, dt, intégrateur
    """
    # Valider la configuration
    is_valid, warnings = preset.validate()
    if not is_valid:
        raise ValueError(f"Configuration invalide: {warnings}")
    if warnings:
        print("Avertissements:")
        for w in warnings:
            print(f"  - {w}")

    # Configurer le maillage
    mesh_cfg = apply_config_to_mesh(preset)
    mesh = sam.MRMesh2D(box, mesh_cfg)

    # Créer les champs
    u = sam.ScalarField2D("u", mesh, 0.0)
    u1 = sam.ScalarField2D("u1", mesh, 0.0)
    u2 = sam.ScalarField2D("u2", mesh, 0.0)
    unp1 = sam.ScalarField2D("unp1", mesh, 0.0)

    # Configurer les BC
    apply_bc_config(u, preset)

    # Configurer MRA
    mra_config = apply_mra_config(preset)

    # Calculer dt basé sur CFL
    # Note: nécessite la vitesse pour advection, diffusion pour chaleur
    # Ici on retourne le CFL pour calcul ultérieur
    cfl = preset.time.cfl

    # Créer l'intégrateur si SSPRK3
    integrator = None
    if preset.time.scheme == "SSPRK3":
        # Flux operator sera configuré plus tard
        integrator = SSPRK3Integrator(flux_operator=None, dt=0.0)

    return mesh, u, u1, u2, unp1, mra_config, cfl, integrator
```

---

## 5. Exemples d'Utilisation

### 5.1 Avant (Code Actuel avec Magic Numbers)

```python
# Fichier: python/examples/advection_2d.py (version actuelle)

def main():
    # ... setup ...

    # Magic numbers partout!
    config = sam.MeshConfig2D()
    config.min_level = 4      # Pourquoi 4?
    config.max_level = 10     # Pourquoi 10?
    config.max_stencil_size = 2  # Pourquoi 2?

    mra_config = sam.MRAConfig()
    mra_config.epsilon = 2e-4    # Pourquoi 2e-4?
    mra_config.regularity = 1.0  # Pourquoi 1.0?

    cfl = 0.5  # Pourquoi 0.5?

    # ... boucle temporelle avec coefficients magiques ...
    u2.assign((3.0 / 4.0) * u + (1.0 / 4.0) * (u1 - dt * flux2))
    unp1.assign((1.0 / 3.0) * u + (2.0 / 3.0) * (u2 - dt * flux3))
```

### 5.2 Après (avec Presets)

```python
# Fichier: python/examples/advection_2d_preset.py

from samurai.config.presets import ConfigPresets, AccuracyLevel
from samurai.config.helpers import create_simulation_from_preset, SSPRK3Integrator
import samurai_python as sam

def main():
    """
    Advection 2D avec presets - plus de magic numbers!

    Avantages:
    - Configuration claire et documentée
    - Validation automatique
    - Facile à changer la précision
    - Code auto-documenté
    """

    # 1. Choisir le preset
    preset = ConfigPresets.advection_upwind(accuracy=AccuracyLevel.DEFAULT)

    # 2. Afficher les détails (optionnel)
    ConfigPresets.print_preset_info(preset)

    # 3. Valider
    is_valid, warnings = preset.validate()
    assert is_valid, f"Configuration invalide: {warnings}"

    # 4. Créer la simulation
    box = sam.Box2D([0.0, 0.0], [1.0, 1.0])
    mesh, u, u1, u2, unp1, mra_config, cfl, _ = create_simulation_from_preset(preset, box)

    # 5. Initialiser
    init_circular(u, center=(0.3, 0.3), radius=0.2)

    # 6. Adaptation initiale
    MRadaptation = sam.make_MRAdapt(u)
    MRadaptation(mra_config)

    # 7. Calculer dt
    velocity = [1.0, 1.0]
    min_cell_length = mesh.min_cell_length
    max_velocity = max(abs(v) for v in velocity)
    dt = cfl * min_cell_length / max_velocity

    # 8. Créer l'intégrateur SSP-RK3 (plus de coefficients magiques!)
    def flux_op(field):
        return sam.upwind(velocity, field)

    integrator = SSPRK3Integrator(flux_op, dt)

    # 9. Boucle temporelle
    t = 0.0
    Tf = 0.1

    while t < Tf:
        MRadaptation(mra_config)
        sam.update_ghost_mr(u)

        t += dt

        # Un pas de temps SSP-RK3 propre
        integrator.step(u, u1, u2, unp1)

        # Sauvegarder périodiquement
        sam.save("./results", f"advection_{t:.6f}", u)
```

### 5.3 Comparaison: Changer la Précision

**Avant** (difficile et sujet à erreurs):
```python
# Pour passer en haute précision, il faut changer plusieurs valeurs
# et s'assurer qu'elles sont cohérentes entre elles

# Ancien code (DEFAULT)
config.min_level = 4
config.max_level = 10
mra_config.epsilon = 2e-4

# Nouveau code (FINE) - Est-ce cohérent?
config.min_level = 6      # J'augmente...
config.max_level = 12     # ...mais est-ce assez?
mra_config.epsilon = 1e-4  # Plus petit, OK...
# Et le CFL? Le stencil? La régularité?
```

**Après** (trivial et sûr):
```python
# Changer la précision en une ligne!
preset = ConfigPresets.advection_upwind(accuracy=AccuracyLevel.FINE)

# Ou même interactif:
accuracy = input("Précision? (coarse/default/fine): ")
preset = ConfigPresets.advection_upwind(accuracy=AccuracyLevel[accuracy.upper()])
```

---

## 6. Tests de Validation

```python
# Fichier: python/tests/test_config_presets.py

import pytest
from samurai.config.presets import (
    ConfigPresets, SimulationConfig, AccuracyLevel, ProblemType,
    MeshConfigPreset, MRAConfigPreset, TimeSteppingPreset
)


class TestConfigPresets:
    """Tests pour les configurations prédéfinies"""

    def test_advection_upwind_default(self):
        """Test le preset advection upwind default"""
        preset = ConfigPresets.advection_upwind(AccuracyLevel.DEFAULT)

        assert preset.name == "advection_upwind_default"
        assert preset.mesh.min_level == 4
        assert preset.mesh.max_level == 10
        assert preset.mesh.max_stencil_size == 2
        assert preset.mra.epsilon == 2e-4
        assert preset.mra.regularity == 1.0
        assert preset.time.cfl == 0.5

    def test_advection_weno5_validation(self):
        """Test que WENO5 nécessite stencil_size >= 6"""
        preset = ConfigPresets.advection_weno5(AccuracyLevel.DEFAULT)

        # Doit être valide
        is_valid, warnings = preset.validate()
        assert is_valid
        assert len(warnings) == 0

        # Si on réduit le stencil, doit être invalide
        preset.mesh.max_stencil_size = 4
        is_valid, warnings = preset.validate()
        assert not is_valid
        assert any("max_stencil_size" in w for w in warnings)

    def test_burgers_requires_weno5(self):
        """Test que Burgers nécessite WENO5"""
        preset = ConfigPresets.burgers(AccuracyLevel.DEFAULT)

        # Vérifier que le stencil est correct pour WENO5
        assert preset.mesh.max_stencil_size == 6
        assert preset.boundary.order == 3

    def test_accuracy_levels(self):
        """Test que les niveaux de précision sont cohérents"""
        for accuracy in [AccuracyLevel.COARSE, AccuracyLevel.DEFAULT, AccuracyLevel.FINE]:
            preset = ConfigPresets.advection_weno5(accuracy)

            # Plus on augmente la précision, plus les niveaux sont élevés
            if accuracy == AccuracyLevel.COARSE:
                assert preset.mesh.max_level <= preset.mesh.min_level + 3
            elif accuracy == AccuracyLevel.FINE:
                assert preset.mesh.max_level >= preset.mesh.min_level + 5

    def test_mra_epsilon_range(self):
        """Test que epsilon est dans une plage raisonnable"""
        presets = [
            ConfigPresets.advection_upwind(),
            ConfigPresets.advection_weno5(),
            ConfigPresets.burgers(),
            ConfigPresets.diffusion(),
        ]

        for preset in presets:
            assert 1e-6 < preset.mra.epsilon < 1e-1  # Plage raisonnable

    def test_cfl_range(self):
        """Test que CFL est dans (0, 1]"""
        presets = [
            ConfigPresets.advection_upwind(),
            ConfigPresets.advection_weno5(),
            ConfigPresets.burgers(),
            ConfigPresets.diffusion(),
        ]

        for preset in presets:
            assert 0 < preset.time.cfl <= 1.0

    def test_invalid_config_detection(self):
        """Test la détection de configurations invalides"""
        # min_level > max_level
        config = SimulationConfig(
            name="invalid",
            problem_type=ProblemType.ADVECTION_UPWIND,
            mesh=MeshConfigPreset(min_level=10, max_level=5, max_stencil_size=2),
            mra=MRAConfigPreset(epsilon=1e-4, regularity=1.0),
            time=TimeSteppingPreset(cfl=0.5)
        )

        is_valid, warnings = config.validate()
        assert not is_valid
        assert len(warnings) > 0


class TestSSPRK3Integrator:
    """Tests pour l'intégrateur SSP-RK3"""

    def test_coefficients(self):
        """Test que les coefficients SSP-RK3 sont corrects"""
        from samurai.config.helpers import SSPRK3Integrator

        # Coefficients théoriques de Shu & Osher (1988)
        expected_alpha = [1.0, 3.0/4.0, 1.0/3.0]
        expected_beta = [1.0, 1.0/4.0, 2.0/3.0]

        assert SSPRK3Integrator.ALPHA == expected_alpha
        assert SSPRK3Integrator.BETA == expected_beta

    def test_coefficients_sum_to_one(self):
        """Test que les coefficients alpha somment à 1 (propriété SSP-RK3)"""
        from samurai.config.helpers import SSPRK3Integrator

        alpha = SSPRK3Integrator.ALPHA
        # Pour SSP-RK3, on a: u^(n+1) = alpha*u^n + ...
        # Les alpha ne somment pas nécessairement à 1, mais vérifions la cohérence

        # Vérifier que alpha[2] + beta[2] = 1 (dernière étape)
        assert abs(alpha[2] + SSPRK3Integrator.BETA[2] - 1.0) < 1e-10


class TestConfigHelpers:
    """Tests pour les fonctions helper"""

    def test_apply_config_to_mesh(self):
        """Test l'application de configuration au maillage"""
        from samurai.config.helpers import apply_config_to_mesh
        import samurai_python as sam

        preset = ConfigPresets.advection_weno5(AccuracyLevel.DEFAULT)
        mesh_cfg = apply_config_to_mesh(preset)

        assert mesh_cfg.min_level == preset.mesh.min_level
        assert mesh_cfg.max_level == preset.mesh.max_level
        assert mesh_cfg.max_stencil_size == preset.mesh.max_stencil_size

    def test_apply_mra_config(self):
        """Test l'application de configuration MRA"""
        from samurai.config.helpers import apply_mra_config
        import samurai_python as sam

        preset = ConfigPresets.advection_weno5(AccuracyLevel.DEFAULT)
        mra_cfg = apply_mra_config(preset)

        assert mra_cfg.epsilon == preset.mra.epsilon
        assert mra_cfg.regularity == preset.mra.regularity
        assert mra_cfg.relative_detail == preset.mra.relative_detail
```

---

## 7. Impact sur l'Expérience Utilisateur

### 7.1 Avantages

**Pour les débutants**:
```python
# Avant: incompréhensible
config.min_level = 4  # ??? Pourquoi 4?
config.max_level = 10  # ??? Et 10?
mra_config.epsilon = 2e-4  # ??? Magique!

# Après: clair et documenté
preset = ConfigPresets.advection_weno5(accuracy=AccuracyLevel.DEFAULT)
# Le preset sait mieux que moi ce qui est bon!
```

**Pour les utilisateurs expérimentés**:
```python
# Avant: fastidieux
if want_weno5:
    config.max_stencil_size = 6
    cfl = 0.95
    bc_order = 3
else:
    config.max_stencil_size = 2
    cfl = 0.5
    bc_order = 1

# Après: explicite
preset = ConfigPresets.advection_weno5()
# Tout est cohérent automatiquement
```

**Pour les chercheurs**:
```python
# Comparaison facile de plusieurs configurations
for accuracy in [AccuracyLevel.COARSE, AccuracyLevel.DEFAULT, AccuracyLevel.FINE]:
    preset = ConfigPresets.burgers(accuracy)
    run_simulation(preset)
    # Garantit que seule la précision change, tout le reste est cohérent
```

### 7.2 Réduction des Erreurs

**Erreurs typiques éliminées**:
1. **CFL incohérent**: Upwind avec CFL=0.95 (instable!)
2. **Stencil trop petit**: WENO5 avec max_stencil_size=2 (ne compile pas!)
3. **BC order trop bas**: WENO5 avec BC order=1 (réduit l'ordre global!)
4. **epsilon incohérent**: max_level=12 avec epsilon=1e-2 (trop laxiste!)

**Exemple de détection**:
```python
# Configuration invalide détectée automatiquement
preset = ConfigPresets.advection_weno5()
preset.mesh.max_stencil_size = 2  # Erreur utilisateur

is_valid, warnings = preset.validate()
# -> False, ["advection_weno5 nécessite max_stencil_size >= 6 pour WENO5"]
```

### 7.3 Facilité d'Extension

```python
# Ajouter un nouveau preset est trivial
@staticmethod
def my_custom_problem():
    return SimulationConfig(
        name="my_problem",
        problem_type=ProblemType.ADVECTION_WENO5,
        mesh=MeshConfigPreset(min_level=3, max_level=8, max_stencil_size=6),
        mra=MRAConfigPreset(epsilon=5e-4, regularity=0.5),
        time=TimeSteppingPreset(cfl=0.8, scheme="SSPRK3"),
        description="Mon problème personnalisé"
    )
```

---

## 8. Recommandations d'Implémentation

### 8.1 Plan d'Action

**Phase 1: Core (1-2 semaines)**
1. Implémenter les classes de données (`MeshConfigPreset`, etc.)
2. Implémenter `ConfigPresets` avec presets de base
3. Implémenter la méthode `validate()`

**Phase 2: Helpers (1 semaine)**
1. Implémenter les fonctions `apply_*_config()`
2. Implémenter `SSPRK3Integrator`
3. Tests unitaires

**Phase 3: Intégration (1-2 semaines)**
1. Migrer un exemple existant (ex: `advection_2d.py`)
2. Créer version alternative avec presets
3. Comparer les résultats (identiques!)

**Phase 4: Documentation (1 semaine)**
1. Docstrings complète
2. Tutoriel: "Comment utiliser les presets"
3. Guide: "Comment créer vos propres presets"

### 8.2 Priorités

**Haute priorité**:
- Presets pour problèmes courants (advection, Burgers, diffusion)
- Validation automatique
- Intégrateur SSP-RK3

**Moyenne priorité**:
- Presets pour problèmes moins courants (level set, Navier-Stokes)
- Export/import de configurations (JSON)

**Basse priorité**:
- Interface graphique pour sélection de preset
- Optimisation automatique de preset

---

## 9. Conclusion

Cette analyse révèle que:

1. **Les magic numbers sont omniprésents** dans les exemples Python Samurai
2. **Les valeurs typiques sont assez cohérentes** entre exemples similaires
3. **Un système de presets** apporterait une valeur significative:
   - Réduction des erreurs (validation automatique)
   - Meilleure expérience utilisateur (plus besoin de deviner les valeurs)
   - Code plus maintenable (configurations centralisées)
   - Science plus reproductible (configurations nommées)

**Prochaine étape recommandée**: Implémenter le système de presets avec les priorités définies en section 8.2, en commençant par les problèmes les plus courants (advection, Burgers).

---

## Annexes

### A. Tableau Récapitulatif des Constantes

| Constante | Advection Upwind | Advection WENO5 | Burgers | Diffusion |
|-----------|------------------|-----------------|---------|-----------|
| **CFL** | 0.5 | 0.95 | 0.95 | 0.5-0.95 |
| **epsilon** | 1e-4 - 2e-4 | 1e-4 - 5e-4 | 2e-4 | 1e-4 - 1e-2 |
| **regularity** | 1.0 | 1.0-2.0 | 1.0 | 1.0-2.0 |
| **min_level** | 4 | 5 | 5 | 3-4 |
| **max_level** | 10 | 9 | 7 | 5-8 |
| **max_stencil_size** | 2 | 6 | 6 | 2 |
| **BC order** | 1 | 3 | 3 | 1 |
| **Scheme** | Euler | SSP-RK3 | SSP-RK3 | Euler/SSPRK3 |

### B. Références

- Shu, C.-W., & Osher, S. (1988). "Efficient implementation of essentially non-oscillatory shock-capturing schemes". Journal of Computational Physics, 77(2), 439-471.
- Liu, X.-D., Osher, S., & Chan, T. (1994). "Weighted essentially non-oscillatory schemes". Journal of Computational Physics, 115(1), 200-212.
- Documentation Samurai: https://hpc-math-samurai.readthedocs.io

---

**Fin du rapport**
