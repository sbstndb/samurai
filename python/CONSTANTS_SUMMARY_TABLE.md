# Tableau Synthétique des Constantes Samurai Python

## Référence Rapide des Valeurs Typiques

### Par Type de Problème

| **Paramètre** | **Advection Upwind** | **Advection WENO5** | **Burgers WENO5** | **Diffusion** | **Level Set** |
|---------------|---------------------|---------------------|-------------------|---------------|---------------|
| **CFL** | 0.5 | 0.95 | 0.95 | 0.5-0.95 | 0.625 (5/8) |
| **epsilon** | 2e-4 | 1e-4 | 2e-4 | 1e-4 | 1e-3 |
| **regularity** | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| **min_level** | 4 | 5 | 5 | 4 | 1-3 |
| **max_level** | 10 | 9 | 7 | 8 | 4-8 |
| **max_stencil_size** | 2 | **6** | **6** | 2 | **6** |
| **BC order** | 1 | **3** | **3** | 1 | **3** |
| **Time Scheme** | Euler | SSP-RK3 | SSP-RK3 | Euler | SSP-RK3 |

### Par Niveau de Précision

| **Paramètre** | **COARSE** | **DEFAULT** | **FINE** | **ULTRA-FINE** |
|---------------|------------|-------------|----------|----------------|
| **min_level** | 3-4 | 4-5 | 5-6 | 6-7 |
| **max_level** | 6-7 | 9-10 | 11-12 | 12-14 |
| **epsilon** | 1e-3 - 5e-4 | 1e-4 - 2e-4 | 5e-5 - 1e-4 | 1e-5 - 5e-5 |
| **regularity** | 0.5-1.0 | 1.0 | 1.5-2.0 | 2.0-3.0 |
| **Ratio max/min** | ~2x | ~2-3x | ~2-3x | ~2-3x |

### Coefficients SSP-RK3 (TVD-RK3)

```python
# Stage 1
alpha_1 = 1.0
beta_1  = 1.0

# Stage 2
alpha_2 = 3.0 / 4.0
beta_2  = 1.0 / 4.0

# Stage 3
alpha_3 = 1.0 / 3.0
beta_3  = 2.0 / 3.0
```

**Formules**:
- `u1 = u - dt * flux(u)`
- `u2 = (3/4) * u + (1/4) * (u1 - dt * flux(u1))`
- `unp1 = (1/3) * u + (2/3) * (u2 - dt * flux(u2))`

---

## Magic Numbers dans les Exemples Actuels

### Exemple: `advection_2d.py`

```python
# Ligne 83-84
config.min_level = 4      # MAGIC: Pourquoi 4?
config.max_level = 10     # MAGIC: Pourquoi 10?

# Ligne 108-109
mra_config.epsilon = 2e-4    # MAGIC: Pourquoi 2e-4?
mra_config.regularity = 1.0  # MAGIC: Pourquoi 1.0?

# Ligne 64
cfl = 0.5  # MAGIC: Pourquoi 0.5?

# Ligne 157
upwind_result = sam.upwind(velocity, u)  # MAGIC: Pourquoi upwind?
```

### Exemple: `burgers_2d.py`

```python
# Ligne 95-96
min_level = 5  # MAGIC: Pourquoi 5?
max_level = 7  # MAGIC: Pourquoi 7?

# Ligne 110
config.max_stencil_size = 6  # MAGIC: Pourquoi 6? (Obligatoire pour WENO5!)

# Ligne 132
sam.make_dirichlet_bc(u, [0.0, 0.0], order=3)  # MAGIC: Pourquoi order=3?

# Ligne 142
mra_config.epsilon = 2e-4  # MAGIC: Pourquoi 2e-4?

# Ligne 88
cfl = 0.95  # MAGIC: Pourquoi 0.95?

# Ligne 206
u2.assign((3.0 / 4.0) * u + (1.0 / 4.0) * (u1 - dt * flux2))  # MAGIC: Coefficients!
```

---

## Relations entre Paramètres

### max_stencil_size vs Schéma

| **Schéma** | **min_stencil_size** | **max_stencil_size** | **Note** |
|------------|---------------------|---------------------|----------|
| Upwind | 2 | 2 | Fixe |
| WENO5 | **6** | 6 | Obligatoire |
| HOUC5 | **6** | 6 | Obligatoire |
| Central diff. | 3 | 5 | Variable |

### CFL vs Stabilité

| **Schéma** | **CFL max** | **CFL recommandé** | **Stabilité** |
|------------|-------------|-------------------|---------------|
| Euler (forward) | 0.5-1.0 | 0.5 | Conditionnelle |
| SSP-RK3 | 1.0 | 0.95-0.99 | Très stable |
| SSP-RK2 | 0.5-1.0 | 0.5 | Conditionnelle |

### BC order vs Précision Globale

| **Schéma** | **BC order min** | **BC order recommandé** | **Impact** |
|------------|------------------|------------------------|------------|
| Upwind (ordre 1) | 1 | 1 | Aucun |
| WENO5 (ordre 5) | **3** | 3 | Si <3: réduit ordre global |
| Central (ordre 2) | 2 | 2 | Si <2: réduit ordre global |

---

## Valeurs par Défaut dans le Code C++

### MRAConfig (d'après les tests Python)

```cpp
// Valeurs par défaut observées dans test_mra_config.py
epsilon = 1e-4        // Ligne 34
regularity = 1.0      // Ligne 38
relative_detail = false  // Ligne 43
```

### MeshConfig

```cpp
// Valeurs typiques (pas de défaut unique, dépend du problème)
min_level = 0         // Souvent 1-5 dans les exemples
max_level = 0         // Souvent 4-12 dans les exemples
max_stencil_size = 2  // Défaut pour upwind
periodic = false
```

---

## Validation des Plages Admissibles

### Règles de Cohérence

```python
def validate_config(config):
    """Règles de validation pour une configuration Samurai"""

    warnings = []

    # 1. Maillage
    if config.min_level > config.max_level:
        warnings.append("min_level doit être <= max_level")

    if config.max_level - config.min_level > 8:
        warnings.append("Écart max_level-min_level trop grand (> 8)")

    # 2. Stencil vs Schéma
    if scheme == "WENO5" and config.max_stencil_size < 6:
        warnings.append("WENO5 nécessite max_stencil_size >= 6")

    if scheme == "HOUC5" and config.max_stencil_size < 6:
        warnings.append("HOUC5 nécessite max_stencil_size >= 6")

    # 3. CFL
    if config.cfl <= 0:
        errors.append("CFL doit être > 0")

    if config.cfl > 1.0:
        warnings.append("CFL > 1.0 peut être instable")

    # 4. MRA
    if config.epsilon <= 0:
        errors.append("epsilon doit être > 0")

    if config.epsilon > 0.1:
        warnings.append("epsilon > 0.1 est très laxiste")

    if config.epsilon < 1e-8:
        warnings.append("epsilon < 1e-8 est inutilement petit")

    if config.regularity < 0:
        errors.append("regularity doit être >= 0")

    if config.regularity > 3.0:
        warnings.append("regularity > 3.0 peut sur-contraindre")

    # 5. BC order vs Schéma
    if scheme == "WENO5" and config.bc_order < 3:
        warnings.append("WENO5 avec bc_order < 3 réduit l'ordre global")

    # 6. CFL vs Schéma
    if scheme == "Euler" and config.cfl > 0.6:
        warnings.append("Euler forward avec CFL > 0.6 peut être instable")

    return warnings
```

---

## Exemples de Configurations Incohérentes

### Erreur 1: WENO5 avec stencil trop petit

```python
# ❌ MAUVAIS
config.max_stencil_size = 2
flux = sam.make_convection_weno5(u)  # ERREUR à l'exécution!

# ✅ BON
config.max_stencil_size = 6
flux = sam.make_convection_weno5(u)
```

### Erreur 2: CFL trop élevé pour Euler

```python
# ❌ MAUVAIS
cfl = 0.95
# Euler forward avec CFL=0.95 -> INSTABLE!

# ✅ BON
cfl = 0.5  # Stable pour Euler
```

### Erreur 3: BC order trop bas pour WENO5

```python
# ❌ MAUVAIS
sam.make_dirichlet_bc(u, 0.0, order=1)  # Réduit l'ordre global à 1!
flux = sam.make_convection_weno5(u)     # Ordre 5, mais BC ordre 1

# ✅ BON
sam.make_dirichlet_bc(u, 0.0, order=3)  # Ordre 3, OK pour WENO5
flux = sam.make_convection_weno5(u)
```

### Erreur 4: epsilon incohérent avec max_level

```python
# ❌ MAUVAIS
config.min_level = 4
config.max_level = 12      # Résolution très fine
mra_config.epsilon = 1e-2  # Tolérance très laxiste!

# ✅ BON
config.min_level = 4
config.max_level = 12
mra_config.epsilon = 1e-4  # Cohérent avec la résolution fine
```

---

## Checklist pour Nouvelle Simulation

### 1. Choisir le Type de Problème

- [ ] Advection linéaire
- [ ] Burgers
- [ ] Diffusion
- [ ] Système hyperbolique
- [ ] Autre

### 2. Choisir le Schéma Numérique

- [ ] Upwind (ordre 1, robuste)
- [ ] WENO5 (ordre 5, précis)
- [ ] Central (ordre 2)
- [ ] Autre

### 3. Définir la Précision

- [ ] COARSE (tests rapides)
- [ ] DEFAULT (production)
- [ ] FINE (haute précision)
- [ ] ULTRA-FINE (benchmark)

### 4. Configurer les Paramètres

```python
# Après choix, utiliser le preset approprié
preset = ConfigPresets.{problem_type}(AccuracyLevel.{accuracy})
```

### 5. Valider

```python
is_valid, warnings = preset.validate()
assert is_valid, f"Configuration invalide: {warnings}"
```

### 6. Exécuter

```python
mesh, u, mra_config, cfl, integrator = create_simulation_from_preset(preset, box)
```

---

## Références Croisées

### Fichiers d'Exemples Analysés

| **Fichier** | **Problème** | **Schéma** | **CFL** | **epsilon** |
|-------------|--------------|------------|---------|-------------|
| `advection_2d.py` | Advection 2D | Upwind | 0.5 | 2e-4 |
| `linear_convection.py` | Convection 2D | WENO5 | 0.95 | 1e-4 |
| `burgers_2d.py` | Burgers 2D | WENO5 | 0.95 | 2e-4 |
| `burgers_2d_simple.py` | Burgers 2D | WENO5 | 0.5 | N/A |
| `linear_convection_obstacle.py` | Convection + obstacle | WENO5 | 0.95 | 1e-3 |

### Fichiers C++ Correspondants

| **Python** | **C++** | **Concordance** |
|------------|---------|-----------------|
| `advection_2d.py` | `demos/FiniteVolume/advection_2d.cpp` | ✅ Identique |
| `linear_convection.py` | `demos/FiniteVolume/linear_convection.cpp` | ✅ Identique |
| `burgers_2d.py` | `demos/FiniteVolume/burgers.cpp` | ✅ Identique |

---

**Note**: Ce tableau est une référence rapide. Pour une analyse détaillée, voir `SAMURAI_CONSTANTS_ANALYSIS.md`.
