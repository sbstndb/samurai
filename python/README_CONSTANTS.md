# Analyse des Constantes Samurai Python - Résumé Exécutif

## Vue d'Ensemble

Ce document présente une analyse approfondie des constantes numériques ("magic numbers") dans les bindings Python de la bibliothèque Samurai (AMR/MRA pour équations aux dérivées partielles).

**Date**: 7 Janvier 2026
**Scope**: 5 exemples Python + ~30 démos C++
**Livrables**: 3 documents + 1 outil utilitaire

---

## Principales Constations

### 1. Les Magic Numbers sont Omniprésents

**Exemple typique** (`advection_2d.py`):
```python
config.min_level = 4      # ❌ Pourquoi 4?
config.max_level = 10     # ❌ Pourquoi 10?
mra_config.epsilon = 2e-4 # ❌ Pourquoi 2e-4?
cfl = 0.5                 # ❌ Pourquoi 0.5?
```

**Problème**: Aucune indication du pourquoi de ces valeurs.

### 2. Les Valeurs sont Cohérentes entre Exemples Similaires

| **Type de Problème** | **CFL** | **epsilon** | **max_stencil_size** | **BC order** |
|----------------------|---------|-------------|---------------------|--------------|
| Advection Upwind | 0.5 | 2e-4 | 2 | 1 |
| Advection WENO5 | 0.95 | 1e-4 | 6 | 3 |
| Burgers | 0.95 | 2e-4 | 6 | 3 |
| Diffusion | 0.5-0.95 | 1e-4 | 2 | 1 |

**Observation**: Les valeurs typiques sont assez constantes, ce qui justifie un système de presets.

### 3. Relations de Dépendance Critiques

Certaines combinaisons sont **obligatoires**:
- WENO5 nécessite `max_stencil_size >= 6`
- WENO5 nécessite `BC order >= 3` pour préserver l'ordre 5
- Euler forward nécessite `CFL <= 0.5-0.6` pour la stabilité
- SSP-RK3 permet `CFL <= 0.95-0.99`

---

## Livrables

### 1. Rapport d'Analyse Détaillé

**Fichier**: `SAMURAI_CONSTANTS_ANALYSIS.md` (600+ lignes)

**Contenu**:
- Inventaire complet des constantes par type de problème
- Proposition de classe `ConfigPresets` avec API complète
- Coefficients SSP-RK3 documentés
- Exemples d'utilisation avant/après
- Tests de validation proposés

**Points clés**:
- Architecture avec `SimulationConfig`, `MeshConfigPreset`, `MRAConfigPreset`
- Méthode `validate()` pour détection automatique d'incohérences
- Intégrateur `SSPRK3Integrator` éliminant les coefficients magiques
- Helper functions pour application aisée des presets

### 2. Tableau de Référence Rapide

**Fichier**: `CONSTANTS_SUMMARY_TABLE.md`

**Contenu**:
- Tableaux synthétiques par type de problème
- Coefficients SSP-RK3
- Relations entre paramètres
- Checklist pour nouvelle simulation
- Exemples de configurations incohérentes

**Utilité**: Référence rapide pour développeurs et utilisateurs.

### 3. Outil de Validation

**Fichier**: `python/scripts/config_validator.py` (600+ lignes)

**Fonctionnalités**:
```bash
# Valider un fichier
python config_validator.py validate advection_2d.py

# Comparer deux configurations
python config_validator.py compare advection_2d.py burgers_2d.py

# Suggérer une configuration
python config_validator.py suggest advection weno5 default

# Scanner un répertoire
python config_validator.py scan python/examples/
```

**Résultats sur les exemples actuels**:
- ✅ Tous les fichiers sont valides
- ⚠️ 1 avertissement (optimisation possible)

---

## Propositions d'Amélioration

### Solution Proposée: Système de Presets

#### Avant (Code Actuel)
```python
# Magic numbers - difficile à maintenir
config.min_level = 4
config.max_level = 10
config.max_stencil_size = 2
mra_config.epsilon = 2e-4
mra_config.regularity = 1.0
cfl = 0.5
```

#### Après (avec Presets)
```python
# Clair, documenté, validé automatiquement
preset = ConfigPresets.advection_upwind(accuracy=AccuracyLevel.DEFAULT)
is_valid, warnings = preset.validate()  # Validation automatique!
mesh, u, mra_config, cfl, integrator = create_simulation_from_preset(preset, box)
```

#### Bénéfices

**Pour les débutants**:
- Plus besoin de connaître les valeurs typiques
- Validation automatique évite les erreurs
- Code auto-documenté

**Pour les experts**:
- Configuration centralisée et cohérente
- Facile de changer la précision (1 ligne!)
- Science plus reproductible

**Pour le projet**:
- Réduction des bugs liés aux configurations
- Documentation intégrée au code
- Tests plus faciles (presets nommés)

### Architecture Proposée

```
samurai.config.presets
├── ProblemType (Enum)
├── AccuracyLevel (Enum)
├── SimulationConfig (dataclass)
│   ├── mesh: MeshConfigPreset
│   ├── mra: MRAConfigPreset
│   ├── time: TimeSteppingPreset
│   └── validate() -> (bool, List[str])
└── ConfigPresets (class)
    ├── advection_upwind(accuracy)
    ├── advection_weno5(accuracy)
    ├── burgers(accuracy)
    └── diffusion(accuracy)
```

---

## Plan d'Implémentation

### Phase 1: Core (1-2 semaines)
- [ ] Implémenter les classes de données
- [ ] Implémenter `ConfigPresets` avec presets de base
- [ ] Implémenter `validate()`

### Phase 2: Helpers (1 semaine)
- [ ] Implémenter `apply_*_config()`
- [ ] Implémenter `SSPRK3Integrator`
- [ ] Tests unitaires

### Phase 3: Intégration (1-2 semaines)
- [ ] Migrer un exemple existant
- [ ] Créer version alternative avec presets
- [ ] Comparer les résultats

### Phase 4: Documentation (1 semaine)
- [ ] Docstrings complètes
- [ ] Tutoriel d'utilisation
- [ ] Guide de création de presets

**Total estimé**: 4-6 semaines

---

## Tests de Validation

### Résultats sur les Exemples Actuels

```bash
$ python config_validator.py scan python/examples/

Scan de 5 fichiers Python dans python/examples/
======================================================================

✅ Tous les fichiers sont valides!
```

### Exemple de Rapport Détaillé

```bash
$ python config_validator.py validate advection_2d.py

======================================================================
RAPPORT DE VALIDATION
======================================================================
Configuration extraite de: python/examples/advection_2d.py
======================================================================
  min_level           : 4
  max_level           : 10
  cfl                 : 0.5
  epsilon             : 0.0002
  regularity          : 1.0
  scheme              : upwind
  time_scheme         : euler

----------------------------------------------------------------------
✅ Configuration VALIDE
✅ Aucun problème détecté!
======================================================================
```

### Exemple de Comparaison

```bash
$ python config_validator.py compare advection_2d.py linear_convection.py

======================================================================
RAPPORT DE COMPARAISON
======================================================================
DIFFÉRENCES:
  min_level           : 4 vs 5
  max_level           : 10 vs 9
  max_stencil_size    : None vs 6
  cfl                 : 5.00e-01 vs 9.50e-01 (+90.0%)
  epsilon             : 2.00e-04 vs 1.00e-04 (+50.0%)
  scheme              : upwind vs weno5
  time_scheme         : euler vs SSPRK3
======================================================================
```

---

## Statistiques

### Fichiers Analysés

| **Type** | **Quantité** |
|----------|--------------|
| Exemples Python | 5 |
| Démos C++ | ~30 |
| Tests Python | 15 |
| **Total** | **~50** |

### Constantes Inventoriées

| **Catégorie** | **Nombre** |
|---------------|------------|
| Paramètres de maillage | 3 (min/max_level, stencil_size) |
| Paramètres MRA | 2 (epsilon, regularity) |
| Paramètres temporels | 1 (CFL) |
| Paramètres BC | 1 (order) |
| Schémas numériques | 6 (upwind, WENO5, etc.) |
| **Total** | **13** |

### Valeurs Typiques Identifiées

| **Paramètre** | **Plage** | **Valeurs typiques** |
|---------------|-----------|---------------------|
| CFL | 0.01 - 1.0 | 0.5 (upwind), 0.95 (WENO5) |
| epsilon | 1e-8 - 0.1 | 1e-4 - 2e-4 |
| regularity | 0.0 - 3.0 | 1.0 |
| min_level | 0 - 10 | 3 - 5 |
| max_level | 0 - 15 | 7 - 12 |
| max_stencil_size | 2 - 6 | 2 (upwind), 6 (WENO5) |
| BC order | 1 - 5 | 1 (upwind), 3 (WENO5) |

---

## Recommandations

### Haute Priorité

1. **Implémenter le système de presets** pour les problèmes courants:
   - Advection (upwind et WENO5)
   - Burgers
   - Diffusion

2. **Ajouter la validation automatique** dans les bindings Python existants

3. **Documenter les coefficients SSP-RK3** et créer un intégrateur dédié

### Moyenne Priorité

4. Créer presets pour problèmes moins courants (level set, Navier-Stokes)

5. Implémenter l'export/import de configurations (JSON/YAML)

6. Ajouter des tests de régression basés sur les presets

### Basse Priorité

7. Interface graphique pour sélection de preset

8. Optimisation automatique de preset (basé sur benchmark)

9. Intégration avec d'autres bibliothèques (PETSc, etc.)

---

## Conclusion

Cette analyse révèle que:

1. **Les magic numbers sont omniprésents** dans les exemples Python Samurai
2. **Les valeurs typiques sont cohérentes** entre exemples similaires
3. **Un système de presets apporterait une valeur significative**:
   - Réduction des erreurs (validation automatique)
   - Meilleure expérience utilisateur
   - Code plus maintenable
   - Science plus reproductible

**Prochaine étape recommandée**: Implémenter le système de presets avec les priorités définies ci-dessus.

---

## Ressources

### Documents Créés

1. `SAMURAI_CONSTANTS_ANALYSIS.md` - Analyse détaillée avec code
2. `CONSTANTS_SUMMARY_TABLE.md` - Tableaux de référence rapide
3. `python/scripts/config_validator.py` - Outil de validation

### Utilisation

```bash
# Valider une configuration
python python/scripts/config_validator.py validate python/examples/advection_2d.py

# Comparer deux configurations
python python/scripts/config_validator.py compare \
  python/examples/advection_2d.py \
  python/examples/burgers_2d.py

# Suggérer une configuration
python python/scripts/config_validator.py suggest advection weno5 default

# Scanner tous les exemples
python python/scripts/config_validator.py scan python/examples/
```

### Contact

Pour toute question ou suggestion concernant cette analyse, consulter les documents détaillés ou ouvrir une issue sur le dépôt GitHub Samurai.

---

**Fin du résumé exécutif**
