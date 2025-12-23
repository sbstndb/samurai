# Analyse des Branches de sbstndb/sbstndbs - Samurai

**Date d'analyse:** 2025-12-23
**Nombre total de branches analysées:** 86
**Auteur des branches:** sbstndb / sbstndbs

---

## Table des Matières

1. [Vue d'ensemble](#vue-densemble)
2. [MPI/Parallélisation](#1-mpiparallélisation-12-branches)
3. [Benchmark/Performance](#2-benchmarkperformance-10-branches)
4. [Load Balancing](#3-load-balancing-7-branches)
5. [CUDA/GPU](#4-cudagpu-5-branches)
6. [Documentation](#5-documentation-5-branches)
7. [Codex/AI-generated](#6-codexai-generated-10-branches)
8. [CI/Testing](#7-citesting-8-branches)
9. [Refactoring/Amélioration](#8-refactoringamélioration-15-branches)
10. [Features Diverses](#9-features-diverses-23-branches)
11. [Statistiques Globales](#statistiques-globales)
12. [Recommandations](#recommandations-de-fusion)

---

## Vue d'ensemble

### Répartition par statut

| Statut | Nombre | Pourcentage |
|--------|--------|-------------|
| **Fusionnées dans master** | ~12 | 14% |
| **Prêtes pour fusion** | ~15 | 17% |
| **En développement actif** | ~25 | 29% |
| **Expérimentales** | ~20 | 23% |
| **Abandonnées/Obsolètes** | ~14 | 16% |

### Thèmes principaux de développement

1. **Parallélisation MPI** - Optimisation des communications, load balancing
2. **Performance** - Benchmarks, timers, profiling
3. **GPU/CUDA** - Backend Thrust, structures CSIR
4. **AMR** - Adaptive Mesh Refinement unifié
5. **Infrastructure** - CI, tests, documentation

---

## 1. MPI/Parallélisation (12 branches)

### Branches fusionnées
| Branche | Description | PR |
|---------|-------------|-----|
| `add-timers-in-mpi` | Timers MPI détaillés | #381 |

### Branches prioritaires (prêtes pour fusion)

#### `do_not_send_all_meshid_2` ⭐
- **Objectif:** Réduire le volume de communication MPI en n'envoyant que les mesh_id nécessaires
- **Commits:** 24 (avril 2025)
- **Statut:** Fonctionnel, version mature
- **Impact:** Optimisation significative des communications

#### `error_when_lots_ranks`
- **Objectif:** Validation MPI - erreur si min_level trop bas
- **Commits:** 4 (juin 2025)
- **Statut:** Fonctionnel, protection importante

#### `mpirun-complete-ci`
- **Objectif:** Tests MPI complets dans CI
- **Commits:** 11 (août-sept 2025)
- **Statut:** Infrastructure CI MPI la plus complète

### Branches en développement

| Branche | Objectif | Statut |
|---------|----------|--------|
| `better_mpi` | Refactorisation infrastructure MPI | Partiellement fusionné puis révoqué |
| `mpi_without_serialize` | Éviter sérialisation Boost | Expérimental |
| `do_not_send_all_meshid` | Version initiale de _2 | Remplacée |
| `mpi_update_details_multilevel` | Détails multi-niveaux | En cours |

### Branches CI
| Branche | Objectif |
|---------|----------|
| `mpirun-3-for-burgers-ci` | Tests 3 rangs |
| `add_mpi_to_burgers_ci` | Tests MPI Burgers |
| `add_mpirun_3_to_burgers_ci` | Variante tests |
| `print_mpi_info` | Debug info rang |

---

## 2. Benchmark/Performance (10 branches)

### Branches fusionnées
| Branche | Description | PR |
|---------|-------------|-----|
| `correct-mean-timer` | Correction calcul moyenne timers | #334, #381 |

### Évolution des benchmarks
```
benchmark (old) → new_benchmark_before_force_rebase → new_benchmark → new_benchmark_light
                                                                    ↑
                                                         (version optimale)
```

### Branches prioritaires

#### `new_benchmark_light` ⭐
- **Objectif:** Suite de benchmarks allégée et optimisée
- **Commits:** 6 (octobre 2025)
- **Contenu:** Benchmarks FIELD 2D, DoNotOptimize + ClobberMemory
- **Statut:** Production-ready

#### `benchmark_mpi`
- **Objectif:** Benchmarks MPI spécialisés (advection 2D, linear convection)
- **Commits:** 13 (avril 2025)
- **Statut:** Fonctionnel

### Autres branches

| Branche | Objectif | Statut |
|---------|----------|--------|
| `benchmark` | Suite complète (37 commits) | Remplacée par new_benchmark |
| `new_benchmark` | Base pour new_benchmark_light | Parent |
| `bench_serialize` | Benchmark sérialisation | Expérimental |
| `clean_timers` | Fix compilation | Correction mineure |
| `perf-exchange-tags` | MPI_Irecv asynchrone | Optimisation non fusionnée |
| `profiling_tools` | Scripts perf/VTune/MAQAO | Outils de développement |

---

## 3. Load Balancing (7 branches)

### Hiérarchie des branches
```
bench_load_balancing (juillet 2024) - RÉFÉRENCE COMPLÈTE
    ↓ [rebase]
rebase_load_balancing (février 2025)
    ↓ [amélioration + simplification]
improve_rebase_improved_load_balancing (juillet 2025) - OPTIMISÉE
    ↓ [expérimentation flux]
rebase_flux_lb_experimental (juillet 2025) - EXPÉRIMENTAL
```

### Branches principales

#### `bench_load_balancing` (Référence)
- **Commits:** 167
- **Stratégies implémentées:** SFC, Diffusion, Force, Metis, Scotch, Life
- **Fichiers clés:**
  - `load_balancing.hpp` (1902 lignes)
  - `load_balancing_sfc.hpp`, `load_balancing_diffusion.hpp`
  - `sfc.hpp` (Morton & Hilbert)
- **Statut:** Fonctionnel mais complexe

#### `improve_rebase_improved_load_balancing` ⭐ RECOMMANDÉ
- **Commits:** 86 (juillet 2025)
- **Améliorations:** Architecture simplifiée, classe Weight, auto-stop diffusion
- **Statut:** Version optimisée, candidat principal pour fusion

#### `28_08_load_balancing`
- **Objectif:** Row-based snapping pour frontières horizontales droites
- **Statut:** Amélioration fonctionnelle, simple à fusionner

### Snapshots datés
| Branche | Date | Particularité |
|---------|------|---------------|
| `12_07_load_balancing` | 12 juillet 2025 | Demo advection 2D |
| `28_08_load_balancing` | 28 août 2025 | Row-based aggregation |
| `naive_load_balancing` | 29 avril 2025 | Baseline/test |

---

## 4. CUDA/GPU (5 branches)

### Timeline de développement
```
2025-01-07  new_find_soa (optimisation find)
2025-03-11  soa (benchmarks AOS→SoA)
2025-10-09  cuda (backend CUDA base)
2025-10-10  cuda-demo-working (fixes + intégration)
2025-10-10  thrust-2 (wrapper thrust/xtensor)
```

### Branches prioritaires

#### `cuda-demo-working` ⭐ PRIORITÉ HAUTE
- **Objectif:** Backend CUDA/Thrust complet avec fixes et intégration
- **Fichiers clés:**
  - `thrust_backend.hpp` (~5200 lignes)
  - 17 tests unitaires
  - `feedback.md` (journal développement)
- **Fonctionnalités:**
  - Mémoire unifiée (UVM)
  - Expression templates lazy
  - Support AOS/SoA
  - Opérations masquées
  - Intégration avec algorithmes Samurai (update_ghost_mr, projection)
- **Statut:** Production-ready

#### `cuda`
- **Objectif:** Base du backend CUDA
- **Tests:** 8 tests unitaires
- **Documentation:** `thrust.md` (401 lignes)
- **Statut:** Base stable

### Autres branches

| Branche | Objectif | Statut |
|---------|----------|--------|
| `thrust-2` | Wrapper Thrust/xtensor | Expérimental, objectif pas clair |
| `soa` | Benchmarks AOS→SoA | Expérimental |
| `new_find_soa` | Optimisation find() | Abandonné ("Slower than naive") |

---

## 5. Documentation (5 branches)

### Branches prioritaires

#### `docs/agents-conventions` ⭐ PRIORITÉ HAUTE
- **Objectif:** Guide `AGENTS.md` pour LLM (Claude, Cursor, Codex)
- **Contenu:** Conventions commits, build, tests, best practices
- **Commits:** 4 (septembre 2025)
- **Statut:** Standalone, très utile, prêt pour fusion

#### `doc_attempt`
- **Objectif:** Documentation utilisateur structurée
- **Contenu:** ~2500 lignes, 19 pages numérotées
- **Topics:** Mesh, fields, BC, FV operators, MPI, PETSc
- **Statut:** Fonctionnel, bien structuré

#### `docs-html`
- **Objectif:** Documentation Sphinx/RST pour ReadTheDocs
- **Statut:** Partiellement fusionné

### Autres branches

| Branche | Objectif | Statut |
|---------|----------|--------|
| `doc` | Doc technique complète (~6700 lignes) | Mélange doc + code, abandonné |
| `cours_massot` | Load Balancing MPI (travail recherche) | 161 commits, développement long |

---

## 6. Codex/AI-generated (10 branches)

### Branches fusionnées
| Branche | Description | PR |
|---------|-------------|-----|
| `codex/create-helper-for-name-width-computation` | Formatage timers | #381 |
| `codex/rename-std-to-std_dev-in-print-loop` | Éviter conflit namespace | #381 |

### Branches prioritaires (prêtes pour fusion)

#### `codex/explore-make_convection_weno5-implementation`
- **Objectif:** Optimisation WENO5 (éliminer pow())
- **Impact:** Amélioration performance schémas haute précision
- **Risque:** Faible

#### `codex/optimize-performance-in-burgers.cpp`
- **Objectif:** Optimisation RK3 + fix bug sauvegarde
- **Impact:** Réduction recalculs + correction bug
- **Risque:** Faible

### Branches en développement

| Branche | Objectif | Statut |
|---------|----------|--------|
| `codex/add-custom-mpi-all_reduce-for-min/max-levels` | Optimisation réduction MPI | Mélangé avec refactoring FV |
| `codex/analyze-performance-of-add_interval-function` | Optimisation add_interval | En cours |
| `codex/refactor-mesh_base-to-support-arbitrary-process-grid` | Grilles MPI arbitraires | Nécessite tests MPI |

### Branches d'exploration AI

| Branche | Modèle supposé | Contenu |
|---------|----------------|---------|
| `first_try_g25pro` | Gemini 2.5 Pro | Agrégation communications MPI |
| `second_try_opus41` | Claude Opus 4.1 | Analyse MPI + documentation |
| `gemini-csr-representation` | Gemini | Structure CSIR avec CUDA (~30k lignes) |

---

## 7. CI/Testing (8 branches)

### Branches prioritaires

#### `improve_tests` ⭐ PRIORITÉ HAUTE
- **Objectif:** Tests unitaires C++ (1112 lignes)
- **Fichiers:** test_bc, test_box, test_cell, test_field, test_interval...
- **Statut:** Prêt pour merge

#### `tools/mpi-commit-runner` ⭐
- **Objectif:** Script régression MPI multi-commits
- **Commits:** 1 (propre, documenté)
- **Statut:** Production-ready

#### `long_run_script_local`
- **Objectif:** Script Python tests stabilité MPI
- **Fichiers:** `run_matrix.py` (539 lignes), README complet
- **Statut:** Haute valeur pour debug local

### Branches ReFrame

| Branche | Objectif | Statut |
|---------|----------|--------|
| `reframe` | Intégration ReFrame HPC | Prototype fonctionnel |
| `reframe_on_self_runner` | ReFrame sur self-hosted runner | Expérimental, nécessite cleanup |

### Autres

| Branche | Objectif | Statut |
|---------|----------|--------|
| `ci_long_run` | Simplification tests longs | Fonctionnel |
| `reproduce-mpi-command-for-stability` | Stabilité benchmarks | Utile |
| `improve-rebase-relaunch-ci` | **ATTENTION:** Nom trompeur - c'est du load balancing MPI | 218 commits, abandonné |

---

## 8. Refactoring/Amélioration (15 branches)

### Branches fusionnées
| Branche | Description | PR |
|---------|-------------|-----|
| `enforce_025` | Force version xtensor | #310 |
| `new_flux` | Refactor flux/cell-based schemes | #380 |
| `update-tag-subdomain-factorized-for-loop` | Remove boolean template param | #316 |

### Branches prioritaires

#### `remove_warnings` ⭐ PRÊTE POUR FUSION
- **Objectif:** Suppression warnings compilation
- **Date:** Octobre 2025 (la plus récente)
- **Commits:** 16
- **Statut:** Propre, ciblé

#### `all-in-fmt`
- **Objectif:** Migration std::cout → fmt/samurai::io::print
- **Commits:** 37
- **Impact:** ~300 fichiers, refactorisation majeure
- **Statut:** Fonctionnel, améliore compatibilité MPI

### Famille improve_rebase (Load Balancing)

| Branche | Commits | Statut |
|---------|---------|--------|
| `improve_rebase` | 187+ | Actif (juillet 2025) |
| `improve_rebase_debug` | 189+ | Debug |
| `improve-rebase-2` | 267+ | Alternative |
| `rebase_new` | 194+ | Supplanté |

### Branches d'optimisation

| Branche | Objectif | Statut |
|---------|----------|--------|
| `donotreconstruct` | Désactiver reconstruction voisinage | En cours |
| `find_improvement` | Benchmarks recherche | Analyse |
| `find_vector` | Optimisation interval_find | En cours |
| `new_find` | Recherche linéaire | Expérimental |
| `xtensor026` | Support xtensor 0.26 | Forward compatibility |
| `cellarray` | Optimisations CellArray | Partiellement fusionné |

---

## 9. Features Diverses (23 branches)

### Branches fusionnées
| Branche | Description | PR |
|---------|-------------|-----|
| `merged/correct-timers-output` | Formatage timers | #381 |
| `nsave-demos` | Fix sauvegarde demos | #384 |
| `naive_neighbour` | Neighbourhood naïf | #318 |

### Branches prioritaires

#### `28_08_interface` ⭐ LOAD BALANCING
- **Objectif:** Load balancing par diffusion
- **Fichiers:** `load_balancing.hpp`, `load_balancing_diffusion.hpp`
- **Statut:** Fonctionnel, prêt pour fusion

#### `amr` ⭐ AMR UNIFIÉ
- **Objectif:** API AMR unifiée compatible MPI
- **Documentation:** `unify_amr.md` (324 lignes)
- **Statut:** En développement actif

#### `uniform`
- **Objectif:** UniformMesh
- **Statut:** Fonctionnel

### Famille Neighbourhood

| Branche | Commits | Statut |
|---------|---------|--------|
| `better_neighbour_2` | 11 | Fonctionnel, support 3D |
| `reconstruct_neighbour_mesh_locally` | 11 | Optimisation MPI |
| `reconstruct_neighbour_mesh_locally_2` | 18 | Version allégée |
| `naive_neighbour` | 2 | **Fusionné** |
| `temp_neighbour` | 1 | **DO NOT MERGE** |

### Branches R&D

#### `csir_in_samurai`
- **Objectif:** Structure CSIR avec CUDA
- **Commits:** 32
- **Impact:** +34590 lignes
- **Statut:** R&D avancée, ne pas fusionner tel quel

### Autres

| Branche | Objectif | Statut |
|---------|----------|--------|
| `boost_datatype` | MPI datatypes | Expérimental |
| `boost_with_buffer` | Éviter multi-sérialisation | Performance |
| `cmake_petsc_fix` | Fix PETSc | Probablement obsolète |
| `include_petsc` | Includes PETSc | Probablement obsolète |
| `debug_file` | Utilitaires debug | Maintenance |
| `output` | Utilitaires output | Maintenance |
| `seb/bug_loic` | Bug report MPI | Diagnostic |
| `oula` | Fix @gouarin | Probablement fusionné |
| `3_try_extended_halo` | Extension halo | En cours/abandonné |
| `add_include_for_wallhy` | Compatibilité Wallhy | Spécifique projet |
| `retry_pr` | Tests graduation | Abandonné |
| `try_modules` | Modules C++ | Expérimental |

---

## Statistiques Globales

### Par thème

| Thème | Branches | Fusionnées | Prêtes | Actives | Expérimentales |
|-------|----------|------------|--------|---------|----------------|
| MPI | 12 | 1 | 3 | 5 | 3 |
| Benchmark | 10 | 1 | 2 | 4 | 3 |
| Load Balancing | 7 | 0 | 2 | 3 | 2 |
| CUDA/GPU | 5 | 0 | 2 | 1 | 2 |
| Documentation | 5 | 1 | 2 | 1 | 1 |
| AI/Codex | 10 | 2 | 2 | 4 | 2 |
| CI/Testing | 8 | 0 | 2 | 4 | 2 |
| Refactoring | 15 | 3 | 2 | 6 | 4 |
| Features | 23 | 3 | 3 | 8 | 9 |

### Fichiers les plus modifiés

1. `include/samurai/timers.hpp`
2. `include/samurai/algorithm/update.hpp`
3. `include/samurai/mesh.hpp`
4. `include/samurai/bc.hpp`
5. `include/samurai/interface.hpp`

---

## Recommandations de Fusion

### Priorité 1 - Fusion immédiate recommandée

| Branche | Raison |
|---------|--------|
| `remove_warnings` | Récente, propre, corrections ciblées |
| `improve_tests` | Tests unitaires essentiels |
| `docs/agents-conventions` | AGENTS.md standalone, très utile |
| `tools/mpi-commit-runner` | Script propre, haute valeur |

### Priorité 2 - Validation stratégique requise

| Branche | Raison |
|---------|--------|
| `cuda-demo-working` | Backend CUDA complet, bien testé |
| `improve_rebase_improved_load_balancing` | Load balancing optimisé |
| `new_benchmark_light` | Benchmarks production-ready |
| `28_08_load_balancing` | Load balancing simple |

### Priorité 3 - Développement à poursuivre

| Branche | Action recommandée |
|---------|-------------------|
| `amr` | Continuer selon plan documenté |
| `doc_attempt` | Finaliser documentation utilisateur |
| `all-in-fmt` | Tester en staging |

### Priorité 4 - Nettoyage recommandé

| Action | Branches |
|--------|----------|
| Archiver (fusionnées) | `enforce_025`, `new_flux`, `update-tag-subdomain`, `merged/*`, `nsave-demos`, `naive_neighbour` |
| Marquer obsolètes | `cmake_petsc_fix`, `include_petsc`, `rebase_new`, `benchmark` (old) |
| Supprimer | `retry_pr`, `temp_neighbour` (DO NOT MERGE), `new_find_soa` (abandonné) |

### Branches à NE PAS fusionner

| Branche | Raison |
|---------|--------|
| `temp_neighbour` | Marquée "DO NOT MERGE" |
| `csir_in_samurai` | R&D, trop expérimental |
| `improve-rebase-relaunch-ci` | Historique sale, 218 commits |
| `gemini-csr-representation` | Exploration architecturale |

---

## Liens et Dépendances entre Branches

### Famille Load Balancing
```
bench_load_balancing
    ├── rebase_load_balancing
    │   ├── improve_rebase_improved_load_balancing ← RECOMMANDÉ
    │   └── rebase_flux_lb_experimental
    ├── 12_07_load_balancing
    │   └── 28_08_load_balancing
    └── naive_load_balancing
```

### Famille Benchmark
```
benchmark (old)
    └── new_benchmark_before_force_rebase
        └── new_benchmark
            └── new_benchmark_light ← RECOMMANDÉ
```

### Famille CUDA
```
cuda (base)
    └── cuda-demo-working ← RECOMMANDÉ
thrust-2 (parallèle, approche différente)
```

### Famille Neighbourhood
```
naive_neighbour (fusionné)
better_neighbour_2
reconstruct_neighbour_mesh_locally
    └── reconstruct_neighbour_mesh_locally_2
temp_neighbour (DO NOT MERGE)
```

---

## Notes Historiques

### Périodes d'activité intense

1. **Octobre 2025** - GPU/CUDA (3 branches en 2 jours)
2. **Septembre 2025** - CI/Testing, AI/Codex
3. **Juillet 2025** - Load Balancing, AMR
4. **Avril 2025** - MPI optimizations

### PR importantes fusionnées dans master

- #381 - Timers output formatting
- #384 - Save issue in demos
- #380 - Refactor flux/cell-based schemes
- #318 - Naive neighbourhood
- #316 - Remove boolean template parameter
- #310 - Enforce xtensor version

---

*Document généré automatiquement par analyse des branches Git*
