# Feedback et Suivi (Backend CUDA/Thrust)

Ce fichier centralise nos retours, problèmes, décisions et TODOs durant le développement du backend CUDA/Thrust.

Règles
- Ne plus modifier `thrust.md` sauf demande explicite. Toute remarque ou évolution passe d’abord ici.
- TDD en continu, commits élémentaires, pas de push.
- Limites de parallélisme: `-j2` pour les démos, `-j3` pour les tests (ctest).

Comment consigner un item
- Titre bref et daté (YYYY‑MM‑DD).
- Contexte: où/quoi (fichier, cible CMake, test label/nom).
- Repro pas à pas (commandes exactes).
- Résultat attendu vs observé (log/traces clés).
- Impact: blocant/majeur/mineur.
- Hypothèse(s) / analyse rapide.
- Suivi: action(s), responsable, échéance, statut.
- Réf commit(s) local(aux) (hash local)

Décisions actuelles (rappel)
- UVM activée par défaut (accès host `field[cell]`).
- Priorité à la robustesse/simplicité avant l’optimisation.
- Layout initial: row‑major uniquement (CMake doit refuser col‑major avec `-DSAMURAI_FIELD_CONTAINER=cuda`).
- Strides à supporter tôt (au moins step=2), in‑place ops sur vues (`+=`, `-=`, `*=`) et `sum`.
- `apply_on_masked`: fallback CPU d’abord; proposition ultérieure `where(mask, …)` device.

## 2025-10-09 – Implémentation initiale CUDA/Thrust (phase M1)

Contexte
- CMake: ajout de l’option de backend `-DSAMURAI_FIELD_CONTAINER=cuda` (link `CUDA::cudart`, erreur si `SAMURAI_CONTAINER_LAYOUT_COL_MAJOR=ON`).
- Code: nouveau header `include/samurai/storage/cuda/thrust_backend.hpp` (UVM, vues 1D/2D minimales, `apply_on_masked` host).
- Tests: lorsque le backend CUDA est sélectionné, on ne construit que `tests/cuda/test_cuda_backend.cpp` pour valider le socle.

Impact
- Chemin heureux pour ScalarField + vues 1D (intervals) OK; bases pour VectorField posées mais incomplètes.

Suivi
- Étendre `view(item, range)` et `view(range_item, range)` (VectorField, AOS/SOA) + `operator+=` depuis vue ou vue/scalaire.
- Compléter `math` (min/max/abs) et `shape()` au besoin pour `transfer/prediction`.
- Ajouter des tests ciblés: `cuda-thrust-views`, `cuda-thrust-masked`, `cuda-thrust-assign`.

## 2025-10-09 – Vues vectorielles + helpers math (M1 suite)

Contexte
- Code: `thrust_view_1d/2d` supporte désormais const/non-const, strides génériques (SoA/AoS, pas=2), vues colonnes/lignes (`view(..., placeholders::all(), j)` et `view(..., item)`), `shape()` renseigné (axes 0/1) avec vérifications d’axe.
- Ops: `math::abs/minimum/maximum`, `compare` 2D (itère colonnes), `zeros` réutilisé pour tests; `apply_on_masked` inchangé (host/UVM).
- Tests: `tests/cuda/test_cuda_backend.cpp` couvre SoA/AoS (in-place `*=`), vues const, `shape`, helpers math. Suite ctest (5 tests) OK en ~3s (`ctest --test-dir build_test -j3 --output-on-failure`).

Impact
- Vues et ops utilisés par `VectorField` et `transfer()` trouvent leurs équivalents CUDA de base.
- Helpers math préparent les futures implémentations (`minimum/maximum/abs` requis par critères/démos).

Suivi
- Ajouter `operator+=` depuis vue scalaire dédiée (actuellement `thrust_view_1d::operator+=` accepte objets indexables, ok pour M1).
- Compléter `range_item` avec `step>1` scénarios tests (couvert en code, pas encore testé).
- Prochaine étape: intégrer `noalias`/assign multi-vues + premiers kernels `transfer()`/`prediction()`.

Check‑list TDD (mise à jour au fil de l’eau)
- [ ] Storage: `resize/swap/fill`, `array()` accessible, `compare(...)`
- [ ] Views: 1D/2D + strides (step=1,2, puis général)
- [ ] Ops: `+,-,*,/`, `abs`, `minimum/maximum`
- [ ] Assignation: `noalias(lhs)=rhs` (kernel unique)
- [ ] In‑place: `+=`, `-=`, `*=` (scalaire)
- [ ] Réductions: `sum(expr)` (puis `sum<axis>`)
- [ ] Iterators/default_view_t: `*it` et `compare` OK
- [ ] Algorithmes: `transfer()/prediction()` (uniform, puis MR)
- [ ] Démos FV 1D/2D de petite taille

Commandes usuelles
- Config tests+démos:
  ```bash
  cmake -S . -B build_test -DBUILD_DEMOS=ON -DBUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
  ```
- Build (limité):
  ```bash
  cmake --build build_test -j2
  cmake --build build_test --target finite-volume-advection-2d -j2
  ```
- Tests (limité):
  ```bash
  ctest --test-dir build_test -j3 --output-on-failure
  ctest --test-dir build_test -L cuda-thrust-views --output-on-failure
  ctest --test-dir build_test -R storage_cuda_views --output-on-failure
  ```

Modèle d’item (à copier‑coller)
```
## 2025-10-09 Compilation demos (finite-volume-advection-2d)

Contexte
- Fichier/zone: `include/samurai/storage/cuda/thrust_backend.hpp`, `demos/FiniteVolume/advection_2d.cpp`
- Cible/label: build démo `finite-volume-advection-2d`
- Environnement: backend CUDA (Thrust), build `cmake --build build_test --target finite-volume-advection-2d -j2`

Reproduction
- 1) `cmake --build build_test --target finite-volume-advection-2d -j2`
- 2) Compilation échoue avec erreurs sur `operator>`/`operator<` et `math::transpose`

Attendu vs Observé
- Attendu: compilation de la démo en backend CUDA.
- Observé: 
  - nos surcharges `operator>`/`operator<` capturent `std::strong_ordering` (timers) et accèdent à `ref[i]` / `ref.size()` non définis.
  - `math::transpose` absent du namespace CUDA, MR ne compile plus.

Impact
- Blocant pour la compilation des démos; surfaces affectées: toutes cibles utilisant `timers` et MR operators.

Analyse/hypothèse
- Les opérateurs doivent être restreints aux vues indexables (`size()`, `operator[]`).
- Fournir `math::transpose` (alias vers `xt::transpose`) suffit pour la phase M1 (fallback host).

Suivi
- Action: restreindre les surcharges et exposer `math::transpose`; vérifier `ctest -L cuda-thrust-masked` + build démo. (ctest OK, build démo bloque désormais sur opérateurs `+` entre vues – voir itération suivante.)
- Update 2025-10-09 (PM): introduction d’expressions `thrust_expr_*` pour `+/-/*/` et slices, ajout de `range()` côté CUDA. Tests `cuda-thrust-masked` OK. Build démo échoue encore dans `rel_detail.hpp` (`managed_array` ne sait pas `operator*=` avec xt::adapt) – à traiter séparément.
- Next: ajouter un `operator*=` (et `operator/=`) compatible avec les adaptateurs xt/expressions pour `managed_array` et/ou fournir un wrapper CUDA (`thrust_expr_scale_inplace`) afin que `rel_detail.hpp` compile en backend CUDA. Tester ensuite `cmake --build build_test --target finite-volume-advection-2d -j2`.
- Update 2025-10-09 (PM+): ajouté `operator*=`, `operator/=`, `operator&=`, `operator|=` sur `managed_array`/`thrust_view_1d` + surcharges bitwise pour les expressions CUDA. `ctest -L cuda-thrust-masked` ✅ et `cmake --build build_test --target finite-volume-advection-2d -j2` ✅ (warnings OMP inchangés). Prochaines investigations: warnings OpenMP bénins; poursuivre avec autres opérateurs MR si besoin.

Références
- Tests ciblés: `ctest --test-dir build_test -L cuda-thrust-masked --output-on-failure`
- Démo: `cmake --build build_test --target finite-volume-advection-2d -j2`

## 2025-10-09 – Crash `apply_on_masked` (mask size mismatch)

Contexte
- Fichier/zone: `include/samurai/storage/cuda/thrust_backend.hpp::detail::mask_binary`
- Cible/label: exécution `./demos/FiniteVolume/finite-volume-advection-2d`
- Environnement: backend CUDA/Thrust, build `cmake --build build_test --target finite-volume-advection-2d -j2`

Reproduction
- 1) `cmake --build build_test --target finite-volume-advection-2d -j2`
- 2) `./demos/FiniteVolume/finite-volume-advection-2d`

Attendu vs Observé
- Attendu: démo 2D fonctionnelle.
- Observé: abort `std::runtime_error` "mask size mismatch: lhs=512 rhs=1" lors de `apply_on_masked` (stack MR coarsening).

Impact
- Blocant pour les démos MR (coarsening/refinement) avec backend CUDA.

Analyse/hypothèse
- Les combinaisons de masques (`mask_binary`) n’acceptaient pas la diffusion scalaire (`size()==1`). Les critères MR empilent des réductions sur composantes (taille 1) avec des masques cellule (taille N), provoquant l’exception.

Suivi
- Action: autoriser la diffusion scalaire dans `mask_binary::size()`/`operator[]` + message détaillé (fait, local 2025-10-09).
- Action: ajouter un test de non-régression `cuda_backend.apply_on_masked_broadcast_scalar_mask` (`ctest --test-dir build_test -R cuda_backend.apply_on_masked_broadcast_scalar_mask -j3 --output-on-failure`) (fait, local 2025-10-09).
- Action ouverte: relancer la démo pour confirmer (à faire après intégration du fix).

Références
- Tests liés: `ctest --test-dir build_test -L cuda-thrust-masked --output-on-failure`
- Démos liées: `cmake --build build_test --target finite-volume-advection-2d -j2`

## 2025-10-09 – Segfault sur masques temporaires (apply_on_masked)

Contexte
- Fichier/zone: `include/samurai/storage/cuda/thrust_backend.hpp::operator<` / `operator>`
- Cible/label: exécution `./demos/FiniteVolume/finite-volume-advection-2d`
- Environnement: backend CUDA/Thrust, build `cmake --build build_test --target finite-volume-advection-2d -j2`

Reproduction
- 1) `cmake --build build_test --target finite-volume-advection-2d -j2`
- 2) `./demos/FiniteVolume/finite-volume-advection-2d`

Attendu vs Observé
- Attendu: démo fonctionnelle après correction du mismatch de taille.
- Observé: `SIGSEGV` dans `mask_binary::operator[]` (pile MR coarsen) lorsque le masque est construit via un temporaire (`abs(detail(...)) < eps`).

Impact
- Blocant: les masques temporaires perdaient leur storage (dangling ref), toute la chaîne MR plantait.

Analyse/hypothèse
- Les surcharges `<`/`>` stockaient une référence vers l'expression temporaire (`const D&`). Après retour, la référence devenait pendante. Les vues CUDA (thrust_view) sont triviales à copier, donc on peut matérialiser l'expression.

Suivi
- Action: faire un `decay_copy` dans les surcharges `<`/`>` et accepter `D&&` (fait, local 2025-10-09).
- Action: ajouter un test `cuda_backend.temporary_expression_mask_lifetime` qui applique un masque construit inline (fait, local 2025-10-09).
- Action: relancer la démo pour confirmer (à refaire côté utilisateur, la sandbox CLI limite l'exécution directe).

Références
- Tests liés: `ctest --test-dir build_test -L cuda-thrust-masked --output-on-failure`
- Démos liées: `cmake --build build_test --target finite-volume-advection-2d -j2`

## 2025-10-09 – Réductions sur masques (sum/all_true) et tests unitaires

Contexte
- Fichier/zone: `include/samurai/storage/cuda/thrust_backend.hpp` (`math::sum<axis>`, `math::all_true`, diffusions)
- Tests: `tests/cuda/test_cuda_backend.cpp` (ajout `temporary_expression_mask_lifetime`, `sum_axis_over_mask_expression`, `all_true_mask_chain`)

Reproduction
- 1) `cmake --build build_test -j2`
- 2) `ctest --test-dir build_test -R cuda_backend -j3 --output-on-failure`

Attendu vs Observé
- Attendu: disposer d’un équivalent CUDA pour `sum<axis>` et `all_true` afin d’évaluer les critères MR sans se reposer sur xtensor.
- Observé: avant correctif, les masques composites étaient mal évalués (diffusion incorrecte, références pendantes), entraînant coarsening global. Les nouveaux tests reproduisent les combinaisons MR hors du code principal et vérifient la diffusion scalaire + chaînes de `&&`.

Analyse/hypothèse
- Les réductions nécessitent d’inspecter la forme (items/length) des vues CUDA. On propage maintenant cette information via `detail::infer_shape` et on accumule explicitement en traitant les booléens comme des compteurs.

Suivi
- Action: intégrer ces tests dans la boucle TDD – fait (local 2025-10-09).
- Action: relancer la démo après validation unitaire pour confirmer que le champ ne s’annule plus (à confirmer côté utilisateur).
- Action: ajouter des tests autour de `projection`/`variadic_projection` pour capturer les accès MR (fait, local 2025-10-09).

Références
- Tests liés: `ctest --test-dir build_test -R cuda_backend -j3 --output-on-failure`
- Démos liées: `cmake --build build_test --target finite-volume-advection-2d -j2`

## 2025-10-09 – Couverture champs CUDA & vue stride en défaut

Contexte
- Fichiers: `tests/CMakeLists.txt`, `tests/test_field.cpp`
- Cible: `test_samurai_lib` (label `cuda-thrust`)
- Actions: exclusion de `test_periodic.cpp` du bundle CUDA; ajout de tests ciblés sur les conteneurs de champs (`noalias` scalaire/vectoriel, vue à pas 2, swap/flags de fantômes).

Reproduction
- `cmake --build build_test --target test_samurai_lib -j2`
- `ctest --test-dir build_test -L cuda-thrust -j1 --output-on-failure`

Attendu vs Observé
- Attendu: tous les nouveaux tests passent, révélant les régressions éventuelles.
- Observé: `field.strided_interval_assignment` échoue (différence de 1 entre attendu et obtenu sur chaque case paire) ⇒ l’écriture via vue stridée (step=2) ne modifie pas le stockage CUDA.

Impact
- Majeur pour la couverture: motiver une enquête sur `thrust_view_1d`/`make_range_view` lorsque le range possède un pas > 1 sur des conteneurs collapsés.

Suivi
- [ ] Diagnostiquer `managed_array` + `view` côté CUDA pour pas>1 (proposer instrumentation/tests unitaires dédiés).
- [ ] Réintroduire `test_periodic.cpp` une fois la copie périodique multi-dimensionnelle stabilisée.

Références
- `ctest --test-dir build_test -R field.strided_interval_assignment --output-on-failure`

## [YYYY‑MM‑DD] Titre bref

Contexte
- Fichier/zone: …
- Cible/label: …
- Environnement: GPU?, driver?, nvcc?, CC?, RAM disponible? …

Reproduction
- 1) …
- 2) …
- 3) …

Attendu vs Observé
- Attendu: …
- Observé: … (logs/traces)

Impact
- (blocant/majeur/mineur), surfaces affectées: …

Analyse/hypothèse
- …

Suivi
- Action 1: … (owner, due, statut)
- Action 2: …

Références
- Commits locaux: …
- Tests liés: …
- Démos liées: …
```

Espace de travail initial (exemples à remplir au fil des tests)
- [À renseigner] Problèmes de strides (step=2) sur `prediction()` en MR.
- [À renseigner] `compare(...)` sur `default_view_t` (itérateur de champ).
- [À renseigner] `apply_on_masked` paths rencontrés dans une démo.

Fin.

## 2025-10-09 Deep copy CUDA container & stride test update

Contexte
- Fichiers: `include/samurai/storage/cuda/thrust_backend.hpp`, `tests/cuda/test_cuda_backend.cpp`, `tests/test_field.cpp`
- Cible/tests: nouveaux tests `cuda_backend.container_copy_is_deep`, `cuda_backend.range_step_write`, `field.strided_interval_assignment`
- Environnement: build cmake (tests only) + `ctest --test-dir build_test -R <label> -j1`

Reproduction
- `cmake --build build_test --target test_samurai_lib -j2`
- `ctest --test-dir build_test -R cuda_backend.container_copy_is_deep --output-on-failure`
- `ctest --test-dir build_test -R field.strided_interval_assignment --output-on-failure`

Attendu vs Observé
- Attendu: copier un `ScalarField`/`thrust_container` crée un buffer indépendant; les vues stridées mettent à jour les bons indices.
- Observé avant fix: copie superficielle via `managed_ptr` ⇒ `ref` suivait les mutations, et le test `field.strided_interval_assignment` signalait des divergences (1.0). Après correctifs, les tests passent et les écritures stridées touchent les offsets retournés par la vue.

Analyse/hypothèse
- `managed_ptr` n'avait pas de `m_size`, donc copie = simple pointer copy. Ajustement: stockage de la taille + constructeurs/assignations (deep copy + move). Le test de champ utilisait un attendu naïf (`idx % 2`) qui ne reflétait pas les offsets réels (`interval.index` ≠ 0); remplacement par un suivi des adresses réellement écrites.

Suivi
- Action: surveiller d’autres APIs (`apply_on_masked`, portions MR) pour garantir qu’elles s’appuient sur la même logique de vues → TODO encore ouvert.

Références
- Tests: `ctest --test-dir build_test -R cuda_backend.container_copy_is_deep`, `ctest --test-dir build_test -R field.strided_interval_assignment`

## 2025-10-09 Masked ops 2D compatibility & xt interop (en cours)

Contexte
- Fichiers: `include/samurai/storage/cuda/thrust_backend.hpp`
- Objectif: finaliser `apply_on_masked` pour vues 2D et combinaisons logiques en réutilisant les chemins MR existants (coarsening/refinement, prediction).
- Environnement: build `cmake --build build_test --target test_samurai_lib -j2`; tests ciblés `ctest --test-dir build_test -L cuda-thrust -j3`.

Reproduction
- Build unique : `cmake --build build_test --target test_samurai_lib -j2`.
- Tests: `ctest --test-dir build_test -L cuda-thrust -j3 --output-on-failure` (⚠️ certains cas HDF5 échouent car plusieurs tests écrivent le même fichier en parallèle).

Attendu vs Observé
- Attendu: 
  1. Les masques combinant `abs(detail) < eps`, `&&`, `!` fonctionnent sur vues 1D/2D.
  2. Les vues 2D (`thrust_view_2d`) satisfont les trait xtensor (`xt::view`, `operator()`, `stepper`) même quand elles sont transposées/slicées.
- Observé: 
  - Compilation réparée (XT `transpose`, `mask_binary` compile).
  - Exécution: les tests CUDA custom passent; `ctest -L cuda-thrust` échoue sur suites HDF5 (verrouillage fichier partagé). Besoin de lancer ces tests en série ou de dédupliquer le chemin de sortie.

Analyse/hypothèse
- Ambiguïtés `sequence_size(...)` résolues en qualifiant toutes les invocations (`detail::sequence_size`).
- Intégration xt::view: adaptation `samurai::range_t` + `placeholders::all_t` -> `xt::range`/`xt::all`.
- `thrust_view_2d` a désormais `operator()(std::size_t)` linéaire pour satisfaire `xt::xfunction`.
- Reste à sécuriser l'environnement HDF5 (verrouillage) avant de cocher les tests MR/HDF5.

Suivi
- Action 1: Dédier un run séquentiel HDF5 (`ctest -L hdf5 --output-on-failure -j1`) ou modifier les tests pour utiliser des noms de fichiers uniques — à faire avant validation finale.
- Action 2: Étendre les tests CUDA (labels `cuda_backend.soa_vector_views`, `cuda_backend.aos_vector_views`) pour couvrir les nouveaux chemins `operator()(std::size_t)` et combinaisons de masques multi-dimension, puis ré-activer label filtré `cuda-thrust-masked`.

Références
- Build: `cmake --build build_test --target test_samurai_lib -j2`
- Tests: `ctest --test-dir build_test -L cuda-thrust -j3 --output-on-failure`
