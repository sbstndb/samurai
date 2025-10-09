# Samurai – Backend CUDA (Thrust)

Ce document décrit comment doter Samurai d’un backend de tableaux GPU fondé sur Thrust (CUDA), sans dépendre de xtensor‑cuda et sans modifier l’API haut‑niveau de Samurai (champs, opérateurs, itérations d’intervalles).

Objectifs clés
- Garder inchangés les modules utilisateurs (Field/VectorField, `for_each_interval`, opérateurs numériques, BC/MPI).
- Remplacer uniquement la couche “storage” aujourd’hui fournie par `storage/xtensor` ou `storage/eigen`.
- Offrir des opérations élémentaires sur GPU (assignations, opérations point‑à‑point, réductions simples) et respecter les layouts AOS/SoA actuels.
- Activation par CMake via `-DSAMURAI_FIELD_CONTAINER=cuda` (nouveau choix à côté de `xtensor`/`eigen3`).

Décisions (actées)
- UVM (mémoire unifiée CUDA) par défaut pour les champs afin de préserver l’accès host `field[cell]` et simplifier le portage.
- Priorité à un chemin correct et simple avant toute optimisation de performance.
- Layout initial: row‑major uniquement; la variante col‑major sera traitée plus tard. Le CMake émettra une erreur si col‑major est demandé avec `cuda`.

Hors‑périmètre initial
- Pas d’intégration xtensor‑cuda.
- Pas de refonte des algorithmes Samurai: on s’aligne sur le contrat minimal attendu par `include/samurai/storage/containers_config.hpp` et utilisé par `field.hpp`/`field_expression.hpp`.

—

1) Architecture proposée (Thrust)
- Conteneur dynamique device: `thrust_container<T, size, SOA, can_collapse>` stocke un buffer linéaire `thrust::device_vector<T>`.
- Vues stridées légères:
  - Scalaire (size==1 && can_collapse): `device_view_1d<T>` = {ptr, n, stride}.
  - Vectoriel (size>1): `device_view_2d<T>` sur la dimension “items” et la dimension “cells”, avec mapping AOS/SoA selon le layout Samurai existant.
- Expressions “paresseuses” (lazy):
  - `expr_unary<Op, V>`, `expr_binary<Op, L, R>`.
  - Opérateurs `+, -, *, /` et fonctions `abs, minimum, maximum` construisent des expressions.
  - L’assignation `noalias(view) = expr;` lance un unique kernel via `thrust::transform` (ou `for_each`/`transform` zipé), sans temporaires.
- Réductions:
  - `sum(expr)` via `thrust::transform_reduce`.
  - Éventuellement `sum<axis>` en 2e étape pour agréger par item/cell.

Pourquoi Thrust ?
- Kernels prêts pour transform/reduce/zip, dépendance naturelle CUDA, API header‑only côté utilisateur, mise en œuvre rapide.
- Permet de couvrir vite le set d’opérations réellement utilisées par Samurai.

—

2) Contrat minimal à respecter (couche storage)
Le code Samurai n’utilise pas directement xtensor/Eigen; il passe par des alias et fonctions en ADL définis dans `include/samurai/storage/*`. Pour ajouter CUDA/Thrust, on fournit des équivalents:

- Types aliasés par `containers_config.hpp`:
  - `field_data_storage_t<value_t, size, SOA, can_collapse>` (dynamique).
  - `local_field_data_t<value_t, size, SOA, can_collapse>` (petit tableau statique par thread; peut rester `std::array` host au début).
  - `Array<T, size>` et `CollapsArray<T, size, can_collapse>` restent côté host (utilisés pour petites tailles fixes, opérateurs locaux).

- Fonctions utilitaires (ADL) à fournir pour le backend CUDA:
  - `view(container, range)`; `view(container, item, range)`; `view(container, range_items, range)`.
  - `eval(x)` (identité ou matérialisation si besoin).
  - `noalias(x)` → proxy d’assignation.
  - `shape(x[, axis])` (au minimum taille de la dimension “cells”).
  - `zeros_like(x)`, `zeros<T>(n)`.
  - Espace `samurai::math`: `abs(x)`, `sum(x)` et plus tard `sum<axis>(x)`, `minimum(a,b)`, `maximum(a,b)`.
  - Comparateurs fréquemment utilisés: `operator>(view, double)`, `operator<(view, double)` (peu utilisés sur champs, surtout dans chiffres de structure).

Notes d’intégration avec `field_expression.hpp`
- Samurai détourne la création d’expressions xt via la spécialisation `select_xfunction_expression` vers `samurai::field_function`.
- Dans `field_function::evaluate(...)`, on appelle `eval(m_f(args...))` où `m_f` est un foncteur (souvent xt) appliqué aux vues des backends.
- Pour que cela fonctionne avec Thrust, nos vues doivent offrir les opérateurs/fonctions qui, lorsqu’ils sont combinés par ces foncteurs, renvoient nos propres types d’expressions (GPU). En pratique, on définit `operator+/-/*///` etc. pour nos vues afin que `m_f` construise nos `expr_*` (pas des tableaux xt).

—

3) Layout mémoire et indexation
- Hypothèses actuelles Samurai (voir `storage/utils.hpp` et `layout_config.hpp`):
  - Scalaire collapsable (`size==1 && can_collapse`): tampon 1D de taille `ncells`.
  - Vectoriel (`size>1`): disposition AOS ou SoA dépend de `SAMURAI_DEFAULT_LAYOUT` et `SOA`.
    - Filtrage déjà implémenté via `detail::static_size_first_v<size, SOA, can_collapse, layout>`.

- Mapping linéaire proposé dans le buffer device (taille logique: items × cells):
  - SoA (items en dimension majeure): index = `item * ncells + cell` (contigu en cell pour chaque item).
  - AoS (cells majeurs): index = `cell * size + item` (contigu en items pour chaque cell).

- Vues et `range.step`:
  - Étape 1: support `step == 1` (contigu). Les intervalles Samurai s’alignent déjà sur des blocs contigus dans la plupart des cas.
  - Étape 2: ajout de strides généraux (itérateur indexé: `counting_iterator` + functor d’adressage, ou “strided transform iterator”).

—

4) Interfaces (esquisses)

4.1 Conteneur
```cpp
template<class T, std::size_t Size, bool SOA, bool CanCollapse>
struct thrust_container {
  using value_type = T;
  using size_type  = std::size_t;
  static constexpr auto static_layout = SAMURAI_DEFAULT_LAYOUT;

  thrust::device_vector<T> data_;   // linéaire
  size_type cells_ = 0;             // nb cells dynamiques

  void resize(size_type ncells) {
    cells_ = ncells;
    if constexpr ((Size == 1) && CanCollapse) {
      data_.resize(cells_);
    } else {
      data_.resize(cells_ * Size);
    }
  }

  // accès brut (pour view)
  T* ptr() { return thrust::raw_pointer_cast(data_.data()); }
  const T* ptr() const { return thrust::raw_pointer_cast(data_.data()); }
};
```

4.2 Vues (scalaire et vectoriel)
```cpp
struct range_ll { long long start, end, step = 1; };

template<class T>
struct device_view_1d {
  T* base;
  std::size_t n;
  std::ptrdiff_t stride; // pour step>1 (étape 2)
};

template<class T>
struct device_view_2d { // items x cells, mapping AOS/SoA
  T* base;
  std::size_t items;  // Size
  std::size_t cells;  // ncells sélectionnés par la vue
  bool soa;           // selon layout
  // helpers: index(item, cell)
};

// Overloads view(...)
auto view(thrust_container<T,1,SOA,CanCollapse>& c, range_ll r) -> device_view_1d<T>;
auto view(thrust_container<T,N,SOA,CanCollapse>& c, range_ll r) -> device_view_2d<T>;
auto view(thrust_container<T,N,SOA,CanCollapse>& c, std::size_t item, range_ll r) -> device_view_1d<T>;
```

4.3 Expressions et opérateurs
```cpp
template<class Op, class L, class R>
struct expr_binary { L l; R r; /* shape info */ };

template<class Op, class V>
struct expr_unary { V v; };

// opérateurs: device_view_*  op  device_view_*  → expr_binary<...>
// opérateurs: device_view_*  op  scalaire       → expr_binary<...>
// fonctions: abs, minimum, maximum → expr_unary/binary

// assignation: noalias(dst_view) = expr;
struct noalias_proxy {
  template<class Expr>
  void operator=(const Expr& e) const { launch_transform(dst, e); }
};

template<class View> auto noalias(View v) { return noalias_proxy{v}; }
```

4.4 Réductions
```cpp
template<class Expr>
auto sum(const Expr& e) { return transform_reduce(e, plus<T>{}, T{0}); }

// Étape 2: sum<axis> pour device_view_2d → reduce par item ou par cell.
```

4.5 Utilitaires requis
```cpp
template<class D> auto eval(const D& x) { return x; } // lazy
template<class D> auto shape(const D& x) -> std::array<std::size_t,2>; // selon vue/expr
template<class D> auto zeros_like(const D& x); // buffer temporaire si nécessaire (à limiter)
template<class T> auto zeros(std::size_t n);   // device_vector<T>(n, T{0})
```

—

5) Chemins de code Samurai impactés
- `include/samurai/storage/containers_config.hpp`:
  - Ajouter un bloc `#elif defined(SAMURAI_FIELD_CONTAINER_CUDA_THRUST)` qui `#include "cuda/thrust_backend.hpp"`
  - Fournir `field_data_storage_t`, `local_field_data_t`, `default_view_t` si nécessaire.

- Nouveaux fichiers backend:
  - `include/samurai/storage/cuda/thrust_backend.hpp` (point d’entrée, re-export des vues/ops/math)
  - `include/samurai/storage/cuda/thrust_views.hpp`
  - `include/samurai/storage/cuda/thrust_math.hpp`
  - Optionnel: `thrust_reduce.hpp`, `thrust_detail.hpp` pour itérateurs stridés.

—

6) Intégration CMake (proposée)
- Étendre la liste:
  - `set(FIELD_CONTAINER_LIST "xtensor" "eigen3" "cuda")`
- Si `SAMURAI_FIELD_CONTAINER MATCHES cuda`:
  - `enable_language(CUDA)`
  - `find_package(CUDAToolkit REQUIRED)` (ou s’appuyer sur le toolchain local CUDA)
  - `target_compile_definitions(samurai INTERFACE SAMURAI_FIELD_CONTAINER_CUDA_THRUST)`
  - `target_link_libraries(samurai INTERFACE CUDA::cudart)` (Thrust est header‑only mais dépend du runtime)
  - Passer l’archi GPU si nécessaire: `-DCMAKE_CUDA_ARCHITECTURES=70;80;90` (à ajuster).
  - Options NVCC utiles: `--expt-relaxed-constexpr`, `--use_fast_math` (optionnel), `-lineinfo` (debug)
  - UVM: activer la mémoire unifiée (par ex. compilation device avec `-Xcompiler -DUSE_UVM` et usages `cudaMallocManaged` si l’on implémente une allocation manuelle; avec `thrust::device_vector`, privilégier un wrapper managé si besoin).
  - Layout: si `SAMURAI_CONTAINER_LAYOUT_COL_MAJOR` est ON, générer une erreur de configuration pour le backend `cuda`.

Build exemple
```bash
cmake -S . -B build \
  -DSAMURAI_FIELD_CONTAINER=cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build -j
```

—

7) Limitations et stratégies
- Strides (`range.step`): support indispensables dès M1 (au moins `step=2` pour prediction/transfer), puis généralisation.
- Les `local_field_data_t` (structures statiques) restent host au début. Si des chemins critiques exigent device, on pourra proposer des versions device‐friendly (peu probable).
- Les comparaisons et masques utilisés sporadiquement (`apply_on_masked`) pourront être implémentés via `thrust::for_each` avec prédicat device.
- Debug NaN (`SAMURAI_CHECK_NAN`): possible via `thrust::any_of` + prédicat `isnan` sur une vue.

—

8) Validation fonctionnelle
- Cas cibles minimaux (POC):
  1. Scalaire: `noalias(phi(level,i)) = a * phi(level,i) + b * rhs(level,i);`
  2. `abs`, `minimum`, `maximum` dans un flux type upwind (juste pour 1D d’abord).
  3. `sum(expr)` pour une mesure globale simple (norme L1).

- Jeux de tests: réutiliser `tests/test_portion.cpp`, `tests/test_box.cpp` et une démo FV 1D réduite, reconstruite avec `-DSAMURAI_FIELD_CONTAINER=cuda`.

—

9) Performance (lignes directrices)
- Fusion d’expressions: une assignation = un seul kernel (`transform`) pour minimiser les lectures/écritures globales.
- Éviter `eval()` matérialisant des temporaires device; préférer des expressions paresseuses jusqu’à l’assignation.
- Strides: introduire des itérateurs indexés pour couvrir `range.step>1` sans copies.
- Lancer des kernels avec `thrust::device` policy explicite.

—

10) Feuille de route (ajustée)
- M1 (correctness d’abord — UVM, row‑major)
  - [ ] CMake: backend `cuda` + CUDA language; erreur si col‑major.
  - [ ] `thrust_container` (resize/swap/fill), `array()` et `compare` adapté.
  - [ ] Vues 1D/2D + strides essentiels (`step=1` et `step=2`, puis général) et mapping AOS/SoA (row‑major seulement).
  - [ ] Ops de base `+,-,*,/`, `abs`, `minimum/maximum` et assignation `noalias(lhs)=rhs` (kernel unique).
  - [ ] In‑place `+=`, `-=`, `*=` (scalaire) sur vues.
  - [ ] Itérateurs/default_view_t utilisables côté host (UVM) et `operator*` stable pour les tests.
  - [ ] Réduction `sum(expr)`.
  - [ ] Tests ciblés (labels), demos FV 1D/2D petites; limites -j (2 démos, 3 tests).

- M2 (couverture algos)
  - [ ] `transfer()`/`prediction()` (uniform + MR) — dépend de strides et in‑place ops.
  - [ ] `sum<axis>`; `apply_on_masked` fallback CPU + proposition `where(mask,…)` device.
  - [ ] Checks NaN optionnels device.

- M3 (vectoriel complet + perfs de base)
  - [ ] AOS/SoA complet pour `size>1`; validations étendues.
  - [ ] Bench micro et premières optimisations locales (fusion d’expressions).
  - [ ] Doc finale.

—

11) Exemple d’utilisation (code Samurai, inchangé)
```cpp
auto& phi  = samurai::make_scalar_field<double>("phi", mesh);
auto& rhs  = samurai::make_scalar_field<double>("rhs", mesh);
double a=0.7, b=0.3;

samurai::for_each_interval(mesh, [&](std::size_t level, const auto& i, const auto& index){
  noalias(phi(level,i,index)) = a * phi(level,i,index) + b * rhs(level,i,index);
});
```
Avec le backend Thrust, l’assignation ci‑dessus sera réalisée par un unique kernel `thrust::transform` par intervalle.

—

12) Q/R de conception
- Pourquoi ne pas réutiliser les ET d’xtensor côté device ?
  - On veut éviter une dépendance xtensor‑cuda, rester léger et contrôler la matérialisation. Les ET simples Thrust suffisent.
- Peut‑on mélanger CPU/GPU ?
  - Oui: via la sélection du backend au moment de la compilation; la logique fantômes/MPI/IO est inchangée.
- Et Kokkos/HIP/SYCL ?
  - Thrust d’abord (livrable rapide). Une abstraction ultérieure pourrait réutiliser le même contrat pour Kokkos si besoin.

—

Appendice A — Signatures minimales (résumé)
```cpp
// containers_config.hpp (nouveau bloc)
#elif defined(SAMURAI_FIELD_CONTAINER_CUDA_THRUST)
  using field_data_storage_t = thrust_container<value_t, size, SOA, can_collapse>;
  using local_field_data_t   = /* std::array-based static */;
  template<class T> using default_view_t = /* view type par défaut */;

// thrust_backend.hpp
namespace samurai {
  template<class T, std::size_t S, bool SOA, bool C>
  struct thrust_container { /* cf. §4.1 */ };

  // view overloads (cf. §4.2) + opérateurs, math, noalias, eval, shape, zeros_like, zeros
}
```

—

Choses à décider (feedback souhaité)
- Architectures CUDA cibles par défaut (ex.: 70/80/90).
- Priorité des opérateurs physiques (flux/limiters) à couvrir en premier.
- Besoin immédiat de `range.step>1` dans vos cas d’usage (sinon on phase en 2e étape).

—

13) Compatibilité ET, Views et Lazy Evaluation (FAQ technique)

- Pipeline ET Samurai (réalité du code)
  - Les combinaisons d’expressions sont captées par `samurai::field_function` (voir `include/samurai/field_expression.hpp`).
  - Au moment de remplir un champ, Samurai fait pour chaque intervalle: `noalias((*this)(level, i, index)) = e.derived_cast()(level, i, index);` (voir `include/samurai/field.hpp:499` et `include/samurai/field.hpp:1130`).
  - Notre backend doit donc fournir: des vues adaptées en RHS/LHS, des opérateurs qui construisent nos propres expressions, et une assignation `noalias` qui déclenche un kernel unique.

- Couverture des views Samurai
  - Requises par `ScalarField`/`VectorField`:
    - `view(container, range)`
    - `view(container, item, range)`
    - `view(container, {item_s,item_e}, range)`
    - `view(container, index)` (accès vectoriel par cellule)
  - Le backend Thrust fournira ces surcharges, en respectant le mapping AOS/SoA décrit au §3 via un indexeur `(item, cell) → offset`.

- Sélection des opérateurs (ET côté GPU)
  - On définit `operator+,-,*,/` pour nos types de vue et d’expression (`device_view_*`, `expr_unary`, `expr_binary`). Par ADL, ces opérateurs sont choisis quand `field_function` appelle `m_f(args...)`.
  - Les fonctions `math::abs`, `math::minimum`, `math::maximum`, `sum` sont également fournies pour nos types, afin d’éviter toute retombée vers xt/eigen.

- Lazy evaluation
  - `eval(expr)` retourne l’expression (identité). Pas de buffer temporaire device par défaut.
  - L’assignation `noalias(LHS)=RHS` déclenche un unique `thrust::transform` avec des iterators zippés qui lisent les sources et écrivent LHS.
  - Les réductions `sum(RHS)` utilisent `thrust::transform_reduce`.

- Masques et apply_on_masked
  - Aujourd’hui, `apply_on_masked` itère côté CPU et appelle un lambda host pour écrire `out(imask) = ...` (cf. implémentations xtensor/Eigen).
  - Stratégie proposée:
    1) Phase 1 (fonctionnelle): conserver une implémentation host pour ces chemins — l’ensemble du reste reste GPU.
    2) Phase 2 (GPU): introduire une primitive `where(mask, expr_true, expr_false)` évaluée dans un seul `transform`, ou une variante `apply_on_masked_device(mask, lambda)` avec lambda `__host__ __device__` et itération via `thrust::counting_iterator` + `thrust::for_each_n`.
  - Les comparateurs `>`, `<` pour nos vues renvoient un type d’expression booléenne (mask) consommable par `where` ou par un `transform` conditionnel.

- `range.step` et strides
  - Étape 1: on supporte `step==1` (contigu). Couvre la majorité des boucles par intervalle.
  - Étape 2: itérateurs stridés (iterator d’index + foncteur d’adressage) pour `step>1` sans copies.

- `sum<axis>` (réduction par composante)
  - Pour `device_view_2d`, on fournira `sum<axis=0>` (par items) ou `sum<axis=1>` (par cells) via des réductions segmentées (scan + gather) ou un kernel dédié.

- Vérifications rapides
  - `field_expression.hpp` n’impose aucun type concret, seulement que nos types d’expressions implémentent `operator()(level, interval, index...)` retournant un objet évaluable par `noalias`.
  - `field.hpp` n’utilise que `view(...)`, `noalias(...)`, et les math/ops listés — tout est couvert par notre contrat backend.

- Risques et mitigations
  - Usage ponctuel d’APIs xt/Eigen statiques (ex.: `xtensor_fixed`) pour de petits calculs locaux: cohabitent avec le backend champs GPU (ces calculs restent CPU, sans blocage).
  - `apply_on_masked` gourmand: migrer vers `where(...)` pour éviter les aller‑retour host/device.
  - Strides non triviaux: phasage explicite (étape 2) + tests dédiés.

—

14) Autres usages repérés dans le code et implications

- Écriture par cellule côté host: `field[cell] = ...`
  - Très courant (init, BC, assemblage FV). Aujourd’hui, `operator[](cell)` fait `m_storage.data()[cell.index]`.
  - Implication backend: le stockage doit être accessible en écriture côté host. Décision: mémoire unifiée CUDA (UVM) par défaut pour que `T&` renvoyé par `operator[]` reste valide et modifiable côté CPU.
  - Alternative (plus complexe): double tampon host/device + synchronisation (dirty flags). À considérer ensuite si la UVM n’est pas souhaitée.

- Accès direct au conteneur: `array()`
  - Ex.: `std::swap(u1.array(), u2.array())` (OK si `container_t` supporte `swap`).
  - Ex. problématique: `detail.array() *= inv_max_fields_xt;` dans `mr/rel_detail.hpp` (multiplie par un `xt::adapt(std::array<...>)`).
    - Implication: ce chemin mélange directement un conteneur de champ et une expression xt. Pour CUDA, prévoir une variante backend-friendly:
      - Implémenter une version CUDA de `compute_relative_detail` qui fait un `transform` par composante via `view(detail, item, range)` et des scalaires `inv_max_fields[item]`.
      - Ou fournir `operator*=(small_host_vector)` sur `container_t` qui lance un kernel de scaling par item (diffuseur). À décider selon préférence.

- Opérations en place sur vues: `+=`, `-=`, `*=`
  - Observé dans `reconstruction.hpp` (ex.: `view(dst, ... ) += view(src, ...) / cst;`) et dans des démos (`field(level, i, j) *= nu;`).
  - Implication: nos types de vues doivent définir `operator+=`, `operator-=`, `operator*=(scalaire)` qui déclenchent un `thrust::transform` fusionnant LHS et RHS.

- Itérateurs de champ: `Field_iterator` et `default_view_t`
  - `*it` retourne `default_view_t<typename Field::data_type>` (voir `field.hpp`). Les tests comparent `*it` à une `samurai::Array{...}` via `compare`.
  - Implication: définir `default_view_t` pour le backend CUDA comme un type de vue “contigu” (host‑accessible si UVM), et fournir `compare(view, static_array)` (device/host) — soit via `thrust::equal`, soit en copiant sur host si nécessaire.

- Placeholders et slicing par range
  - Observé: `view(qs, placeholders::all(), range(...))` et `shape(qs, axis)` (ex.: `numeric/prediction.hpp`).
  - Implication: exposer `namespace samurai::placeholders { inline constexpr all_t all{}; inline constexpr _t _{}; }` et une famille `range(start,end[,step])` retournant `range_t<>` compris par nos `view(...)`.

- `zeros_like`, `shape`, `compare`
  - Plusieurs appels dans `stencil_field.hpp`, tests, prédiction. Implémentations CUDA requises:
    - `zeros_like(view)` → buffer device temporaire (à minimiser) ou expression constante appliquée en assignation.
    - `shape(view)` et `shape(container, axis)` → retourner les tailles logiques (items, cells) de la vue/conteneur.
    - `compare(a,b)` sur container/vues → `thrust::equal` ou copie vers host temporaire.

- `fill(v)` et `SAMURAI_CHECK_NAN`
  - `field.fill(v)` appelle `m_storage.data().fill(v)`; en debug NaN ils font aussi `m_storage.data().fill(nan)`. 
  - Implication: fournir `container_t::fill(value)` pour écrire sur tout le buffer (thrust::fill).

- `transfer()` et contraintes de layout
  - Utilise `view(dst, ...) += view(src, ...)`. Exige `operator+=` sur vues et `view(container, item, range)`.
  - Contient des `static_assert` sur le layout (row-major) pour certains chemins. Proposition: documenter que le backend CUDA supporte initialement seulement le layout row-major.

- Démos spécifiques xtensor (masques, `xt::masked_view`)
  - Présent dans plusieurs démos WENO/LBM; la lib cœur n’en dépend pas.
  - Implication: ces démos ne bénéficieront pas automatiquement du backend CUDA sans adaptation (`where(...)`/kernels dédiés). OK pour une phase ultérieure.
