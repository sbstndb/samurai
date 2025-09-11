# CSIR : Un Prototype d'Algèbre d'Ensembles Haute Performance

## 1. Introduction et Vision

Ce projet est une démonstration autonome et une preuve de concept pour la structure de données **CSIR (Compressed Sparse Interval Row)**. Il a été conçu en réponse aux limitations de performance observées dans les bibliothèques d'algèbre d'ensembles géométriques complexes, comme Samurai.

**Problème :** Les implémentations traditionnelles, bien que puissantes sur le plan expressif (évaluation paresseuse, templates d'expression), souffrent souvent de goulots d'étranglement dus à une mauvaise localité des données. L'utilisation de structures de données récursives ou d'une multitude de petits objets alloués dynamiquement (ex: `std::vector` ou `std::list` par ligne de maillage) est en conflit direct avec le fonctionnement des architectures matérielles modernes.

**Solution (CSIR) :** Le CSIR propose une refonte radicale de la représentation de la mémoire. Au lieu d'une hiérarchie d'objets, un maillage entier (à un niveau de raffinement donné) est stocké dans **quelques grands tableaux plats et contigus**. Cette approche, inspirée du format CSR (Compressed Sparse Row) utilisé pour les matrices creuses, est conçue dès le départ pour une performance maximale sur les CPU et les GPU.

**L'objectif est de passer d'une pensée "orientée objet" à une pensée "orientée données" et "orientée matériel".**

## 2. La Structure de Données CSIR

Le CSIR représente un maillage 2D à un niveau donné via 3 tableaux principaux :

1.  `std::vector<int> y_coords;`
    -   Un tableau trié contenant uniquement les coordonnées `y` qui possèdent des cellules actives.

2.  `std::vector<std::size_t> intervals_ptr;`
    -   Le "pointeur de ligne". `intervals_ptr[i]` donne l'indice de départ dans le tableau `intervals` pour la ligne correspondant à `y_coords[i]`. `intervals_ptr[i+1]` donne l'indice de fin.

3.  `std::vector<Interval> intervals;`
    -   **Le cœur de l'innovation.** Un unique tableau plat contenant TOUS les intervalles `[start, end)` du maillage, stockés consécutivement, ligne par ligne.

#### Exemple Concret

Un maillage en forme de 'T' :

```
y=2 |  #####
y=1 |    #
y=0 |    #
    +----------
      x=0  x=4
```

Sera représenté par :
- `y_coords`: `[0, 1, 2]`
- `intervals_ptr`: `[0, 1, 2, 3]`
- `intervals`: `[{2, 3}, {2, 3}, {0, 5}]`

### Extension à la 3D

Le concept se généralise en une structure "CSIR de CSIRs". Un maillage 3D est une liste de tranches 2D le long de l'axe Z, où chaque tranche est elle-même un objet CSIR 2D complet. Cela préserve la contiguïté des données à chaque niveau de dimension.

## 3. Algorithmes et Opérations Multi-Niveaux

### Algorithmes Ensemblistes

Toutes les opérations (`intersection`, `union_`, `difference`) suivent un schéma en deux étapes, optimisé pour le parallélisme :
1.  **Boucle Principale :** On itère sur les `y_coords` des deux maillages en entrée pour trouver les lignes communes ou uniques.
2.  **Logique 1D :** Pour chaque ligne, on extrait les listes d'intervalles correspondantes (qui sont de petites `slices` de grands tableaux contigus) et on leur applique un algorithme 1D très rapide.

### Opérations Multi-Niveaux

La complexité des maillages adaptatifs (AMR) est gérée par la **projection**. Pour opérer sur deux maillages de niveaux `L_A` et `L_B`, on doit d'abord les projeter sur un niveau de référence commun (généralement le plus fin).

La fonction `project_to_level(source, target_level)` démontre ce principe. Elle prend un maillage CSIR et en crée un nouveau en mettant à l'échelle toutes ses coordonnées (`coord * 2^level_diff`). Cette opération est également une transformation de données en bloc, bien plus efficace qu'une approche cellule par cellule.

## 4. Analyse de Performance : Pourquoi est-ce Rapide ?

C'est la question centrale. Les gains de performance proviennent de l'exploitation directe des mécanismes du matériel moderne.

#### a) Localité du Cache : Le Nerf de la Guerre

Les processeurs sont des milliers de fois plus rapides que la RAM. Le cache est roi. En stockant les intervalles dans un unique tableau contigu, on garantit que lorsque le CPU charge un intervalle, il précharge également les dizaines suivants dans ses caches (L1/L2). Les opérations sur les lignes se font donc majoritairement à la vitesse du cache, éliminant la latence de la RAM qui plombe les approches à base de pointeurs ou de vecteurs dispersés.

#### b) Parallélisme Massif (CPU & GPU)

-   **CPU (Multi-threading) :** Le traitement de chaque ligne `y` est indépendant des autres. La boucle principale est donc "trivialement parallélisable" via une simple directive `#pragma omp parallel for` en C++. On peut s'attendre à une mise à l'échelle quasi-linéaire avec le nombre de cœurs.
-   **GPU (Data-Parallelism) :** La structure CSIR est **nativement compatible avec les GPU**. Les tableaux plats sont parfaits pour un transfert rapide vers la VRAM (`cudaMemcpy`). On peut lancer un kernel où chaque thread GPU traite une ligne `y`. Les accès mémoires contigus des threads sont "coalescés" par le GPU, ce qui est la condition sine qua non pour atteindre des performances élevées.

#### c) Vectorisation (SIMD)

Les algorithmes 1D, qui opèrent sur de petites listes d'intervalles, sont des candidats idéaux pour la vectorisation. Les instructions AVX/AVX512 des CPU modernes peuvent traiter 4 ou 8 intervalles simultanément, offrant une accélération supplémentaire significative.

#### d) Gestion de la Mémoire

L'approche CSIR remplace des millions de petites allocations mémoires (lentes et sources de fragmentation) par **quelques allocations de grands blocs**. C'est une stratégie de gestion de la mémoire bien plus saine et performante.

## 5. Instructions de la Démonstration

Ce projet est autonome et ne nécessite pas Samurai.

### Compilation

```bash
mkdir build
cd build
cmake ..
make
```

### Exécution

```bash
./demo
```

### Scénarios de Test

L'exécutable `demo` enchaîne 3 scénarios pour prouver la robustesse du concept :
1.  **Intersection Multi-Niveaux :** Un carré au niveau 4 est projeté au niveau 5 et intersecté avec un autre carré. Valide la projection et l'intersection.
2.  **Union de Formes Complexes :** Deux barres (verticale et horizontale) sont unies pour former une croix. Valide l'union sur des formes non-convexes.
3.  **Différence pour Créer un Anneau :** Un petit carré est soustrait d'un grand carré pour former un anneau. Valide la différence et la capacité à gérer des maillages avec des "trous".

## 6. Prochaines Étapes et Vision à Long Terme

Ce prototype est une fondation. Les prochaines étapes seraient :

1.  **Benchmark Rigoureux :** Mesurer quantitativement les gains de performance par rapport à l'implémentation de référence de Samurai.
2.  **Implémentation 3D Complète :** Développer et tester la structure "CSIR-de-CSIRs" pour la 3D.
3.  **Intégration dans Samurai :** Planifier une refactorisation majeure pour remplacer le cœur de la représentation des maillages par le CSIR. Cela nécessiterait d'adapter le moteur de templates d'expression pour qu'il génère des évaluations CSIR impatientes et ordonnées.

En conclusion, le CSIR est une approche pragmatique et puissante, consciente du matériel, qui promet de débloquer des niveaux de performance inaccessibles aux designs logiciels traditionnels pour ce type de problème.
