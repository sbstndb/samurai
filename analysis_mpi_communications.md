# Analyse des communications MPI dans Samurai

## Zones identifiées avec communications niveau par niveau

### 1. update_ghost_subdomains (update.hpp:647-718)
- Pour chaque niveau, envoi/réception avec tous les voisins MPI
- Pattern : isend non-bloquant puis recv bloquant
- Communications des valeurs de champs sur les interfaces

### 2. update_ghost_periodic (update.hpp:940-1091)  
- Gestion des conditions périodiques niveau par niveau
- Communications pour les cellules aux bords périodiques

### 3. update_tag_subdomains (update.hpp:742-805)
- Propagation des tags (refine/coarsen/keep) entre sous-domaines
- Synchronisation des décisions d'adaptation

### 4. graduation (graduation.hpp)
- Propagation niveau par niveau pour garantir la cohérence
- Nécessite une progression séquentielle du fin vers le grossier

## Pourquoi c'est intrinsèquement mono-niveau

### Dépendances dans la multirésolution

1. **Opérateur de projection** (niveau l+1 → niveau l)
   - Moyenne des valeurs fines vers les cellules grossières
   - Nécessite que le niveau l+1 soit complet avant de projeter

2. **Opérateur de prédiction** (niveau l-1 → niveau l)  
   - Interpolation des valeurs grossières vers les cellules fines
   - Utilise des stencils d'ordre élevé (ordre 1 à 5)
   - Nécessite les cellules fantômes du niveau l-1

3. **Séquence dans update_ghost_mr** (lignes 538-566)
   ```
   max_level → min_level : projection + communications
   min_level → max_level : prédiction + communications
   ```

### Dépendances de données
- Chaque niveau dépend du précédent pour les calculs
- Les cellules fantômes d'un niveau influencent plusieurs niveaux
- La graduation propage les contraintes sur plusieurs niveaux

## Défis pour la factorisation

1. **Dépendances séquentielles** : La projection/prédiction crée une chaîne de dépendances
2. **Tailles variables** : Chaque niveau a des patterns de communication différents
3. **Cohérence multi-échelle** : La graduation nécessite une vue globale
4. **Complexité des stencils** : Les opérateurs d'ordre élevé augmentent les dépendances
