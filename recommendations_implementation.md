# Recommandations d'implémentation pour la factorisation des communications MPI

## Analyse du problème

Le code actuel effectue des communications MPI niveau par niveau dans plusieurs endroits critiques :

1. **update_ghost_subdomains** : ~70 lignes de code par niveau × nombre de niveaux × nombre de voisins
2. **update_ghost_periodic** : Pattern similaire pour les conditions périodiques  
3. **update_tag_subdomains** : Synchronisation des décisions d'adaptation
4. **graduation** : Propagation des contraintes multi-échelles

Avec 10 niveaux et 26 voisins (3D), cela peut représenter jusqu'à 260 communications par étape !

## Solutions proposées par ordre de priorité

### 1. Solution immédiate : Agrégation des messages (Gain estimé : 50-70%)

**Modifications nécessaires dans `update.hpp` :**

```cpp
// Remplacer update_ghost_subdomains par une version agrégée
template <class Field>
void update_ghost_subdomains_all_levels(Field& field) {
    // Un seul send/recv par voisin contenant tous les niveaux
    // Voir example_aggregated_communication.hpp
}

// Modifier update_ghost_mr pour appeler la version agrégée
void update_ghost_mr(Field& field, Fields&... other_fields) {
    // ...
    // Remplacer la boucle sur les niveaux par :
    update_ghost_subdomains_all_levels(field, other_fields...);
    update_ghost_periodic_all_levels(field, other_fields...);
    
    // Garder la boucle pour projection/prédiction (dépendances)
    for (std::size_t level = max_level; level > min_level; --level) {
        // Projection seulement
    }
}
```

### 2. Solution à moyen terme : Pipeline asynchrone (Gain additionnel : 20-30%)

**Principe :**
- Chevaucher les communications avec les calculs de projection/prédiction
- Utiliser MPI_Isend/Irecv avec gestion fine des dépendances

**Points d'attention :**
- La projection du niveau l+1 vers l nécessite les ghosts du niveau l+1
- La prédiction du niveau l-1 vers l nécessite les ghosts du niveau l-1
- Possibilité de commencer la projection pendant que les ghosts se communiquent

### 3. Solution avancée : Refonte algorithmique

**Idées à explorer :**
- Algorithmes de multirésolution "communication-avoiding"
- Relaxation des contraintes de graduation (graduation approchée)
- Méthodes de type "multiplicative cascade" permettant plus de parallélisme

## Plan d'implémentation recommandé

### Phase 1 (2-3 semaines)
1. Implémenter `MultiLevelGhostData` et la sérialisation boost
2. Créer `update_ghost_subdomains_aggregated`
3. Tester sur des cas simples (2D, peu de niveaux)
4. Mesurer les gains de performance

### Phase 2 (2-3 semaines)
1. Étendre aux conditions périodiques
2. Implémenter l'agrégation pour les tags
3. Optimiser la taille des buffers
4. Gérer les cas dégénérés (niveaux vides)

### Phase 3 (1 mois)
1. Introduire le pipeline asynchrone
2. Analyser finement les dépendances
3. Implémenter un ordonnanceur de communications
4. Validation extensive

## Métriques de succès

- Réduction du nombre de messages MPI de O(L×N) à O(N) où L=niveaux, N=voisins
- Réduction du temps de communication d'au moins 50%
- Maintien de la précision numérique
- Pas d'augmentation significative de la mémoire (< 20%)

## Risques et mitigation

1. **Mémoire** : Les buffers agrégés peuvent être volumineux
   - Mitigation : Compression, envoi par chunks si nécessaire

2. **Latence** : Attendre tous les niveaux peut augmenter la latence
   - Mitigation : Pipeline partiel, priorisation des niveaux critiques

3. **Complexité** : Le code devient plus complexe
   - Mitigation : Bonne encapsulation, tests unitaires exhaustifs

## Code de test proposé

```cpp
// Benchmark simple pour mesurer les gains
void benchmark_ghost_update() {
    // Version actuelle
    auto t1 = MPI_Wtime();
    for (int iter = 0; iter < 100; ++iter) {
        update_ghost_mr(field);
    }
    auto time_current = MPI_Wtime() - t1;
    
    // Version agrégée
    auto t2 = MPI_Wtime();
    for (int iter = 0; iter < 100; ++iter) {
        update_ghost_mr_aggregated(field);
    }
    auto time_aggregated = MPI_Wtime() - t2;
    
    std::cout << "Speedup: " << time_current / time_aggregated << "x" << std::endl;
}
```
