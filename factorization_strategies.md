# Stratégies de factorisation des communications MPI

## Stratégie 1 : Agrégation des messages par voisin

### Principe
Au lieu d'envoyer niveau par niveau, créer un buffer unique par voisin contenant tous les niveaux.

### Implémentation
```cpp
// Structure pour stocker les données multi-niveaux
struct MultiLevelBuffer {
    std::vector<std::vector<value_t>> data_per_level;
    std::vector<size_t> level_sizes;
};

// Un seul send/recv par voisin
for (auto& neighbour : mesh.mpi_neighbourhood()) {
    MultiLevelBuffer buffer;
    // Remplir le buffer avec tous les niveaux
    for (level = min_level; level <= max_level; ++level) {
        // Collecter les données du niveau
    }
    // Un seul envoi
    world.isend(neighbour.rank, tag, buffer);
}
```

### Avantages
- Réduction du nombre de messages de O(niveaux × voisins) à O(voisins)
- Meilleure utilisation de la bande passante

### Inconvénients  
- Plus de mémoire pour les buffers
- Latence potentiellement plus élevée (attendre tous les niveaux)

## Stratégie 2 : Pipeline asynchrone

### Principe
Utiliser des communications non-bloquantes pour chevaucher calculs et communications.

### Implémentation
```cpp
// Phase descendante (projection)
std::vector<mpi::request> requests;
for (level = max_level; level > min_level; --level) {
    // Lancer les comms du niveau précédent pendant le calcul
    if (level < max_level) {
        requests.push_back(isend_ghost_data(level+1));
    }
    
    // Calcul de projection
    apply_projection(level);
    
    // Attendre seulement si nécessaire
    if (need_data_from_level(level+1)) {
        wait_requests(level+1);
    }
}
```

### Avantages
- Recouvrement calcul/communication
- Réduction du temps total

### Inconvénients
- Complexité de gestion des dépendances
- Besoin d'analyser finement les dépendances

## Stratégie 3 : Communication par blocs de niveaux

### Principe  
Regrouper les niveaux qui n'ont pas de dépendances directes.

### Analyse des dépendances
- Projection : niveau l+1 → niveau l (séquentiel descendant)
- Prédiction : niveau l-1 → niveau l (séquentiel ascendant)
- Graduation : peut nécessiter 2-3 niveaux de distance

### Implémentation
```cpp
// Identifier les groupes indépendants
std::vector<std::vector<size_t>> level_groups;

// Pour la phase de prédiction ascendante
// Les niveaux pairs peuvent être traités en parallèle
// puis les niveaux impairs
for (auto& group : level_groups) {
    parallel_for(group, [&](size_t level) {
        update_ghost_subdomains(level, field);
    });
    barrier(); // Synchronisation entre groupes
}
```

### Avantages
- Parallélisation partielle possible
- Réduction du nombre de barrières

### Inconvénients
- Limite par les dépendances de la multirésolution
- Complexité d'implémentation

## Stratégie 4 : Communication persistante

### Principe
Pré-allouer et réutiliser les canaux de communication.

### Implémentation  
```cpp
class PersistentComm {
    std::vector<MPI_Request> send_requests;
    std::vector<MPI_Request> recv_requests;
    std::vector<Buffer> buffers;
    
    void setup() {
        for (level : all_levels) {
            for (neighbour : neighbours) {
                // Créer des requêtes persistantes
                MPI_Send_init(...);
                MPI_Recv_init(...);
            }
        }
    }
    
    void communicate() {
        MPI_Startall(send_requests);
        MPI_Startall(recv_requests);
        MPI_Waitall(...);
    }
};
```

### Avantages
- Réduction du coût d'initialisation
- Meilleure prédictibilité

## Recommandations

1. **Court terme** : Implémenter la Stratégie 1 (agrégation)
   - Plus simple à intégrer
   - Gains immédiats sur le nombre de messages

2. **Moyen terme** : Combiner Stratégies 1 et 2
   - Agrégation + pipeline asynchrone
   - Nécessite une refonte modérée

3. **Long terme** : Repenser l'algorithme
   - Algorithmes de multirésolution adaptés au parallèle
   - Relaxation des contraintes de graduation
