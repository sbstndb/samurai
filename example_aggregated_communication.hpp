#pragma once

#include <vector>
#include <boost/mpi.hpp>
#include <boost/serialization/vector.hpp>

namespace samurai {

template <typename Field>
struct MultiLevelGhostData {
    struct LevelData {
        std::size_t level;
        std::vector<typename Field::value_type> values;
        std::vector<std::size_t> cell_indices; // Pour reconstruire où placer les données
        
        template<class Archive>
        void serialize(Archive & ar, const unsigned int version) {
            ar & level;
            ar & values;
            ar & cell_indices;
        }
    };
    
    std::vector<LevelData> levels_data;
    
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version) {
        ar & levels_data;
    }
};

template <class Field>
void update_ghost_subdomains_aggregated(Field& field) {
#ifdef SAMURAI_WITH_MPI
    using mesh_t    = typename Field::mesh_t;
    using value_t   = typename Field::value_type;
    using mesh_id_t = typename mesh_t::mesh_id_t;
    
    auto& mesh = field.mesh();
    mpi::communicator world;
    
    std::size_t min_level = mesh.min_level();
    std::size_t max_level = mesh.max_level();
    
    // Préparer les buffers agrégés pour chaque voisin
    std::vector<MultiLevelGhostData<Field>> send_buffers(mesh.mpi_neighbourhood().size());
    std::vector<mpi::request> requests;
    
    // Phase 1: Collecter toutes les données à envoyer pour tous les niveaux
    std::size_t neighbor_idx = 0;
    for (auto& neighbour : mesh.mpi_neighbourhood()) {
        auto& buffer = send_buffers[neighbor_idx];
        
        // Collecter les données pour chaque niveau
        for (std::size_t level = min_level; level <= max_level; ++level) {
            if (!mesh[mesh_id_t::reference][level].empty() && 
                !neighbour.mesh[mesh_id_t::reference][level].empty()) {
                
                typename MultiLevelGhostData<Field>::LevelData level_data;
                level_data.level = level;
                
                // Interface sortante
                auto out_interface = intersection(mesh[mesh_id_t::reference][level],
                                                neighbour.mesh[mesh_id_t::reference][level],
                                                mesh.subdomain())
                                        .on(level);
                                        
                out_interface([&](const auto& i, const auto& index) {
                    // Stocker les valeurs
                    auto field_values = field(level, i, index);
                    std::copy(field_values.begin(), field_values.end(), 
                             std::back_inserter(level_data.values));
                    
                    // Stocker les indices pour reconstruction
                    level_data.cell_indices.push_back(i.start);
                    level_data.cell_indices.push_back(i.size());
                    for (std::size_t d = 0; d < mesh_t::dim - 1; ++d) {
                        level_data.cell_indices.push_back(index[d]);
                    }
                });
                
                // Ajouter aussi les coins du sous-domaine
                auto subdomain_corners = outer_subdomain_corner<true>(level, field, neighbour);
                for_each_interval(subdomain_corners,
                    [&](const auto, const auto& i, const auto& index) {
                        auto field_values = field(level, i, index);
                        std::copy(field_values.begin(), field_values.end(), 
                                 std::back_inserter(level_data.values));
                        
                        level_data.cell_indices.push_back(i.start);
                        level_data.cell_indices.push_back(i.size());
                        for (std::size_t d = 0; d < mesh_t::dim - 1; ++d) {
                            level_data.cell_indices.push_back(index[d]);
                        }
                    });
                
                if (!level_data.values.empty()) {
                    buffer.levels_data.push_back(std::move(level_data));
                }
            }
        }
        
        // Envoyer le buffer agrégé (un seul message par voisin)
        if (!buffer.levels_data.empty()) {
            requests.push_back(world.isend(neighbour.rank, neighbour.rank, buffer));
        }
        neighbor_idx++;
    }
    
    // Phase 2: Recevoir et traiter les buffers agrégés
    for (auto& neighbour : mesh.mpi_neighbourhood()) {
        MultiLevelGhostData<Field> recv_buffer;
        world.recv(neighbour.rank, world.rank(), recv_buffer);
        
        // Traiter chaque niveau reçu
        for (const auto& level_data : recv_buffer.levels_data) {
            std::size_t level = level_data.level;
            
            if (!mesh[mesh_id_t::reference][level].empty() && 
                !neighbour.mesh[mesh_id_t::reference][level].empty()) {
                
                std::size_t value_idx = 0;
                std::size_t cell_idx = 0;
                
                // Reconstruire et appliquer les données
                while (cell_idx < level_data.cell_indices.size()) {
                    auto i_start = level_data.cell_indices[cell_idx++];
                    auto i_size = level_data.cell_indices[cell_idx++];
                    
                    index_t index;
                    for (std::size_t d = 0; d < mesh_t::dim - 1; ++d) {
                        index[d] = level_data.cell_indices[cell_idx++];
                    }
                    
                    interval_t interval{i_start, i_start + i_size};
                    
                    // Copier les valeurs
                    auto field_ref = field(level, interval, index);
                    std::copy(level_data.values.begin() + value_idx,
                             level_data.values.begin() + value_idx + i_size * Field::n_comp,
                             field_ref.begin());
                    
                    value_idx += i_size * Field::n_comp;
                }
            }
        }
    }
    
    // Attendre que tous les envois soient terminés
    mpi::wait_all(requests.begin(), requests.end());
#endif
}

// Version avec pipeline pour la phase descendante (projection)
template <class Field, class... Fields>
void update_ghost_mr_pipelined(Field& field, Fields&... other_fields) {
    using mesh_id_t = typename Field::mesh_t::mesh_id_t;
    constexpr std::size_t pred_order = Field::mesh_t::config::prediction_order;
    
    auto& mesh = field.mesh();
    auto max_level = mesh.max_level();
    std::size_t min_level = 0;
    
    // Structures pour gérer le pipeline
    std::vector<std::vector<mpi::request>> requests_per_level(max_level + 1);
    
    update_outer_ghosts(max_level, field, other_fields...);
    
    // Phase descendante avec pipeline
    for (std::size_t level = max_level; level > min_level; --level) {
        // Lancer les communications du niveau actuel (non-bloquant)
        update_ghost_periodic_async(level, field, other_fields..., requests_per_level[level]);
        update_ghost_subdomains_async(level, field, other_fields..., requests_per_level[level]);
        
        // Projection (peut commencer pendant les communications)
        auto set_at_levelm1 = intersection(mesh[mesh_id_t::reference][level], 
                                         mesh[mesh_id_t::proj_cells][level - 1]).on(level - 1);
        set_at_levelm1.apply_op(variadic_projection(field, other_fields...));
        
        // Attendre seulement si nécessaire pour le niveau suivant
        if (level > min_level + 1) {
            // Vérifier si les données du niveau actuel sont nécessaires
            // pour les calculs du niveau suivant
            if (need_ghost_data_for_projection(level - 1)) {
                mpi::wait_all(requests_per_level[level].begin(), 
                            requests_per_level[level].end());
            }
        }
        
        update_outer_ghosts(level - 1, field, other_fields...);
    }
    
    // S'assurer que toutes les communications sont terminées
    for (auto& level_requests : requests_per_level) {
        if (!level_requests.empty()) {
            mpi::wait_all(level_requests.begin(), level_requests.end());
        }
    }
    
    // Suite du traitement...
}

} // namespace samurai
