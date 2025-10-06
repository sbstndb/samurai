Unification AMR avec MR/MRA — Conception, écarts et plan d’action

Objectif
- Proposer une interface AMR inspirée de l’API MR/MRA existante, compatible MPI et périodicité, en uniformisant les patterns d’adaptation, de mise à jour des ghosts et des MeshIDs.
- Documenter précisément les écarts actuels et un plan d’implémentation incrémental, avec extraits de code ciblés.

Contexte et état des lieux
- AMR (include/samurai/amr/mesh.hpp)
  - AMR_Id = { cells, cells_and_ghosts, proj_cells, pred_cells, all_cells, reference=all_cells }.
  - update_sub_mesh_impl: construit cells_and_ghosts, proj_cells, pred_cells, all_cells (union cells_and_ghosts ∪ proj_cells).
  - Pas d’API exists(...). Pas de synchronisation MPI explicite des MeshIDs (pas d’appels à update_neighbour_subdomain / update_meshid_neighbour). Pas d’extension périodique sur all_cells.
  - Adaptation “ad hoc” dans les démos (ex: gradient thresholds), séquencée via graduation(tag, ...) + update_field(tag, ...).
- MR/MRA (include/samurai/mr/*)
  - MeshIds pour MR: { cells, cells_and_ghosts, proj_cells, union_cells, all_cells, reference=all_cells }.
  - MRO (overleaves) ajoute overleaves pour correction de flux.
  - MRMesh::update_sub_mesh_impl: logique avancée (périodicité, MPI) avec update_neighbour_subdomain/update_meshid_neighbour, remplissage all_cells et images périodiques.
  - Adaptation générique via samurai::Adapt + mra_config (epsilon, regularity, relative_detail), calcul de détails (Harten), opérateurs dédiés (compute_detail, to_refine_mr, to_coarsen_mr, enlarge…).
  - Ghosts/BC: update_ghost_mr(...), project_bc/predict_bc couche par couche, support MPI/périodicité.

Synthèse des écarts à combler côté AMR
1) Parité Mesh API
   - Ajouter une méthode utilitaire exists(...) comme sur MR/MRO.
   - Étendre update_sub_mesh_impl pour gérer périodicité et synchronisation MPI des sets clés (cells_and_ghosts, reference/all_cells), à la MR.
2) Adaptation unifiée
   - Introduire un module amr::Adapt (inspiré de mr/adapt.hpp) orchestrant: update_ghost(...), critère utilisateur (ou défaut), graduation(...), update_field(...), avec options “refine-only” en boucle interne puis coarsen optionnel — compatible MPI.
   - Fournir une amr_config (ou un adapt_config commun) + intégration CLI (flags AMR, à l’image de mra_config et args::...).
3) Flux coarse/fine
   - Laisser overleaves à MR/MRO. Pour AMR, exposer un helper générique de correction de flux coarse/fine (comme dans demos/FiniteVolume/AMR_Burgers_Hat.cpp) sous forme d’API, pour homogénéiser les usages.

Principes de conception
- Non-intrusif: conserver AMR_Id actuel. Utiliser get_union() de Mesh_base plutôt que d’ajouter union_cells à AMR_Id.
- Réutiliser les briques communes (graduation, update_field, update_ghost) afin de maximiser la compatibilité MPI existante (update_*_subdomains/periodic).
- Offrir une façade AMR Adapt proche de MRA: mêmes fabriques make_AMRAdapt, même ergonomie “multi-champs”. Les différences résident dans le critère (AMR ≠ Harten MR) et dans l’usage de pred_cells (spécifique AMR).

Design proposé — API AMR unifiée
1) Mesh: utilitaire exists(...) et robustesse MPI/périodique

Extrait à ajouter dans include/samurai/amr/mesh.hpp (interface):

```cpp
template <class Config>
class Mesh : public Mesh_base<Mesh<Config>, Config>
{
    // ...
    template <typename... T>
    xt::xtensor<bool, 1> exists(mesh_id_t type, std::size_t level, interval_t interval, T... index) const;
};

template <class Config>
template <typename... T>
inline xt::xtensor<bool, 1> Mesh<Config>::exists(mesh_id_t type, std::size_t level, interval_t interval, T... index) const
{
    using coord_index_t      = typename interval_t::coord_index_t;
    const auto& lca          = this->cells()[type][level];
    std::size_t size         = interval.size() / interval.step;
    xt::xtensor<bool, 1> out = xt::empty<bool>({size});
    std::size_t iout         = 0;
    for (coord_index_t i = interval.start; i < interval.end; i += interval.step)
    {
        auto offset = find(lca, {i, index...});
        out[iout++] = (offset != -1);
    }
    return out;
}
```

Étendre update_sub_mesh_impl pour:
- Appeler update_neighbour_subdomain() et update_meshid_neighbour(...) sur les MeshIDs pertinents (cells_and_ghosts, reference/all_cells) afin de synchroniser en MPI.
- Ajouter les images périodiques pour les fantômes, comme dans MRMesh, en utilisant get_periodic_shift(...) et translate(...).

Point d’ancrage dans AMR::Mesh::update_sub_mesh_impl (pseudo-code ciblé, inspiré de MR):

```cpp
// Après avoir construit cells_and_ghosts, proj_cells, pred_cells, all_cells...

// 1) Synchroniser le sous-domaine MPI
this->update_neighbour_subdomain();

// 2) Synchroniser des MeshIDs spécifiques avec les voisins
this->update_meshid_neighbour(mesh_id_t::cells_and_ghosts);
this->update_meshid_neighbour(mesh_id_t::reference); // alias all_cells

// 3) Étendre périodiquement (par direction périodique) autour du sous-domaine
for (std::size_t level = this->min_level(); level <= this->max_level(); ++level)
{
    for (std::size_t d = 0; d < dim; ++d)
    {
        if (this->is_periodic(d))
        {
            auto domain_shift = get_periodic_shift(this->domain(), level, d);
            auto left  = intersection(nestedExpand(self(this->subdomain()).on(level), config::ghost_width),
                                      translate(this->cells()[mesh_id_t::reference][level], -domain_shift));
            auto right = intersection(nestedExpand(self(this->subdomain()).on(level), config::ghost_width),
                                      translate(this->cells()[mesh_id_t::reference][level],  domain_shift));
            left ( [&](const auto& i, const auto& idx){ this->cells()[mesh_id_t::all_cells][level][idx].add_interval(i); });
            right( [&](const auto& i, const auto& idx){ this->cells()[mesh_id_t::all_cells][level][idx].add_interval(i); });
        }
    }
}

// 4) Re-publier all_cells si modifié
this->update_meshid_neighbour(mesh_id_t::all_cells);
```

Remarques:
- Le code MR contient des variantes plus complètes (gestion fine de delta de niveaux et boxes min/max). L’AMR n’a pas besoin de la totalité de cette sophistication si les fantômes/pred_cells/proj_cells sont cohérents au voisinage — miser sur nestedExpand(..., ghost_width) + translate + intersections.
- Mesh_base fournit déjà get_union(), domain/subdomain, periodicité et MPI_neighbourhood.

2) Module d’adaptation AMR — façade compatible MR

But: proposer une API “Adapt” pour AMR inspirée de mr/adapt.hpp, mais pilotée par un critère utilisateur (ou par défaut). Le pipeline est:
  - update_ghost(...) (AMR s’appuie sur pred_cells/proj_cells)
  - calcul du tag par niveau (keep/refine/coarsen)
  - graduation(tag, stencil)
  - update_field(tag, fields...) → projection/prediction + swap du mesh

Interface proposée: include/samurai/amr/adapt.hpp

```cpp
#pragma once

#include "../algorithm/graduation.hpp"
#include "../algorithm/update.hpp"
#include "../arguments.hpp"
#include "../field.hpp"

namespace samurai::amr
{
    struct amr_config
    {
        double refine_threshold = 0.0; // interprétation dépend du critère
        double coarsen_ratio    = 0.7; // 0<r<1
        bool   allow_coarsen    = true;
        bool   refine_boundary  = false; // garder le bord raffiné au max_level

        // parse depuis args:: (à ajouter dans arguments.hpp)
        void parse_args();
    };

    // Critère: f(fields, level) -> marque tag (keep/refine/coarsen)
    template <class CriterionFn, class Field, class... Fields>
    class Adapt
    {
      public:
        using mesh_t    = typename Field::mesh_t;
        using mesh_id_t = typename mesh_t::mesh_id_t;
        static constexpr std::size_t dim = Field::dim;

        Adapt(CriterionFn&& crit, Field& f, Fields&... others)
            : m_crit(std::forward<CriterionFn>(crit)), m_fields(f, others...), m_tag("tag", f.mesh())
        {
        }

        void operator()(amr_config cfg)
        {
            auto& mesh     = std::get<0>(m_fields.elements()).mesh();
            auto min_level = mesh.min_level();
            auto max_level = mesh.max_level();
            if (min_level == max_level) return;

            cfg.parse_args();

            // Boucle interne: refine-only jusqu’à stabilité
            for (;;)
            {
                update_ghost(std::get<0>(m_fields.elements())); // variadique possible
                m_tag.resize();
                m_tag.fill(static_cast<int>(CellFlag::keep));

                // Appliquer le critère
                m_crit(m_fields, m_tag, cfg);

                // Graduation et mises à jour périodiques/MPI sur tag si besoin
                auto stencil = star_stencil<dim>();
                graduation(m_tag, stencil);

                if (update_field(m_tag, m_fields)) break; // swap mesh + projection/prediction
            }

            // Optionnel: passe unique de coarsening
            if (cfg.allow_coarsen)
            {
                update_ghost(std::get<0>(m_fields.elements()));
                m_tag.resize();
                m_tag.fill(static_cast<int>(CellFlag::keep));
                m_crit(m_fields, m_tag, cfg, /*coarsen_pass=*/true);
                auto stencil = star_stencil<dim>();
                graduation(m_tag, stencil);
                update_field(m_tag, m_fields);
            }
        }

      private:
        CriterionFn m_crit;
        Field_tuple<Field, Fields...> m_fields;
        ScalarField<mesh_t, int> m_tag;
    };

    // Fabriques
    template <class CriterionFn, class Field, class... Fields>
    auto make_AMRAdapt(CriterionFn&& crit, Field& f, Fields&... others)
    {
        return Adapt<CriterionFn, Field, Fields...>(std::forward<CriterionFn>(crit), f, others...);
    }
}
```

Critère par défaut (optionnel), inspiré de demos/FiniteVolume/AMR_Burgers_Hat.cpp (norme du gradient). Exemple d’esquisse (dans le même header ou dédié):

```cpp
struct GradientThreshold
{
    template <class FieldsTuple, class Tag>
    void operator()(FieldsTuple& fields, Tag& tag, const amr_config& cfg, bool coarsen_pass = false) const
    {
        auto& field = std::get<0>(fields.elements());
        auto  mesh  = field.mesh();
        using mesh_id_t = typename decltype(mesh)::mesh_id_t;
        constexpr std::size_t dim = Tag::dim;

        for (std::size_t level = mesh.min_level(); level <= mesh.max_level(); ++level)
        {
            double dx = mesh.cell_length(level);
            for_each_interval(mesh[mesh_id_t::cells][level],
                [&](std::size_t, auto& i, auto& index)
                {
                    if constexpr (dim == 1)
                    {
                        auto grad = samurai::eval(abs((field(level, i + 1) - field(level, i - 1)) / (2.0 * dx)));
                        apply_on_masked(tag(level, i), coarsen_pass ? (grad <= cfg.refine_threshold * cfg.coarsen_ratio)
                                                                     : (grad >  cfg.refine_threshold),
                                        [&](auto& e){ e = static_cast<int>(coarsen_pass ? CellFlag::coarsen : CellFlag::refine); });
                    }
                    else if constexpr (dim == 2)
                    {
                        auto j   = index[0];
                        auto dfx = samurai::eval((field(level, i + 1, j) - field(level, i - 1, j)) / (2.0 * dx));
                        auto dfy = samurai::eval((field(level, i, j + 1) - field(level, i, j - 1)) / (2.0 * dx));
                        auto g   = samurai::eval(sqrt(dfx * dfx + dfy * dfy));
                        apply_on_masked(tag(level, i, j), coarsen_pass ? (g <= cfg.refine_threshold * cfg.coarsen_ratio)
                                                                        : (g >  cfg.refine_threshold),
                                        [&](auto& e){ e = static_cast<int>(coarsen_pass ? CellFlag::coarsen : CellFlag::refine); });
                    }
                });
        }
    }
};
```

3) Intégration CLI (arguments.hpp)
- Étendre samurai::args avec:

```cpp
namespace samurai::args {
    static double amr_threshold    = std::numeric_limits<double>::infinity();
    static double amr_coarsen_ratio = 0.7;
    static bool   amr_allow_coarsen = true;
}

// Dans read_samurai_arguments(...):
app.add_option("--amr-threshold", args::amr_threshold, "AMR refine threshold (level-invariant)").group("AMR");
app.add_option("--amr-coarsen-ratio", args::amr_coarsen_ratio, "AMR coarsen as ratio of refine threshold (0<r<1)").group("AMR");
app.add_flag  ("--amr-allow-coarsen", args::amr_allow_coarsen, "Allow coarsening during adaptation").group("AMR");
```

Et dans amr_config::parse_args():

```cpp
void amr_config::parse_args()
{
    if (samurai::args::amr_threshold != std::numeric_limits<double>::infinity())
        refine_threshold = samurai::args::amr_threshold;
    if (samurai::args::amr_coarsen_ratio > 0.0 && samurai::args::amr_coarsen_ratio < 1.0)
        coarsen_ratio = samurai::args::amr_coarsen_ratio;
    allow_coarsen = samurai::args::amr_allow_coarsen;
    refine_boundary = samurai::args::refine_boundary; // déjà présent
}
```

4) Compatibilité MPI et périodicité — parcours d’exécution
- update_sub_mesh_impl (AMR) doit, après construction des sets, appeler:
  - update_neighbour_subdomain();
  - update_meshid_neighbour(mesh_id_t::cells_and_ghosts);
  - update_meshid_neighbour(mesh_id_t::reference) et, si besoin, (mesh_id_t::all_cells) après extension périodique.
- Dans Adapt::operator(): graduation(tag, ...) et update_field(tag, ...) sont déjà compatibles MPI (Mesh_base::operator==, update_fields). Pour la robustesse aux frontières de sous-domaines, il est possible d’ajouter, comme en MR, des synchronisations de tag:
  - update_tag_periodic(level, tag);
  - update_tag_subdomains(level, tag, /*only_keep=*/true);
  Ces helpers existent dans algorithm/update.hpp et peuvent être intégrés si nécessaire après le marquage par critère et avant graduation.

Flux coarse/fine (optionnel)
- Exposer un helper générique de correction de flux AMR similaire au code des démos (1D/2D), par exemple dans include/samurai/schemes/fv/utils.hpp, afin d’homogénéiser l’approche sans introduire overleaves côté AMR.

Plan d’implémentation (par étapes, incrémental)
1) Mesh utils AMR
   - Ajouter Mesh::exists(...) (copie de MR/MRO adaptée à AMR::Mesh).
   - Étendre amr::Mesh::update_sub_mesh_impl avec synchronisation MPI/périodique (appels update_neighbour_subdomain/update_meshid_neighbour + extension périodique basique comme montré plus haut).
   - Vérifier que save/restart et HDF5 restent inchangés (Mesh_base déjà générique).
2) Module amr::Adapt
   - Créer include/samurai/amr/adapt.hpp avec amr_config, Adapt<Crit, ...>, make_AMRAdapt(...), et un critère par défaut GradientThreshold (optionnel).
   - Utiliser update_ghost(...) (pas update_ghost_mr) car AMR dispose de pred_cells/proj_cells.
   - Orchestration: boucle refine-only jusqu’à stabilité, puis coarsen pass optionnelle.
   - Intégrer refine_boundary si activé (éventuel helper keep_boundary_refined analogue MR, optionnel et simple à porter).
3) CLI
   - Étendre include/samurai/arguments.hpp avec les options AMR et implémenter amr_config::parse_args().
4) Tests
   - Ajouter tests unitaires: construction AMR Mesh avec périodicité et MPI (si CI le permet), test de stabilité de la boucle Adapt (critère trivial), test exists(...).
   - Tester update_ghost sur cas multi-niveaux avec frontières périodiques (vérifier absence de trous dans all_cells au voisinage du bord périodique).
5) Démos
   - Proposer une variante des démos AMR utilisant make_AMRAdapt + GradientThreshold au lieu du code ad hoc, en gardant l’ancienne version pour comparaison.

Notes de migration
- Aucun changement de signature publique existante n’est requis; l’ajout de exists(...) est additive.
- Les utilisateurs actuels d’AMR pourront soit conserver leur critère spécifique, soit le brancher via make_AMRAdapt(custom_criterion, fields...).
- Les options CLI AMR proposées alignent le comportement sur les démos existantes et permettent un passage progressif vers la façade unifiée.

Questions ouvertes / décisions à prendre
- Faut-il introduire un adapt_config commun MR/AMR (fusion de mra_config et amr_config) ?
  - Avantage: une seule façade pour les deux mondes; Inconvénient: concepts différents (epsilon/regularity vs thresholds). Recommandation: garder 2 configs pour l’instant.
- Souhaitez-vous porter keep_boundary_refined(...) de MR tel-quel pour AMR ? Simple et utile dans certains cas.
- Souhaitez-vous un helper générique de correction de flux AMR dans la bibliothèque (en dehors des démos) ?

Résumé
- Cette proposition aligne AMR sur MR/MRA: utilitaires Mesh (exists, synchro MPI/périodique des MeshIDs), façade d’adaptation Adapt inspirée MR, et intégration CLI. L’implémentation est compacte, réutilise les composants existants (graduation, update_field, update_ghost) et renforce la robustesse MPI/périodicité.

