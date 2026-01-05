// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

// Include the new implementation
#include "uniform/mesh.hpp"

// This file provides backward compatibility for the old UniformMesh API.
// The new API uses samurai::uniform::make_mesh() with mesh_config.
// See include/samurai/uniform/mesh.hpp for the modern implementation.

namespace samurai
{
    //==========================================================================
    //                     UniformConfig (DEPRECATED)
    //==========================================================================

    template <std::size_t dim_, int ghost_width_ = default_config::ghost_width, class TInterval = default_config::interval_t>
    struct [[deprecated("Use samurai::mesh_config and samurai::uniform::make_mesh instead")]] UniformConfig
    {
        static constexpr std::size_t dim                  = dim_;
        static constexpr int ghost_width                  = ghost_width_;
        static constexpr int prediction_stencil_radius    = 0; // No prediction for uniform
        static constexpr std::size_t max_refinement_level = default_config::max_level;
        using interval_t                                  = TInterval;
        using mesh_id_t                                   = UniformMeshId;
    };

} // namespace samurai
