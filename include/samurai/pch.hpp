// Precompiled header for samurai and demos
// List here the most common and heavy headers used across sources.
// CMake will force-include this header for targets that opt-in.

#pragma once

// Standard library
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

// Third-party heavy headers
#include <fmt/format.h>

// xtensor is heavily used across the project
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xfixed.hpp>

// Note: We intentionally do not include CLI11 here since
// not all targets link CLI11; including it would require
// its include directories when generating the PCH.

