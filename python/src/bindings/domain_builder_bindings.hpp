// Samurai Python Bindings - DomainBuilder header
//
// Header for DomainBuilder class bindings

#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

// Module initialization function for DomainBuilder bindings
void init_domain_builder_bindings(py::module_& m);
