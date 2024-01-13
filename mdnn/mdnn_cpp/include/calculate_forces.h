#include "head.h"

#include "symm_func/symmetric_functions.h"

#pragma once

Tensor calculate_forces(const Tensor& cartesians, const Tensor& e_nn, const Tensor& g,
                const py::list& nets, const double& r_cutoff,
                const double& eta, const double& rs, const double& k,
                const int& lambda, const double& xi, const double& h);