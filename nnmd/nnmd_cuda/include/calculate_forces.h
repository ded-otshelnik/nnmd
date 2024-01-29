#ifndef CALCULATE_FORCES_H
#define CALCULATE_FORCES_H

#include "head.h"
#include "symm_func/symmetric_functions.h"

#pragma once

Tensor calculate_forces(const Tensor& cartesians, const Tensor& e_nn, const Tensor& g,
                const py::list& nets, const float& r_cutoff,
                const float& eta, const float& rs, const float& k,
                const int& lambda, const float& xi, const float& h);

#endif