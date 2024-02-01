#include "head.hpp"
#include "cuda/symmetric_functions.hpp"

#pragma once

namespace cuda{

    Tensor calculate_forces(const Tensor& cartesians, const Tensor& e_nn, const Tensor& g,
                    const py::list& nets, const float& r_cutoff,
                    const float& eta, const float& rs, const float& k,
                    const int& lambda, const float& xi, const float& h);

}