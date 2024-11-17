#include "cuda/symmetric_functions.hpp"

#pragma once

namespace cuda{
    Tensor calculate_forces(const Tensor& cartesians, const Tensor& g,
                            const Tensor& dE, const float& r_cutoff,
                            const float& eta, const float& rs, const float& k,
                            const int& lambda, const float& xi, const float& h);
}