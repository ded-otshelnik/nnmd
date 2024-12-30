#pragma once
#include "cuda/symmetric_functions.hpp"

namespace cuda{
    Tensor calculate_dG(const Tensor& cartesians, const Tensor& g, const float& r_cutoff,
                        const float& eta, const float& rs, const float& k,
                        const int& lambda, const float& xi, const float& h);
}