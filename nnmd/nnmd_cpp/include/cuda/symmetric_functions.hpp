#include "cuda/cuda_header.hpp"
#include "head.hpp"

#pragma once

namespace cuda{
    Tensor calculate_sf(const Tensor& cartesians, const float& r_cutoff,
                        const float& eta, const float& rs, const float& k_param,
                        const int& lambda, const float& xi);
}