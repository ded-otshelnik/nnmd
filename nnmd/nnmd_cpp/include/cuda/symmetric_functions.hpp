#include "cuda/cuda_header.hpp"
#include "head.hpp"

#pragma once

namespace cuda{
    Tensor calculate_sf(const Tensor& cartesians, const vector<int>& features, const vector<vector<float>>& params);
}