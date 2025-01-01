#pragma once
#include "cuda/symmetric_functions.hpp"
#include "cuda/cuda_header.hpp"

namespace cuda{
    Tensor calculate_dG(const Tensor& cartesians, const Tensor& g, 
                        const vector<int>& features, const vector<vector<float>>& params,
                        const float h);
}