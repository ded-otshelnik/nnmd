#include "head.hpp"

#include "cpu/symmetric_functions.hpp"

#pragma once
namespace cpu{
    Tensor calculate_dG(const Tensor& cartesians, const Tensor& g, 
                        const vector<int>& features, const vector<vector<float>>& params,
                        const float h);
}