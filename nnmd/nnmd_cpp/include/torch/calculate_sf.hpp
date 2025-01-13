#pragma once
#include "head.hpp"

std::tuple<torch::Tensor, torch::Tensor> calculate_input(const torch::Tensor& cartesians,
            const vector<int> features, const vector<vector<double>> params);