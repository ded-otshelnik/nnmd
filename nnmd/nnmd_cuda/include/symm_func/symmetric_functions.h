#ifndef SYMMETRIC_FUNCTIONS_H
#define SYMMETRIC_FUNCTIONS_H

#define _USE_MATH_DEFINES

#include "head.h"

#include <cmath>

Tensor calculate_sf(const Tensor& cartesians, const float& r_cutoff,
                const float& eta, const float& rs, const float& k_param,
                const int& lambda, const float& xi);
#endif