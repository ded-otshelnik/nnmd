#include "head.hpp"

#define _USE_MATH_DEFINES

#include <cmath>

namespace cpu{
    float G1(const float rij, const float rc);
    float G2(const float rij, const float rc, const float eta, const float rs);
    float G3(const float rij, const float rc, const float k);
    float G4(const float rij, const float rik, const float rjk, const float rc,
             const float eta, const float lambda, const float xi,
             const float cos_v);
    float G5(const float rij, const float rik, const float rjk, const float rc,
             const float eta, const float lambda, const float xi,
             const float cos_v);

    Tensor calculate_sf(const Tensor &cartesians, const float& r_cutoff,
                        const float& eta, const float& rs, const float& k_param,
                        const int& lambda, const float& xi);
}