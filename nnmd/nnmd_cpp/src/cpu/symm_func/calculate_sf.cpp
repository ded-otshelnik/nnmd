#include "head.hpp"
#include "cpu/symmetric_functions.hpp"

namespace cpu{

    // @brief Calculates symmetric functions and its derivatives
    // @param cartesians: atomic positions
    // @param r_cutoff: cutoff radius
    // @param eta: parameter of symmetric functions
    // @param rs: parameter of symmetric functions
    // @param lambda: parameter of symmetric functions
    // @param xi: parameter of symmetric functions
    Tensor calculate_sf(const Tensor &cartesians, const float& r_cutoff,
                    const float& eta, const float& rs, const float& k_param,
                    const int& lambda, const float& xi){

            auto opts = torch::TensorOptions()
                .dtype(torch::kFloat);
            // atoms amount
            int n_atoms = cartesians.sizes()[0];

            // output g values
            Tensor g_total = torch::zeros({n_atoms, 5}, opts);

            // get accessors to tensors 
            auto cartesians_accessor = cartesians.accessor<float, 2>();

            // current g value
            float g;

            // loop by atoms
            for (int i = 0; i < n_atoms; i++){
                // loop by symmetric functions type
                for (int g_type = 1; g_type <= 5; g_type++){
                    g = 0;
                    switch (g_type){
                        // G1
                        case 1:{
                            for (int j = 0; j < n_atoms; j++){
                                if (i == j){
                                    continue;
                                }
                                auto ri = cartesians_accessor[i];
                                auto rj = cartesians_accessor[j];
                                float rij = 0;
                                for (int dim = 0; dim < 3; dim++){
                                    rij += (ri[dim] - rj[dim]) * (ri[dim] - rj[dim]);
                                }

                                rij = sqrt(rij);
                                g += G1(rij, r_cutoff);
                            }
                            break;
                        }
                        // G2
                        case 2:
                        {
                            for (int j = 0; j < n_atoms; j++){
                                if (i == j){
                                    continue;
                                }
                                auto ri = cartesians_accessor[i];
                                auto rj = cartesians_accessor[j];
                                float rij = 0;
                                for (int dim = 0; dim < 3; dim++){
                                    rij += (ri[dim] - rj[dim]) * (ri[dim] - rj[dim]);
                                }
                                rij = sqrt(rij);
                                g += G2(rij, r_cutoff, eta, rs);
                            }
                            break;
                        }
                        // G3
                        case 3:
                        {
                            for (int j = 0; j < n_atoms; j++){
                                if (i == j){
                                    continue;
                                }
                                auto ri = cartesians_accessor[i];
                                auto rj = cartesians_accessor[j];
                                float rij = 0;
                                for (int dim = 0; dim < 3; dim++){
                                    rij += (ri[dim] - rj[dim]) * (ri[dim] - rj[dim]);
                                }
                                rij = sqrt(rij);
                                g += G3(rij, r_cutoff, k_param);
                            }
                            break;
                        }
                        // G4
                        case 4:
                        {
                            for (int j = 0; j < n_atoms; j++){
                                for (int k = 0; k < n_atoms; k++){
                                    if (i == j || i == k || j == k){
                                        continue;
                                    }
                                    auto ri = cartesians_accessor[i];
                                    auto rj = cartesians_accessor[j];
                                    auto rk = cartesians_accessor[k];

                                    float rij = 0;
                                    float rik = 0;
                                    float rjk = 0;

                                    for (int dim = 0; dim < 3; dim++){
                                        rij += (ri[dim] - rj[dim]) * (ri[dim] - rj[dim]);
                                        rjk += (rk[dim] - rj[dim]) * (rk[dim] - rj[dim]);
                                        rik += (rk[dim] - ri[dim]) * (rk[dim] - ri[dim]);
                                    }

                                    rij = sqrt(rij);
                                    rik = sqrt(rik);
                                    rjk = sqrt(rjk);

                                    float cos_v = (rij * rij + rik * rik - rjk * rjk) / 2 / rij / rik;

                                    g += G4(rij, rik, rjk, r_cutoff, eta, lambda, xi, cos_v);
                                }
                            }
                            break;
                        }
                        // G5
                        case 5:
                        {
                            for (int j = 0; j < n_atoms; j++){
                                for (int k = 0; k < n_atoms; k++){
                                    if (i == j || i == k || j == k){
                                        continue;
                                    }

                                    auto ri = cartesians_accessor[i];
                                    auto rj = cartesians_accessor[j];
                                    auto rk = cartesians_accessor[k];

                                    float rij = 0;
                                    float rik = 0;
                                    float rjk = 0;
                                    for (int dim = 0; dim < 3; dim++){
                                        rij += (ri[dim] - rj[dim]) * (ri[dim] - rj[dim]);
                                        rjk += (rk[dim] - rj[dim]) * (rk[dim] - rj[dim]);
                                        rik += (rk[dim] - ri[dim]) * (rk[dim] - ri[dim]);
                                    }
                                    rij = sqrt(rij);
                                    rik = sqrt(rik);
                                    rjk = sqrt(rjk);
                                    float cos_v = (rij * rij + rik * rik - rjk * rjk) / 2 / rij / rik;

                                    g += G5(rij, rik, rjk, r_cutoff, eta, lambda, xi, cos_v);
                                }
                            }
                            break;
                        }
                    }
                    // pass g and dg values of atom
                    g_total[i][g_type - 1] = g;
                }
            }
            // normalize g values
            g_total = torch::nn::functional::normalize(g_total,
                                            torch::nn::functional::NormalizeFuncOptions()
                                                                    .p(2.0)
                                                                    .dim(1));

            return g_total;
    }
}