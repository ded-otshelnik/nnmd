#include "cpu/symmetric_functions.hpp"

namespace cpu{

    // @brief Calculates symmetric functions and its derivatives
    // @param cartesians: atomic positions
    // @param r_cutoff: cutoff radius
    // @param eta: parameter of symmetric functions
    // @param rs: parameter of symmetric functions
    // @param lambda: parameter of symmetric functions
    // @param xi: parameter of symmetric functions
    Tensor calculate_sf(const Tensor& cartesians, const vector<int>& features, const vector<vector<float>>& params){

            auto opts = torch::TensorOptions()
                .dtype(torch::kFloat);
            // atoms amount
            int n_atoms = cartesians.size(0);
            int n_features = features.size();

            // output g values
            Tensor g_total = torch::zeros({n_atoms, n_features}, opts);

            // get accessors to tensors 
            auto cartesians_accessor = cartesians.accessor<float, 2>();

            // current g value
            float g;

            // loop by atoms
            for (int i = 0; i < n_atoms; i++){
                // loop by symmetric functions type
                for (int feature_index = 0; feature_index < n_features; feature_index++){
                    g = 0;
                    switch (features[feature_index]){
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
                                g += G1(rij, params[feature_index][0]);
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
                                g += G2(rij, params[feature_index][0], params[feature_index][1], params[feature_index][2]);
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
                                g += G3(rij, params[feature_index][0], params[feature_index][1]);
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

                                    g += G4(rij, rik, rjk,
                                             params[feature_index][0], params[feature_index][1], 
                                             params[feature_index][2], params[feature_index][3], cos_v);
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

                                    g += G5(rij, rik, rjk,
                                             params[feature_index][0], params[feature_index][1], 
                                             params[feature_index][2], params[feature_index][3], cos_v);
                                }
                            }
                            break;
                        }
                    }
                    // pass g value of atom
                    g_total[i][feature_index - 1] = g;
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