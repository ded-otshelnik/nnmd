#include "cpu/calculate_dG.hpp"

namespace cpu{
    // @brief Calculates dG for symmetry functions.
    // @param cartesians: atomic positions
    // @param g: symmetry functions values
    // @param features: symmetry functions set
    // @param params: parameters of symmetry functions
    // @param h: step of coordinate-wise atom moving
    Tensor calculate_dG(const Tensor& cartesians, const Tensor& g, 
                        const vector<int>& features, const vector<vector<float>>& params,
                        const float h){

        auto opts = torch::TensorOptions()
                .dtype(torch::kFloat)
                .device(torch::kCPU);
        Tensor cartesians_copy = cartesians;
        Tensor g_new;

        int n_structs = cartesians.size(0);
        int n_atoms = cartesians.size(1);
        int n_dims = cartesians.size(2);
        // number of symmetry functions
        // can be calculated from g
        int n_features = features.size();

        // output forces
        Tensor dG = torch::zeros({n_structs, n_dims, n_atoms, n_features}, opts);

        for (int atom_struct = 0; atom_struct < n_structs; atom_struct++){
            // difference between new and actual energies
            for (int atom = 0; atom < n_atoms; atom++){
                for (int dim = 0; dim < n_dims; dim++){
                    // move atom along the dim with step h
                    cartesians_copy[atom_struct][atom][dim] += h;

                    // calculate new symmetry functions values
                    g_new = cpu::calculate_sf(cartesians_copy[atom_struct],
                                            features, params);
                    // difference between new and actual g values
                    dG[atom_struct][dim][atom] = torch::sub(g_new[atom], g[atom_struct][atom]) / h;

                    // back to initial state
                    cartesians_copy[atom_struct][atom][dim] -= h; 
                }
            }
        }
        dG = dG.reshape({n_structs, n_atoms, n_features, n_dims});
        return dG;
    }
}