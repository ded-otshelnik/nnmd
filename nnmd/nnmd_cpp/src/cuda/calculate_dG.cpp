#include "head.hpp"

#include "cuda/calculate_dG.hpp"
#include "cuda/cuda_header.hpp"
namespace cuda{
    // @brief Calculates dG for symmetry functions.
    // @param cartesians: atomic positions
    // @param r_cutoff: cutoff radius
    // @param eta: parameter of symmetry functions
    // @param rs: parameter of symmetry functions
    // @param lambda: parameter of symmetry functions
    // @param zeta: parameter of symmetry functions
    // @param h: step of coordinate-wise atom moving
    Tensor calculate_dG(const Tensor& cartesians, const Tensor& g, const float& r_cutoff,
                        const float& eta, const float& rs, const float& k,
                        const int& lambda, const float& zeta, const float& h){
        CHECK_INPUT(cartesians);
        CHECK_INPUT(g);

        auto opts = torch::TensorOptions()
                .dtype(torch::kFloat)
                .device(torch::kCUDA);
        Tensor cartesians_copy = cartesians;
        Tensor g_new;

        int n_structs = cartesians.size(0);
        int n_atoms = cartesians.size(1);
        int n_dims = cartesians.size(2);
        // number of symmetry functions
        // can be calculated from g
        int n_symm_funcs = g.size(2);

        // output forces
        Tensor forces = torch::zeros_like({cartesians}, opts);
        Tensor dG = torch::zeros({n_structs, n_dims, n_atoms, n_symm_funcs}, opts);

        for (int atom_struct = 0; atom_struct < n_structs; atom_struct++){
            // difference between new and actual energies
            for (int atom = 0; atom < n_atoms; atom++){
                for (int dim = 0; dim < n_dims; dim++){
                    // move atom along the dim with step h
                    cartesians_copy[atom_struct][atom][dim] += h;

                    // calculate new symmetry functions values
                    g_new = cuda::calculate_sf(cartesians_copy[atom_struct],
                                               r_cutoff, eta, rs, k, lambda, zeta);
                    // difference between new and actual g values
                    dG[atom_struct][dim][atom] = torch::sub(g_new[atom], g[atom_struct][atom]) / h;

                    // back to initial state
                    cartesians_copy[atom_struct][atom][dim] -= h; 
                }
            }
        }
        dG = dG.reshape({n_structs, n_atoms, n_symm_funcs, n_dims});
        return dG;
    }
}