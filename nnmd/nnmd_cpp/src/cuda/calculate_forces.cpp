#include "head.hpp"

#include "cuda/calculate_forces.hpp"
#include "cuda/cuda_header.hpp"

namespace cuda{
    // @brief Calculates forces of atomic system on iteration using AtomicNNs.
    // @param cartesians: atomic positions
    // @param e_nn: actual calculated energies
    // @param nets: list of AtomicNNs
    // @param r_cutoff: cutoff radius
    // @param eta: parameter of symmetric functions
    // @param rs: parameter of symmetric functions
    // @param lambda: parameter of symmetric functions
    // @param xi: parameter of symmetric functions
    // @param h: step of coordinate-wise atom moving
    Tensor calculate_forces(const Tensor& cartesians, const Tensor& g, const Tensor& dE, 
                    /*const py::list& nets,*/ const float& r_cutoff,
                    const float& eta, const float& rs, const float& k,
                    const int& lambda, const float& xi, const float& h){
        CHECK_INPUT(cartesians);
        CHECK_INPUT(dE);
        CHECK_INPUT(g);

        auto opts = torch::TensorOptions()
                .dtype(torch::kFloat)
                .device(torch::kCUDA);
        Tensor cartesians_copy = cartesians;
        Tensor g_new, e_new, dG;

        // atoms amount
        int n_structs = cartesians.size(0);
        int n_atoms = cartesians.size(1);
        int n_dims = cartesians.size(2);

        // output forces
        Tensor forces = torch::zeros_like({cartesians}, opts);
        for (int atom_struct = 0; atom_struct < n_structs; atom_struct++){
            // difference between new and actual energies
            for (int atom = 0; atom < n_atoms; atom++){
                for (int dim = 0; dim < n_dims; dim++){
                    // move atom along the dim with step h
                    cartesians_copy[atom_struct][atom][dim] += h;

                    // calculate new symmetric functions values
                    g_new = cuda::calculate_sf(cartesians_copy[atom_struct],
                                                r_cutoff, eta, rs, k, lambda, xi);

                    // difference between new and actual g values
                    dG = torch::sub(g_new, g[atom_struct]);

                    forces[atom_struct][atom][dim] -= torch::sum(torch::mul(dE, dG));

                    // back to initial state
                    cartesians_copy[atom_struct][atom][dim] -= h; 
                }
            }
        }
        return forces;
    }
}