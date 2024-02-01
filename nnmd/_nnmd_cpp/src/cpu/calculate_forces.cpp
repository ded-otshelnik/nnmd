#include "head.hpp"

#include "cpu/symmetric_functions.hpp"
#include "cpu/calculate_forces.hpp"

namespace cpu{
    
    const int n_dims = 3;

    // @brief Calculates forces of atomic system on iteration using AtomicNNs.
    // TODO: check forces formula
    // @param cartesians: atomic positions
    // @param e_nn: actual calculated energies
    // @param nets: list of AtomicNNs
    // @param r_cutoff: cutoff radius
    // @param eta: parameter of symmetric functions
    // @param rs: parameter of symmetric functions
    // @param lambda: parameter of symmetric functions
    // @param xi: parameter of symmetric functions
    // @param h: step of coordinate-wise atom moving
    Tensor calculate_forces(const Tensor& cartesians, const Tensor& e_nn, const Tensor& g,
                    const py::list& nets, const float& r_cutoff,
                    const float& eta, const float& rs, const float& k,
                    const int& lambda, const float& xi, const float& h){
        
        auto opts = torch::TensorOptions()
                .dtype(torch::kFloat);
        // atoms amount
        int n_structs = cartesians.size(0);
        int n_atoms = cartesians.size(1);

        Tensor cartesians_copy = cartesians;

        // output forces
        Tensor forces = torch::zeros(cartesians.sizes(), opts);
        // loop by atom structs
        for (int atom_struct = 0; atom_struct < n_structs; atom_struct++){
            // loop by atoms
            for (int atom = 0; atom < n_atoms; atom++){
                // loop by dimentions
                for (int dim = 0; dim < n_dims; dim++){
                    // move atom along the dim in step h
                    cartesians_copy[atom_struct][atom][dim] += h;

                    // calculate new symmetric functions values
                    Tensor g_new = calculate_sf(cartesians_copy[atom_struct], r_cutoff,
                                                    eta, rs, k, lambda, xi);
                    // difference between new and actual g values
                    auto dG = torch::sub(g_new, g[atom_struct]) / h;

                    Tensor e_new = torch::empty(n_atoms, opts);

                    for (int i = 0; i < n_atoms; i++){
                        // AtomicNN of i atom
                        py::object obj = nets[i];
                        // recalculate energies according new g values
                        auto temp = obj(g_new[i]);
                        e_new[i] = temp.cast<Tensor>().squeeze();
                    }
                    // difference between new and actual energies
                    auto dE = torch::sub(e_new, e_nn[atom_struct]);

                    forces[atom_struct][atom][dim] -= torch::sum(torch::matmul(dE, dG));

                    // back to initial state
                    cartesians_copy[atom_struct][atom][dim] -= h;            
                }
            }
        
        }
        return forces;
    }
}