#include "head.hpp"

#include "cpu/calculate_forces.hpp"

namespace cpu{

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
        Tensor cartesians_copy = cartesians;
        Tensor g_new, e_new, dG, dE;

        // atoms amount
        int n_structs = cartesians.size(0);
        int n_atoms = cartesians.size(1);
        int n_dims = cartesians.size(2);

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
                    g_new = calculate_sf(cartesians_copy[atom_struct], r_cutoff,
                                                    eta, rs, k, lambda, xi);
                    // difference between new and actual g values
                    dG = torch::sub(g_new, g[atom_struct]);

                    // compute new energies
                    e_new = torch::empty({n_atoms, 1}, opts);
                    for (int i = 0; i < n_atoms; i++){
                        // AtomicNN of i atom
                        py::object obj = nets[i];
                        // recalculate energies according new g values
                        e_new[i] = obj(g_new[i]).cast<Tensor>();
                    }

                    // difference between new and actual energies
                    dE = torch::sub(e_new, e_nn[atom_struct]);
                    
                    forces[atom_struct][atom][dim] -= torch::sum(torch::matmul(dE, dG));

                    // back to initial state
                    cartesians_copy[atom_struct][atom][dim] -= h;            
                }
            }
        
        }
        return forces;
    }
}