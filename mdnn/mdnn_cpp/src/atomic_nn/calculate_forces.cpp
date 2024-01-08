#include "head.h"

#include "symm_func/symmetric_functions.h"
#include "atomic_nn/atomic_nn.h"

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
Tensor calculate_forces(const Tensor cartesians, Tensor e_nn, const py::list& nets, const double& r_cutoff,
                const double& eta, const double& rs, const double& k,
                const int& lambda, const double& xi, const double& h){
    
    auto opts = torch::TensorOptions()
            .dtype(torch::kDouble);
    // atoms amount
    int n_atoms = cartesians.sizes()[0];
    Tensor cartesians_copy = cartesians;

    // output forces
    Tensor forces = torch::zeros(cartesians.sizes(), opts);
    // loop by atoms
    for (int atom = 0; atom < n_atoms; atom++){
        // loop by dimentions
        for (int dim = 0; dim < 3; dim++){
            // move atom along the dim in step h
            cartesians_copy[atom][dim] += h;

            // calculate new symmetric functions values
            // and its derivatives
            Tensor dg_new = torch::zeros({n_atoms, 5, 3}, opts);
            Tensor g_new = calculate_sf(cartesians_copy, r_cutoff,
                                            eta, rs, k, lambda, xi, dg_new);
            
            Tensor e_new = torch::zeros(n_atoms, opts);
            for (int i = 0; i < n_atoms; i++){
                // AtomicNN of i atom
                py::object obj = nets[i];
                if (py::isinstance<AtomicNN>(obj)){
                    AtomicNN nn = obj.cast<AtomicNN>();
                    // recalculate energies according new g values
                    e_new[i] = nn.forward(g_new[i]);
                }
                // difference between new and actual energies
                auto dE = torch::sub(e_new[i], e_nn[i]);
                // calculate dim component of force for atom
                for (int g_type = 0; g_type < 5; g_type++){
                    for (int j = 0; j < 3; j++){
                        forces[atom][dim] -= dE * dg_new[i][g_type][j] * g_new[i][g_type];
                    }
                }
            }
            // back to initial state
            cartesians_copy[atom][dim] -= h;            
        }
    }
    return forces;
}