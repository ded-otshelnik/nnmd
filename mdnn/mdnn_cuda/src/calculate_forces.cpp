#include "calculate_forces.h"

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

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
    CHECK_INPUT(cartesians);
    CHECK_INPUT(e_nn);
    CHECK_INPUT(g);

    auto opts = torch::TensorOptions()
            .dtype(torch::kFloat)
            .device(torch::kCUDA);

    Tensor cartesians_copy = cartesians.clone();
    // atoms amount
    int n_structs = cartesians.size(0);
    int n_atoms = cartesians.size(1);

    // output forces
    Tensor forces = torch::zeros(cartesians.sizes(), opts);

    for (int atom_struct = 0; atom_struct < n_structs; atom_struct++){
        for (int atom = 0; atom < n_atoms; atom++){
            for(int dim = 0; dim < 3; dim++){
                // move atom along the dim in step h
                cartesians_copy[atom_struct][atom][dim] += h;

                // calculate new symmetric functions values
                Tensor g_new = calculate_sf(cartesians_copy[atom_struct], r_cutoff,
                                                    eta, rs, k, lambda, xi);
                        
                Tensor e_new = torch::zeros(n_atoms, opts);
                for (int i = 0; i < n_atoms; i++){
                    // AtomicNN of i atom
                    py::object obj = nets[i];
                    // recalculate energies according new g values
                    auto temp = obj(g_new[i]);
                    e_new[i] = temp.cast<Tensor>().squeeze();
                }
                // difference between new and actual energies
                auto dE = torch::sub(e_new, e_nn[atom_struct]);
                for (int i = 0; i < n_atoms; i++){
                    // calculate dim component of force for atom
                    for (int g_type = 0; g_type < 5; g_type++){
                        forces[atom_struct][atom][dim] -= dE[i] * (g_new[i][g_type] - g[atom_struct][i][g_type]) / h;
                    }
                }
                // back to initial state
                cartesians_copy[atom_struct][atom][dim] -= h; 
            }
        }
    }

    return forces;
}