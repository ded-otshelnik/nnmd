#include "head.h"

#include "symm_func/symmetric_functions.h"
#include "atomic_nn/atomic_nn.h"

Tensor calculate_forces(const Tensor cartesians, Tensor e_nn, const py::list& nets, const double& r_cutoff,
                const double& eta, const double& rs, const double& k,
                const int& lambda, const double& xi, const double& h){
    
    auto opts = torch::TensorOptions()
            .dtype(torch::kDouble);

    int n_atoms = cartesians.sizes()[0];
    Tensor cartesians_copy = cartesians;

    Tensor forces = torch::zeros(cartesians.sizes(), opts);
    for (int atom = 0; atom < n_atoms; atom++){
        for (int dim = 0; dim < 3; dim++){
            cartesians_copy[atom][dim] += h;
            Tensor dg_new = torch::zeros({n_atoms, 5, 3}, opts);
            Tensor g_new = calculate_sf(cartesians_copy, r_cutoff,
                                            eta, rs, k, lambda, xi, dg_new);
            cout << dg_new << endl;
            Tensor e_new = torch::zeros(n_atoms, opts);
            for (int i = 0; i < n_atoms; i++){
                py::object obj = nets[i];
                if (py::isinstance<AtomicNN>(obj)){
                    AtomicNN nn = obj.cast<AtomicNN>();
                    e_new[i] = nn.forward(g_new[i]);
                }
                auto dE = torch::sub(e_new[i], e_nn[i]);
                cout << dE << endl; 
                for (int g_type = 0; g_type < 5; g_type++){
                    for (int j = 0; j < 3; j++){
                        forces[atom][dim] -= dE * dg_new[i][g_type][j];
                    }
                }
            }
            cartesians_copy[atom][dim] -= h;            
        }
    }
    cout << forces << endl;
    return forces;
}