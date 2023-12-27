#include "bind_src.h"
#include "atomic_nn/atomic_nn.h"

#include <iostream>
using std::cout, std::endl;

#include <string>
using std::to_string;

AtomicNetImpl::AtomicNetImpl(int input_nodes, vector<int> hidden_nodes){
    Linear layer = nullptr;
    for (int i = 0; i < hidden_nodes.size(); i++){
        if (i == 0){
            layer = Linear(input_nodes, hidden_nodes[i]);
        }
        else{
            layer = Linear(hidden_nodes[i - 1], hidden_nodes[i]);
        }
        register_module("layer" + to_string(i), layer);
        layers.push_back(layer);
    }
    layer = Linear(hidden_nodes[hidden_nodes.size() - 1], 1);
    register_module("layer" + to_string(hidden_nodes.size()), layer);
    layers.push_back(layer); 
}

Tensor AtomicNetImpl::forward(Tensor g)
{
    Tensor x = g;
    for (int i = 0; i < layers.size() - 1; i++){
        x = layers[i]->forward(x);
        x = torch::sigmoid(x);
    }
    x = layers[layers.size() - 1]->forward(x);
    return x;
}

Tensor AtomicNetImpl::calculate_forces(Tensor cartesians, Tensor e_nn, int atom,
        const double& r_cutoff, const double& h, 
        const double& eta, const double& rs, const double& k,
        const int& lambda, const double& xi){

    auto opts = torch::TensorOptions()
            .dtype(torch::kDouble);
            
    vector< vector <double>> cartesians_copy;
    int n_atoms = cartesians.sizes()[0], dims = cartesians.sizes()[1];
    for (int i = 0; i < n_atoms; i++){
        vector <double> atom;
        for (int j = 0; j < dims; j++){
            atom.push_back(cartesians[i][j].item<double>());
        }
        cartesians_copy.push_back(atom);
    }

    vector< vector <double>> dg_new;

    Tensor forces = torch::zeros({3}, opts);

    for (int dim = 0; dim < 3; dim++){
        cartesians_copy[atom][dim] += h;

        auto g_new = calculate_sf(cartesians_copy, r_cutoff,
                                        eta, rs, k, lambda, xi, dg_new);
        
        Tensor g_new_tensor = torch::from_blob(g_new.data(),
                                            {(int)g_new.size(), (int)g_new[0].size()}).to(torch::kFloat);

        Tensor dg_new_tensor = torch::from_blob(dg_new.data(),
                                            {(int)dg_new.size(), (int)dg_new[0].size()});
        g_new_tensor.requires_grad_(true);
        Tensor e_new = forward(g_new_tensor[atom]);
        cout << e_new << endl;
        cout << torch::sub(e_new, e_nn[atom]) << endl;
        cout << dg_new_tensor[atom][dim].unsqueeze(0) << endl;
        forces[dim] = torch::sub(forces[dim],
                               torch::matmul(torch::sub(e_new, e_nn[atom]),
                                             dg_new_tensor[atom][dim].unsqueeze(0)));
        cartesians_copy[atom][dim] -= h;            
    }
    return forces;
}