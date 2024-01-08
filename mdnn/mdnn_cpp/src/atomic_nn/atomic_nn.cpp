#include "head.h"

#include "atomic_nn/atomic_nn.h"

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