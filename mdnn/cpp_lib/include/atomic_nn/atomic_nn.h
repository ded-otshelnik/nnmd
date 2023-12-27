#include "bind_src.h"

#pragma once

class AtomicNetImpl: public torch::nn::Module {
    private:
        vector<Linear> layers;
    public:
        AtomicNetImpl(int input_nodes, vector<int> hidden_nodes);
        ~AtomicNetImpl() {};
        Tensor forward(Tensor g);
        Tensor calculate_forces(Tensor cartesians, Tensor e_nn, int atom,
                const double& r_cutoff, const double& h, 
                const double& eta, const double& rs, const double& k,
                const int& lambda, const double& xi);
};

TORCH_MODULE(AtomicNet);

class AtomicNN: public AtomicNetImpl{
    public:
        AtomicNN(int input_nodes, vector<int> hidden_nodes): AtomicNetImpl(input_nodes, hidden_nodes) {}
        Tensor forward(Tensor g){
            return AtomicNetImpl::forward(g);
        };

        AtomicNN(const AtomicNN&) = default;
};