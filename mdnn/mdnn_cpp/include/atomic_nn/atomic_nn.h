#include "head.h"

#include "symm_func/symmetric_functions.h"

#pragma once
class AtomicNetImpl: public torch::nn::Module {
    private:
        vector<Linear> layers;
    public:
        AtomicNetImpl(int input_nodes, vector<int> hidden_nodes);
        ~AtomicNetImpl() {};
        Tensor forward(Tensor g);
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

Tensor calculate_forces(const Tensor cartesians, Tensor e_nn, const py::list& nets, const double& r_cutoff,
                const double& eta, const double& rs, const double& k,
                const int& lambda, const double& xi, const double& h);