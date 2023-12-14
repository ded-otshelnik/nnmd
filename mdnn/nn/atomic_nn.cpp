
#include <torch/torch.h>
#include <torch/extension.h>
#include <vector>

#include "../symm_func/symmetric_functions.h"

using std::vector;
using torch::Tensor;
using namespace torch::nn;

TORCH_LIBRARY(atomic_nn, m){
    m.class_<AtomicNN>("AtomicNN")
     .def(torch::init())
     .def("compile", &AtomicNN::compile)
     .def("forward", &AtomicNN::forward)
     .def("calculate_forces", &AtomicNN::calculate_forces);
};

class AtomicNN: torch::nn::Module {
    private:
        torch::nn::Sequential layers;
    public:
        AtomicNN() {
            torch::nn::Module(AtomicNN);
            layers = register_module("Atomic NN", torch::nn::Sequential());
        }
        
        void compile(int input_nodes, vector<int> hidden_nodes){
            torch::nn::ModuleList list;
            for (int i = 0; i < hidden_nodes.size(); i++){
                if (i == 0){
                    list->push_back(Linear(input_nodes, hidden_nodes[i]));
                    list->push_back(Sigmoid());
                }
                else{
                    list->push_back(Linear(hidden_nodes[i - 1], hidden_nodes[i]));
                    list->push_back(Sigmoid());
                }
            }
            list->push_back(Linear(hidden_nodes[hidden_nodes.size() - 1], 1));
            
            for (auto layer: *list){
                layers->push_back(layer);
            }
        }

        Tensor forward(Tensor g)
        {
            return layers->forward(g);
        }

        Tensor calculate_forces(Tensor cartesians, Tensor e_nn, int num_atoms,
                const double& r_cutoff, const double& h, 
                const double& eta, const double& rs, const double& k,
                const int& lambda, const double& xi){

            vector< vector <double>> cartesians_copy(cartesians.data_ptr<double>(),
                                                     cartesians.data_ptr<double>() + cartesians.numel());    
            vector< vector <double>> dg_new;

            Tensor forces = torch::zeros({(int)cartesians_copy.size(), 3});

            for (int atom = 0; atom < num_atoms; atom++){
                for (int dim = 0; dim < 3; dim++){
                    cartesians_copy[atom][dim] += h;
                    auto g_new = calculate_sf(cartesians_copy, r_cutoff,
                                                      eta, rs, k, lambda, xi, dg_new);
                    Tensor g_new_tensor = torch::from_blob(g_new.data(),
                                                           {(int)g_new.size(), (int)g_new[0].size()});

                    Tensor e_new = forward(g_new_tensor);
                    forces[atom][dim] = torch::sub(forces[atom][dim],
                                                   (e_new - e_nn[atom]) * dg_new[atom][dim]);
                    cartesians_copy[atom][dim] -= h;            
                }
            }
            return forces;
        }
};