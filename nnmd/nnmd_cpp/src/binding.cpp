#include "cpu/symmetric_functions.hpp"
#include "cpu/calculate_forces.hpp"
#include "cuda/symmetric_functions.hpp"
#include "cuda/calculate_dG.hpp"

PYBIND11_MODULE(nnmd_cpp, module){
    module.doc() = R"pbdoc(PyTorch extention with CUDA/C++ bindings.)pbdoc";

    auto cpu_module = module.def_submodule("cpu", "CPU extension");
    init_cpu_module(cpu_module);

    auto cuda_module = module.def_submodule("cuda", "CUDA extension");
    init_cuda_module(cuda_module);
}