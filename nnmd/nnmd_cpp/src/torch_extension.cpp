#include "head.hpp"
#include "torch/calculate_sf.hpp"

void init_torch_module(py::module_& module){
    module.def("calculate_input", &calculate_input, R"pbdoc(
        Calculates input for neural network

        Args:
            cartesians: atomic positions
            symm_funcs_data: symmetry functions data

        Returns:
            Input for neural network
        )pbdoc",
        py::arg("cartesians"),
        py::arg("features"),
        py::arg("params")
    );
}