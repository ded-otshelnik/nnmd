#include "bind_src.h"
#include "atomic_nn/atomic_nn.h"
#include "symm_func/symmetric_functions.h"

#include <pybind11/pybind11.h>
#include <pybind11/attr.h>
#include <pybind11/functional.h>
using namespace torch::nn;
namespace py = pybind11;

PYBIND11_MODULE(mdnn_cpp, module){
    py::doc("PyTorch network extention with Python/C++ bindings.");

    torch::python::bind_module<AtomicNN>(module, "AtomicNN")
        .def(py::init<const int &, const vector<int>>(),
                         py::arg("input_layers"),
                         py::arg("hidden_layers"))
        .def("forward", &AtomicNN::forward,
                        py::arg("inputs"))
        .def("calculate_forces", &AtomicNN::calculate_forces,
                                 py::arg("cartesians"),
                                 py::arg("e_nn"),
                                 py::arg("atom"),
                                 py::arg("r_cutoff"),
                                 py::arg("h"),
                                 py::arg("eta"),
                                 py::arg("rs"),
                                 py::arg("k"),
                                 py::arg("lambda"),
                                 py::arg("xi")          
    );
        
    module.def("calculate_sf", &calculate_sf, "Calculates symmetric descriptors of atom structure",
                                    py::arg("cartesians"),
                                    py::arg("r_cutoff"),
                                    py::arg("eta"),
                                    py::arg("rs"),
                                    py::arg("k"),
                                    py::arg("lambda"),
                                    py::arg("xi"),
                                    py::arg("dg_total")
    );
}