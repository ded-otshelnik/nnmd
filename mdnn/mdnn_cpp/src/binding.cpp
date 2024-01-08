#include "head.h"

#include "atomic_nn/atomic_nn.h"
#include "symm_func/symmetric_functions.h"

PYBIND11_MODULE(mdnn_cpp, module){
    module.doc() = R"pbdoc(PyTorch extention with Python/C++ bindings.)pbdoc";

    torch::python::bind_module<AtomicNN>(module, "AtomicNN")
        .def(py::init<const int &, const vector<int>>(),
                         py::arg("input_layers"),
                         py::arg("hidden_layers"))
        .def("forward", &AtomicNN::forward,
                        py::arg("inputs")    
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


    module.def("calculate_forces", &calculate_forces, "Calculates forces of atom structure",
                                    py::arg("cartesians"),
                                    py::arg("e_nn"),
                                    py::arg("nets"),
                                    py::arg("r_cutoff"),
                                    py::arg("eta"),
                                    py::arg("rs"),
                                    py::arg("k"),
                                    py::arg("lambda"),
                                    py::arg("xi"),
                                    py::arg("h")
    );
}