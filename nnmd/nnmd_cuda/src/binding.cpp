#include "head.h"

#include "symm_func/symmetric_functions.h"
#include "calculate_forces.h"

PYBIND11_MODULE(nnmd_cuda, module){
    module.doc() = R"pbdoc(PyTorch extention with Python/C++/CUDA bindings.)pbdoc";
    
    module.def("calculate_forces", &calculate_forces, R"pbdoc(
        Calculates forces of atomic system on iteration using AtomicNNs.
        TODO: check forces formula

        Args
            cartesians: atomic positions
            e_nn: actual calculated energies
            g: actual g values
            nets: list of AtomicNNs
            r_cutoff: cutoff radius
            eta: parameter of symmetric functions
            rs: parameter of symmetric functions
            lambda: parameter of symmetric functions
            xi: parameter of symmetric functions
            h: step of coordinate-wise atom moving
        )pbdoc",
        py::arg("cartesians"),
        py::arg("e_nn"),
        py::arg("g"),
        py::arg("nets"),
        py::arg("r_cutoff"),
        py::arg("eta"),
        py::arg("rs"),
        py::arg("k"),
        py::arg("lambda"),
        py::arg("xi"),
        py::arg("h")
    );

    module.def("calculate_sf", &calculate_sf, R"pbdoc(
        Calculates symmetric descriptors of atom structure

        Args:
            cartesians: atomic positions
            r_cutoff: cutoff radius
            eta: parameter of symmetric functions
            rs: parameter of symmetric functions
            lambda: parameter of symmetric functions
            xi: parameter of symmetric functions
            dg_total: storage of output derivatives
        
        )pbdoc",
        py::arg("cartesians"),
        py::arg("r_cutoff"),
        py::arg("eta"),
        py::arg("rs"),
        py::arg("k"),
        py::arg("lambda"),
        py::arg("xi")
    );
}