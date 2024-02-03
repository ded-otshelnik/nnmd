#include "head.hpp"
#include "cpu/symmetric_functions.hpp"
#include "cpu/calculate_forces.hpp"
#include "cuda/symmetric_functions.hpp"
#include "cuda/calculate_forces.hpp"

PYBIND11_MODULE(nnmd_cpp, module){
    module.doc() = R"pbdoc(PyTorch extention with CUDA/C++ bindings.)pbdoc";

    auto cpu_module = module.def_submodule("cpu", "CPU extension");
    cpu_module.def("calculate_sf", &cpu::calculate_sf, R"pbdoc(
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


    cpu_module.def("calculate_forces", &cpu::calculate_forces, R"pbdoc(
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
    auto cuda_module = module.def_submodule("cuda", "CUDA extension");
    cuda_module.def("calculate_sf", &cuda::calculate_sf, R"pbdoc(
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


    cuda_module.def("calculate_forces", &cuda::calculate_forces, R"pbdoc(
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
}