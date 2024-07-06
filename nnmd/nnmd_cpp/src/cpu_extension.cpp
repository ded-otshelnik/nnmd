#include "head.hpp"
#include "cpu/symmetric_functions.hpp"
#include "cpu/calculate_forces.hpp"
#include "cpu/symmetric_functions.hpp"

using namespace cpu;

void init_cpu_module(py::module_& module){
    module.def("G1", &cpu::G1,
        R"pbdoc(
        )pbdoc",
        py::arg("rij"),
        py::arg("rc")
    );

    module.def("G2", &cpu::G2,
        R"pbdoc(
        )pbdoc",
        py::arg("rij"),
        py::arg("rc"),
        py::arg("eta"),
        py::arg("rs")
    );

    module.def("G3", &cpu::G3,
        R"pbdoc(
        )pbdoc",
        py::arg("rij"),
        py::arg("rc"),
        py::arg("k")
    );

    module.def("G4", &cpu::G4,
        R"pbdoc(
        )pbdoc",
        py::arg("rij"),
        py::arg("rik"),
        py::arg("rjk"),
        py::arg("rc"),
        py::arg("eta"),
        py::arg("lambda"),
        py::arg("xi"),
        py::arg("cos_v")
    );

    module.def("G5", &cpu::G5,
        R"pbdoc(
        )pbdoc",
        py::arg("rij"),
        py::arg("rik"),
        py::arg("rjk"),
        py::arg("rc"),
        py::arg("eta"),
        py::arg("lambda"),
        py::arg("xi"),
        py::arg("cos_v")
    );

    module.def("calculate_sf", &cpu::calculate_sf, R"pbdoc(
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


    module.def("calculate_forces", &cpu::calculate_forces, R"pbdoc(
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