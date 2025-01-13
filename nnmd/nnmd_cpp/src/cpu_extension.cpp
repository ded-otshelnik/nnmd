#include "head.hpp"
#include "cpu/symmetric_functions.hpp"
#include "cpu/calculate_dG.hpp"

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
        py::arg("kappa")
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
        py::arg("zeta"),
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
        py::arg("zeta"),
        py::arg("cos_v")
    );

    module.def("calculate_sf", &cpu::calculate_sf, R"pbdoc(
        Calculates symmetric descriptors of atom structure

        Args:
            cartesians: atomic positions
            features: symmetry functions set
            params: parameters of symmetry functions

        Returns:
            Symmetric descriptors of atom structure
        )pbdoc",
        py::arg("cartesians"),
        py::arg("features"),
        py::arg("params")
    );

    module.def("calculate_dG", &cpu::calculate_dG, R"pbdoc(
        Calculates dG for symmetry functions

        Args:
            cartesians: atomic positions
            g: symmetry functions values
            features: symmetry functions set
            params: parameters of symmetry functions
            h: step of coordinate-wise atom moving

        Returns:
            dG for symmetry functions
        )pbdoc",
        py::arg("cartesians"),
        py::arg("g"),
        py::arg("features"),
        py::arg("params"),
        py::arg("h")
    );
}