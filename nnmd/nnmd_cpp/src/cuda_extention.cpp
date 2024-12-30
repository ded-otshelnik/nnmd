#include "cuda/symmetric_functions.hpp"
#include "cuda/calculate_dG.hpp"
#include "cuda/calculate_gdf.hpp"

using namespace cuda;

void init_cuda_module(py::module_& module){

    module.def("calculate_sf", &cuda::calculate_sf, R"pbdoc(
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
        py::arg("kappa"),
        py::arg("lambda"),
        py::arg("zeta")
    );


    module.def("calculate_dG", &cuda::calculate_dG, R"pbdoc(
        Calculates dG for atomic system.

        Args
            cartesians: atomic positions
            g: actual g values
            r_cutoff: cutoff radius
            eta: parameter of symmetric functions
            rs: parameter of symmetric functions
            lambda: parameter of symmetric functions
            xi: parameter of symmetric functions
            h: step of coordinate-wise atom moving
        )pbdoc",
        py::arg("cartesians"),
        py::arg("g"),
        py::arg("r_cutoff"),
        py::arg("eta"),
        py::arg("rs"),
        py::arg("kappa"),
        py::arg("lambda"),
        py::arg("zeta"),
        py::arg("h")
    );

    module.def("calculate_gdf", &cuda::calculate_gdf, R"pbdoc(
        Calculates gdf values for atomic structures

        Args:
            refs: reference atomic structures
            num_refs: number of reference structures
            num_targets: number of target structures
            num_features: number of features
            sigma: standard deviance value
        )pbdoc",
        py::arg("refs"),
        py::arg("num_refs"),
        py::arg("num_targets"),
        py::arg("num_features"),
        py::arg("sigma")
    );
}