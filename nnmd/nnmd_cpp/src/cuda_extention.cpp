#include "cuda/symmetric_functions.hpp"
#include "cuda/calculate_dG.hpp"
#include "cuda/calculate_gdf.hpp"

using namespace cuda;

void init_cuda_module(py::module_& module){

    module.def("calculate_sf", &cuda::calculate_sf, R"pbdoc(
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


    module.def("calculate_dG", &cuda::calculate_dG, R"pbdoc(
        Calculates dG for atomic system.

        Args
            cartesians: atomic positions
            g: actual g values
            features: symmetry functions set
            params: parameters of symmetry functions
            h: step of coordinate-wise atom moving

        Returns:
            dG values
        )pbdoc",
        py::arg("cartesians"),
        py::arg("g"),
        py::arg("features"),
        py::arg("params"),
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