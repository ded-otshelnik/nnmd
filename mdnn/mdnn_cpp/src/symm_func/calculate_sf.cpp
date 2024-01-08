#include "head.h"
#include "symm_func/symmetric_functions.h"

// @brief Calculates symmetric functions and its derivatives
// @param cartesians: atomic positions
// @param r_cutoff: cutoff radius
// @param eta: parameter of symmetric functions
// @param rs: parameter of symmetric functions
// @param lambda: parameter of symmetric functions
// @param xi: parameter of symmetric functions
// @param dg_total: storage of output derivatives
Tensor calculate_sf(const Tensor &cartesians, const double& r_cutoff,
                const double& eta, const double& rs, const double& k_param,
                const int& lambda, const double& xi, Tensor &dg_total){

        auto opts = torch::TensorOptions()
            .dtype(torch::kDouble);
        // atoms amount
        int n_atoms = cartesians.sizes()[0];

        // output g values
        Tensor g_total = torch::zeros({n_atoms, 5}, opts);

        // get accessors to tensor that  

        auto cartesians_accessor = cartesians.accessor<double, 2>();
        auto g_total_accessor = g_total.accessor<double, 2>();
        auto dg_total_accessor = dg_total.accessor<double, 3>();

        // current g value
        double g;

        // loop by atoms
        for (int i = 0; i < n_atoms; i++){
            // loop by symmetric functions type
            for (int g_type = 1; g_type <= 5; g_type++){
                g = 0;
                // derivative of current symm func i for atom
                double dg[3] = {0, 0, 0};
                switch (g_type){
                    // G1
                    case 1:{
                        for (int j = 0; j < n_atoms; j++){
                            if (i == j){
                                continue;
                            }
                            auto ri = cartesians_accessor[i];
                            auto rj = cartesians_accessor[j];
                            double rij = 0;
                            for (int dim = 0; dim < 3; dim++){
                                rij += (ri[dim] - rj[dim]) * (ri[dim] - rj[dim]);
                            }

                            rij = sqrt(rij);
                            g += G1(rij, r_cutoff, dg);
                        }
                        break;
                    }
                    // G2
                    case 2:
                    {
                        for (int j = 0; j < n_atoms; j++){
                            if (i == j){
                                continue;
                            }
                            auto ri = cartesians_accessor[i];
                            auto rj = cartesians_accessor[j];
                            double rij = 0;
                            for (int dim = 0; dim < 3; dim++){
                                rij += (ri[dim] - rj[dim]) * (ri[dim] - rj[dim]);
                            }
                            rij = sqrt(rij);
                            g += G2(rij, r_cutoff, eta, rs, dg);
                        }
                        break;
                    }
                    // G3
                    case 3:
                    {
                        for (int j = 0; j < n_atoms; j++){
                            if (i == j){
                                continue;
                            }
                            auto ri = cartesians_accessor[i];
                            auto rj = cartesians_accessor[j];
                            double rij = 0;
                            for (int dim = 0; dim < 3; dim++){
                                rij += (ri[dim] - rj[dim]) * (ri[dim] - rj[dim]);
                            }
                            rij = sqrt(rij);
                            g += G3(rij, r_cutoff, k_param, dg);
                        }
                        break;
                    }
                    // G4
                    case 4:
                    {
                        for (int j = 0; j < n_atoms; j++){
                            for (int k = 0; k < n_atoms; k++){
                                if (i == j || i == k || j == k){
                                    continue;
                                }
                                auto ri = cartesians_accessor[i];
                                auto rj = cartesians_accessor[j];
                                auto rk = cartesians_accessor[k];

                                double rij = 0;
                                double rik = 0;
                                double rjk = 0;

                                for (int dim = 0; dim < 3; dim++){
                                    rij += (ri[dim] - rj[dim]) * (ri[dim] - rj[dim]);
                                    rjk += (rk[dim] - rj[dim]) * (rk[dim] - rj[dim]);
                                    rik += (rk[dim] - ri[dim]) * (rk[dim] - ri[dim]);
                                }

                                rij = sqrt(rij);
                                rik = sqrt(rik);
                                rjk = sqrt(rjk);

                                double cos_v = (rij * rij + rik * rik - rjk * rjk) / 2 / rij / rik;
                                double dcos_v[3] = {0, 0, 0};
                                
                                dcos_v[0] += (0.5 * (1 / rik + 1 / rij / rij * (rjk * rjk / rik - rik)));
                                dcos_v[1] += (0.5 * (1 / rij + 1 / rik / rik * (rjk * rjk / rij - rij)));
                                dcos_v[2] += (-rjk / rij / rik);

                                g += G4(rij, rik, rjk, r_cutoff, eta, lambda, xi, cos_v, dcos_v, dg);
                            }
                        }
                        break;
                    }
                    // G5
                    case 5:
                    {
                        for (int j = 0; j < n_atoms; j++){
                            for (int k = 0; k < n_atoms; k++){
                                if (i == j || i == k || j == k){
                                    continue;
                                }

                                auto ri = cartesians_accessor[i];
                                auto rj = cartesians_accessor[j];
                                auto rk = cartesians_accessor[k];

                                double rij = 0;
                                double rik = 0;
                                double rjk = 0;
                                for (int dim = 0; dim < 3; dim++){
                                    rij += (ri[dim] - rj[dim]) * (ri[dim] - rj[dim]);
                                    rjk += (rk[dim] - rj[dim]) * (rk[dim] - rj[dim]);
                                    rik += (rk[dim] - ri[dim]) * (rk[dim] - ri[dim]);
                                }
                                rij = sqrt(rij);
                                rik = sqrt(rik);
                                rjk = sqrt(rjk);
                                double cos_v = (rij * rij + rik * rik - rjk * rjk) / 2 / rij / rik;
                                double dcos_v[3] = {0, 0, 0};
                                dcos_v[0] += (0.5 * (1 / rik + 1 / rij / rij * (rjk * rjk / rik - rik)));
                                dcos_v[1] += (0.5 * (1 / rij + 1 / rik / rik * (rjk * rjk / rij - rij)));
                                dcos_v[2] += (-rjk / rij / rik);
                                g += G5(rij, rik, rjk, r_cutoff, eta, lambda, xi, cos_v, dcos_v, dg);
                            }
                        }
                        break;
                    }
                }
                // pass g and dg values of atom
                g_total[i][g_type - 1] = g;
                for (int dim = 0; dim < 3; dim++){
                    dg_total[i][g_type - 1][dim] = dg[dim];
                }
            }
        }
        // normalize g values
        g_total = torch::nn::functional::normalize(g_total,
                                         torch::nn::functional::NormalizeFuncOptions()
                                                                .p(2.0)
                                                                .dim(1));
        // normalize dg values
        dg_total = torch::nn::functional::normalize(dg_total, torch::nn::functional::NormalizeFuncOptions()
                                                                .p(2.0)
                                                                .dim(1));

        return g_total;
}