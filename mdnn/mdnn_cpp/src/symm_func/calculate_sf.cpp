#include "head.h"
#include "symm_func/symmetric_functions.h"

Tensor calculate_sf(const Tensor &cartesians, const double& r_cutoff,
                const double& eta, const double& rs, const double& k_param,
                const int& lambda, const double& xi, Tensor &dg_total){

        auto opts = torch::TensorOptions()
            .dtype(torch::kDouble);

        int n_atoms = cartesians.sizes()[0];
        Tensor g_total = torch::zeros({n_atoms, 5}, opts);

        auto cartesians_accessor = cartesians.accessor<double, 2>();
        auto g_total_accessor = g_total.accessor<double, 2>();
        auto dg_total_accessor = dg_total.accessor<double, 3>();

        double g;
        for (int i = 0; i < n_atoms; i++){
            for (int g_type = 1; g_type <= 5; g_type++){
                g = 0;
                double dg[3] = {0, 0, 0};
                switch (g_type)
                {
                case 1:
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
                case 2:
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
                case 3:
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
                case 4:
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
                case 5:
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
                g_total[i][g_type - 1] = g;
                for (int dim = 0; dim < 3; dim++){
                    dg_total[i][g_type - 1][dim] = dg[dim];
                }
            }
        }

        g_total = torch::nn::functional::normalize(g_total,
                                         torch::nn::functional::NormalizeFuncOptions()
                                                                .p(2.0)
                                                                .dim(1));

        dg_total = torch::nn::functional::normalize(dg_total, torch::nn::functional::NormalizeFuncOptions()
                                                                .p(2.0)
                                                                .dim(1));

        return g_total;
}