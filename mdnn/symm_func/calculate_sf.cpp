#include "symmetric_functions.h"

vector< vector<double> > calculate_sf(const vector< vector<double> > &cartesians, const double& r_cutoff,
                const double& eta, const double& rs, const double& k_param,
                const int& lambda, const double& xi, vector< vector<double> >& dg_total){
        int n_atoms = cartesians.size();
        vector< vector<double> > g_total;
        vector<double> ri, rj, rk;
        for (int i = 0; i < n_atoms; i++){
            vector<double> g_atom, dg_atom;
            double out_dg[3] = {0, 0, 0};
            dg_atom = {0, 0, 0};
            double g;
            for (int g_type = 1; g_type <= 5; g_type++){
                switch (g_type)
                {
                case 1:
                    g = 0;
                    for (int j = 0; j < n_atoms; j++){
                        if (i == j){
                            continue;
                        }
                        ri = cartesians[i];
                        rj = cartesians[j];
                        double rij = 0;
                        for (int dim = 0; dim < 3; dim++){
                            rij += (ri[dim] - rj[dim]) * (ri[dim] - rj[dim]);
                        }
                        rij = sqrt(rij);
                        g += G1(rij, r_cutoff, out_dg);
                    }
                    break;
                case 2:
                    g = 0;
                    for (int j = 0; j < n_atoms; j++){
                        if (i == j){
                            continue;
                        }
                        ri = cartesians[i];
                        rj = cartesians[j];
                        double rij = 0;
                        for (int dim = 0; dim < 3; dim++){
                            rij += (ri[dim] - rj[dim]) * (ri[dim] - rj[dim]);
                        }
                        rij = sqrt(rij);
                        g += G2(rij, r_cutoff, eta, rs, out_dg);
                    }
                    break;
                case 3:
                    g = 0;
                    for (int j = 0; j < n_atoms; j++){
                        if (i == j){
                            continue;
                        }
                        ri = cartesians[i];
                        rj = cartesians[j];
                        double rij = 0;
                        for (int dim = 0; dim < 3; dim++){
                            rij += (ri[dim] - rj[dim]) * (ri[dim] - rj[dim]);
                        }
                        rij = sqrt(rij);
                        g += G3(rij, r_cutoff, k_param, out_dg);
                    }
                    break;
                case 4:
                    g = 0;
                    for (int j = 0; j < n_atoms; j++){
                        for (int k = 0; k < n_atoms; k++){
                            if (i == j || i == k || j == k){
                                continue;
                            }
                            ri = cartesians[i];
                            rj = cartesians[j];
                            rk = cartesians[k];
                            double rij = 0, rik = 0, rjk = 0;
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
                            dcos_v[0] = 0.5 * (1 / rik + 1 / rij / rij * (rjk * rjk / rik - rik));
                            dcos_v[1] = 0.5 * (1 / rij + 1 / rik / rik * (rjk * rjk / rij - rij));
                            dcos_v[2] = -rjk / rij / rik;

                            g += G4(rij, rik, rjk, r_cutoff, eta, lambda, xi, cos_v, dcos_v, out_dg);
                        }
                    }
                    break;
                case 5:
                    g = 0;
                    for (int j = 0; j < n_atoms; j++){
                        for (int k = 0; k < n_atoms; k++){
                            if (i == j || i == k || j == k){
                                continue;
                            }
                            ri = cartesians[i];
                            rj = cartesians[j];
                            rk = cartesians[k];

                            double rij = 0, rik = 0, rjk = 0;
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
                            
                            dcos_v[0] = 0.5 * (1 / rik + 1 / rij / rij * (rjk * rjk / rik - rik));
                            dcos_v[1] = 0.5 * (1 / rij + 1 / rik / rik * (rjk * rjk / rij - rij));
                            dcos_v[2] = -rjk / rij / rik;
                            g += G5(rij, rik, rjk, r_cutoff, eta, lambda, xi, cos_v, dcos_v, out_dg);
                        }
                    }
                    break;
                }
                g_atom.push_back(g);
            }

            for (int dim = 0; dim < 3; dim++){
                dg_atom[dim] = out_dg[dim];
            }

            g_total.push_back(g_atom);
            dg_total.push_back(dg_atom);
        }
        double min[5] = {__DBL_MAX__, __DBL_MAX__, __DBL_MAX__, __DBL_MAX__, __DBL_MAX__};
        double max[5] = {__DBL_MIN__, __DBL_MIN__, __DBL_MIN__, __DBL_MIN__, __DBL_MIN__};

        for (int i = 0; i < n_atoms; i++){
            for (int g_type = 0; g_type < 5; g_type++){
                if (g_total[i][g_type] < min[g_type]){
                    min[g_type] = g_total[i][g_type];
                }
                if(g_total[i][g_type] > max[g_type]){
                    max[g_type] = g_total[i][g_type];
                }
            }
        }

        for (int i = 0; i < n_atoms; i++){
            for (int g_type = 0; g_type < 5; g_type++){
                g_total[i][g_type] = (g_total[i][g_type] - min[g_type]) / (max[g_type] - min[g_type]);
            }
        }
        double dg_min[3] = {__DBL_MAX__, __DBL_MAX__, __DBL_MAX__};
        double dg_max[3] = {__DBL_MIN__, __DBL_MIN__, __DBL_MIN__};

        for (int i = 0; i < n_atoms; i++){
            for (int dim = 0; dim < 3; dim++){
                if (dg_total[i][dim] < dg_min[dim]){
                    dg_min[dim] = dg_total[i][dim];
                }
                else if(dg_total[i][dim] > dg_max[dim]){
                    dg_max[dim] = dg_total[i][dim];
                }
            }
        }
        for (int i = 0; i < n_atoms; i++){
            for (int dim = 0; dim < 3; dim++){
                dg_total[i][dim] = (dg_total[i][dim] - dg_min[dim]) / (dg_max[dim] - dg_min[dim]);
            }
        }
        return g_total;
}