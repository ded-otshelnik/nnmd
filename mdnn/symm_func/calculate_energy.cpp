#include "symmetric_functions.h"

double calculate_energy(const vector< vector<double> > &cartesians, const double& r_cutoff,
                const double& eta, const double& rs, const double& k_param,
                const int& lambda, const double& xi){
        double energy = 0;
        vector< vector<double> > dg_total;
        vector< vector<double> > g_total = calculate_sf(cartesians, r_cutoff, eta, rs, k_param, lambda, xi, dg_total);
        for (int i = 0; i < g_total.size(); i++){
            for (int j = 0; j < 5; j++){
                energy += g_total[i][j];
            }
        }
        return energy;
}