#include <vector>
#include "symmetric_functions.h"

vector< vector<double> > calculate_forces(const vector< vector<double> > &cartesians, const double& r_cutoff,
                const double& eta, const double& rs, const double& k,
                const int& lambda, const double& xi, const double& h){
        int atoms_amount = cartesians.size();
        vector< vector<double> > cartesians_copy(cartesians);
        vector< vector<double> > forces, dg_actual;
        vector< vector<double> > g_actual = calculate_sf(cartesians_copy, r_cutoff, eta, rs, k, lambda, xi, dg_actual);
        for (int atom = 0; atom < atoms_amount; atom++){
            vector<double> forces_atom;
            for (int coord = 0; coord < 3; coord++){
                cartesians_copy[atom][coord] += h;
                vector< vector<double> > dg;
                vector< vector<double> > g = calculate_sf(cartesians_copy, r_cutoff, eta, rs, k, lambda, xi, dg);
                double force = 0;
                for (int i = 0; i < 5; i++){
                    for (int j = 0; j < 3; j++){
                        force -= (g[atom][i] - g_actual[atom][i]) * (dg[atom][j] - dg_actual[atom][j]);
                    }
                }
                forces_atom.push_back(force);
                cartesians_copy[atom][coord] -= h;
            }
            forces.push_back(forces_atom);
        }
        return forces;
}