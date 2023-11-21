#ifndef SYMMETRIC_FUNCTIONS_H
#define SYMMETRIC_FUNCTIONS_H

#define _USE_MATH_DEFINES
#include <cmath>

double G1(const double rij, const double rc, double* out_dg);
double G2(const double rij, const double rc, const double eta, const double rs, double* out_dg);
double G3(const double rij, const double rc, const double k, double* out_dg);
double G4(const double rij, const double rik, const double rjk, const double rc,
          const double eta, const double lambda, const double xi,
          const double cos_v, const double* dcos_v, double* out_dg);
double G5(const double rij, const double rik, const double rjk, const double rc,
          const double eta, const double lambda, const double xi,
          const double cos_v, const double* dcos_v, double* out_dg);
#endif