#include "symmetric_functions.h"

double cutf(const double rij, const double rc){
    if (rij < rc){
        return 0.5 * (cos(M_PI * rij / rc) + 1);
    }
    return 0;
}

double dcutf(const double rij, const double rc){
    if (rij < rc){
        return 0.5 * (-M_PI * sin(M_PI * rij / rc) / rc);
    }
    return 0;
}

double G1(const double rij, const double rc, double* out_dg){
    double out_g = cutf(rij, rc);
    out_dg[0] += dcutf(rij, rc);
    out_dg[1] += 0;
    out_dg[2] += 0;
    return out_g;
}

double G2(const double rij, const double rc, const double eta, const double rs, double* out_dg){ 
    double out_g = exp(-eta * (rij - rs) * (rij - rs));
    out_dg[0] += out_g * (-2 * eta * (rij - rs) * cutf(rij, rc) + dcutf(rij, rc));
    out_dg[1] += 0;
    out_dg[2] += 0;
    return out_g * cutf(rij, rc);
}

double G3(const double rij, const double rc, const double k, double* out_dg){ 
    double out_g = cos(k * rij);
    out_dg[0] += - k * sin(k * rij) * cutf(rij, rc) + out_g * dcutf(rij, rc);
    out_dg[1] += 0;
    out_dg[2] += 0;
    return out_g * cutf(rij, rc);
}

double G4(const double rij, const double rik, const double rjk, const double rc,
          const double eta, const double lambda, const double xi,
          const double cos_v, const double* dcos_v, double* out_dg){ 
        
    double out_g;
    double expv = exp(-eta * (rij * rij + rik * rik + rjk * rjk)); 
    double cosv = 1 + lambda * cos_v;
    double powcos;
    if (fabs(cosv) < 10e-4){
        powcos = 0;
    }
    else{
        powcos = pow(cosv, xi);
    }
    out_g = pow(2, 1 - xi) * powcos * expv * \
            cutf(rij, rc) * cutf(rik, rc) * cutf(rjk, rc);

    out_dg[0] += expv * powcos * cutf(rik, rc) * cutf(rjk, rc) * \
            (( -2 * eta * rij * cutf(rij, rc) + dcutf(rij, rc)) * cosv) + \
            xi * lambda * cutf(rij, rc) * \
            dcos_v[0];
    out_dg[1] += expv * powcos * cutf(rij, rc) * cutf(rjk, rc) * \
            (( -2 * eta * rik * cutf(rik, rc)) * cosv) + \
            xi * lambda * cutf(rik, rc) * \
            dcos_v[1];
    out_dg[2] += expv * powcos * cutf(rij, rc) * cutf(rik, rc) * \
            (( -2 * eta * rjk * cutf(rjk, rc)) * cosv) + \
            xi * lambda * cutf(rjk, rc) * \
            dcos_v[2];
    return out_g;
}

double G5(const double rij, const double rik, const double rjk, const double rc,
          const double eta, const double lambda, const double xi,
          const double cos_v, const double* dcos_v, double* out_dg){ 
        
    double out_g;
    double expv = exp(-eta * (rij * rij + rik * rik)); 
    double cosv = 1 + lambda * cos_v;
    double powcos;
    if (fabs(cosv) < 10e-4){
        powcos = 0;
    }
    else{
        powcos = pow(cosv, xi);
    }
    out_g = pow(2, 1 - xi) * powcos * expv * \
            cutf(rij, rc) * cutf(rik, rc);

    out_dg[0] += expv * powcos * cutf(rik, rc) * \
            (( -2 * eta * rij * cutf(rij, rc) + dcutf(rij, rc)) * cosv) + \
            xi * lambda * cutf(rij, rc) * \
            dcos_v[0];
    out_dg[1] += expv * powcos * cutf(rij, rc) * \
            (( -2 * eta * rik * cutf(rik, rc)) * cosv) + \
            xi * lambda * cutf(rik, rc) * \
            dcos_v[1];
    out_dg[2] += expv * powcos * cutf(rij, rc) * cutf(rik, rc) * \
            (( -2 * eta * rjk * cutf(rjk, rc)) * cosv) + \
            xi * lambda * cutf(rjk, rc) * \
            dcos_v[2];
    return out_g;
}