#include "head.h"
#include "symm_func/symmetric_functions.h"

Tensor cutf(const Tensor rij, const double rc){
    if (*rij.data_ptr<double>() < rc){
        return 0.5 * (cos(M_PI * rij / rc) + 1);
    }
    return torch::zeros(1);
}

Tensor dcutf(const Tensor rij, const double rc){

    if (*rij.data_ptr<double>() < rc){
        return (0.5 * -M_PI/ rc) * sin(M_PI * rij / rc);
    }
    return torch::zeros(1);
}

Tensor G1(const Tensor rij, const double rc, Tensor& out_dg){
    out_dg[0] += dcutf(rij, rc);
    return cutf(rij, rc);
}

Tensor G2(const Tensor rij, const double rc, const double eta, const double rs, Tensor& out_dg){ 
    Tensor out_g = exp(-eta * (rij - rs) * (rij - rs));
    out_dg[0] += out_g * (-2 * eta * (rij - rs) * cutf(rij, rc) + dcutf(rij, rc));
    return out_g * cutf(rij, rc);
}

Tensor G3(const Tensor rij, const double rc, const double k, Tensor& out_dg){ 
    Tensor out_g = cos(k * rij);
    out_dg[0] += - k * sin(k * rij) * cutf(rij, rc) + out_g * dcutf(rij, rc);
    return out_g * cutf(rij, rc);
}

Tensor G4(const Tensor rij, const Tensor rik, const Tensor rjk, const double rc,
          const double eta, const double lambda, const double xi,
          const Tensor cos_v, const Tensor dcos_v, Tensor& out_dg){ 
    Tensor cutf_rij = cutf(rij, rc), cutf_rik = cutf(rik, rc), cutf_rjk = cutf(rjk, rc);
    Tensor out_g;
    Tensor expv = exp(-eta * (rij * rij + rik * rik + rjk * rjk)); 
    Tensor cosv = 1 + lambda * cos_v;
    Tensor powcos = fabs(*cosv.data_ptr<double>()) < 10e-4 ? torch::zeros(1) : torch::pow(cosv, xi);
    out_g = pow(2, 1 - xi) * powcos * expv * \
            cutf_rij * cutf_rik * cutf_rjk;
    out_dg[0] += expv * powcos * cutf_rik * cutf_rjk * \
            (( -2 * eta * rij * cutf_rij + dcutf(rij, rc)) * cosv) + \
            xi * lambda * cutf_rij * \
            dcos_v[0];
    out_dg[1] += expv * powcos * cutf_rij * cutf_rjk * \
            (( -2 * eta * rik * cutf_rik) * cosv) + \
            xi * lambda * cutf_rik * \
            dcos_v[1];
    out_dg[2] += expv * powcos * cutf_rij * cutf_rik * \
            (( -2 * eta * rjk * cutf(rjk, rc)) * cosv) + \
            xi * lambda * cutf_rjk * \
            dcos_v[2];
    return out_g;
}

Tensor G5(const Tensor rij, const Tensor rik, const Tensor rjk, const double rc,
          const double eta, const double lambda, const double xi,
          const Tensor cos_v, const Tensor dcos_v, Tensor& out_dg){ 
    Tensor cutf_rij = cutf(rij, rc), cutf_rik = cutf(rik, rc), cutf_rjk = cutf(rjk, rc);
    Tensor out_g;
    Tensor expv = exp(-eta * (rij * rij + rik * rik)); 
    Tensor cosv = 1 + lambda * cos_v;
    Tensor powcos = fabs(*cosv.data_ptr<double>()) < 10e-4 ? torch::zeros(1) : torch::pow(cosv, xi);
    out_g = pow(2, 1 - xi) * powcos * expv * 
            cutf_rij * cutf_rik;

    out_dg[0] += expv * powcos * cutf_rik * \
            (( -2 * eta * rij * cutf_rij + dcutf(rij, rc)) * cosv) + \
            xi * lambda * cutf_rij * \
            dcos_v[0];
    out_dg[1] += expv * powcos * cutf_rij * \
            (( -2 * eta * rik * cutf_rik) * cosv) + \
            xi * lambda * cutf_rik * \
            dcos_v[1];
    out_dg[2] += expv * powcos * cutf_rij * cutf(rik, rc) * \
            (( -2 * eta * rjk * cutf_rik) * cosv) + \
            xi * lambda * cutf_rjk * \
            dcos_v[2];
    return out_g;
}