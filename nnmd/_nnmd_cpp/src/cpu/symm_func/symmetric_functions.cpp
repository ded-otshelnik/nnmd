#include "cpu/symmetric_functions.hpp"

namespace cpu{
    
    float cutf(const float rij, const float rc){
        if (rij < rc){
            return 0.5 * (cos(M_PI * rij / rc) + 1);
        }
        return 0;
    }

    float dcutf(const float rij, const float rc){
        if (rij < rc){
            return 0.5 * (-M_PI * sin(M_PI * rij / rc) / rc);
        }
        return 0;
    }

    float G1(const float rij, const float rc){
        float out_g = cutf(rij, rc);
        return out_g;
    }

    float G2(const float rij, const float rc, const float eta, const float rs){ 
        float out_g = exp(-eta * (rij - rs) * (rij - rs));
        return out_g * cutf(rij, rc);
    }

    float G3(const float rij, const float rc, const float k){ 
        float out_g = cos(k * rij);
        return out_g * cutf(rij, rc);
    }

    float G4(const float rij, const float rik, const float rjk, const float rc,
            const float eta, const float lambda, const float xi,
            const float cos_v){ 
            
        float out_g;
        float expv = exp(-eta * (rij * rij + rik * rik + rjk * rjk)); 
        float cosv = 1 + lambda * cos_v;
        float powcos;
        if (fabs(cosv) < 10e-4){
            powcos = 0;
        }
        else{
            powcos = pow(cosv, xi);
        }
        out_g = pow(2, 1 - xi) * powcos * expv * \
                cutf(rij, rc) * cutf(rik, rc) * cutf(rjk, rc);
        return out_g;
    }

    float G5(const float rij, const float rik, const float rjk, const float rc,
            const float eta, const float lambda, const float xi,
            const float cos_v){ 
            
        float out_g;
        float expv = exp(-eta * (rij * rij + rik * rik)); 
        float cosv = 1 + lambda * cos_v;
        float powcos;
        if (fabs(cosv) < 10e-4){
            powcos = 0;
        }
        else{
            powcos = pow(cosv, xi);
        }
        out_g = pow(2, 1 - xi) * powcos * expv * \
                cutf(rij, rc) * cutf(rik, rc);

        return out_g;
    }
}