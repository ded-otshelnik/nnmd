#include "cuda/symmetric_functions.hpp"
#include "cuda/cuda_header.hpp"

namespace cuda{
    __global__ void calculate_sf_kernel(
                    const at::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> cartesians,
                    at::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> g_total,
                    const float r_cutoff, const float eta, const float rs,
                    const float k_param, const float lambda, const float xi,
                    const int n_atoms
    );

    __device__ float G1(const float rij, const float rc);
    __device__ float G2(const float rij, const float rc, const float eta, const float rs);
    __device__ float G3(const float rij, const float rc, const float k);
    __device__ float G4(const float rij, const float rik, const float rjk, const float rc,
            const float eta, const float lambda, const float xi,
            const float cos_v);
    __device__ float G5(const float rij, const float rik, const float rjk, const float rc,
            const float eta, const float lambda, const float xi,
            const float cos_v);

    // @brief Calculates symmetric functions and its derivatives
    // @param cartesians: atomic positions
    // @param r_cutoff: cutoff radius
    // @param eta: parameter of symmetric functions
    // @param rs: parameter of symmetric functions
    // @param lambda: parameter of symmetric functions
    // @param xi: parameter of symmetric functions
    // @param dg_total: storage of output derivatives
    Tensor calculate_sf(const Tensor& cartesians, const float& r_cutoff,
                    const float& eta, const float& rs, const float& k_param,
                    const int& lambda, const float& xi){

        CHECK_INPUT(cartesians);
        
        torch::TensorOptions opts = torch::TensorOptions()
                                        .dtype(torch::kFloat)
                                        .device(torch::kCUDA);
        int N = cartesians.size(0);

        // output g values
        Tensor g_total = torch::zeros({N, 5}, opts);

        int threads = 512;
        dim3 blocks(N, 5, N);

        auto cartesians_accessor = cartesians.packed_accessor32<float, 2, torch::RestrictPtrTraits>();
        auto g_total_accessor = g_total.packed_accessor32<float, 2, torch::RestrictPtrTraits>();
        
        calculate_sf_kernel<<<blocks, threads>>>(
                cartesians_accessor, g_total_accessor,
                r_cutoff, eta, rs, k_param, lambda, xi, N
        );

        // normalize g values
        g_total = torch::nn::functional::normalize(g_total,
                                                torch::nn::functional::NormalizeFuncOptions()
                                                                        .p(2.0)
                                                                        .dim(1));
        return g_total;
    }

    __global__ void calculate_sf_kernel(
                    const at::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> cartesians,
                    at::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> g_total,
                    const float r_cutoff, const float eta, const float rs,
                    const float k_param, const float lambda, const float xi,
                    const int n_atoms
    ){

            // current g value
            float g;

            int i = blockIdx.x * blockDim.x + threadIdx.x;
            int g_type = blockIdx.y;
            int j = blockIdx.z * blockDim.z + threadIdx.z;

            if(i < n_atoms && g_type < 5){
                    g = 0;
                    switch (g_type){
                        // G1
                        case 1:{
                            if(j < n_atoms){
                                if (i == j){
                                    break;
                                }
                                auto ri = cartesians[i];
                                auto rj = cartesians[j];
                                float rij = 0;
                                for (int dim = 0; dim < 3; dim++){
                                    rij += (ri[dim] - rj[dim]) * (ri[dim] - rj[dim]);
                                }

                                rij = sqrt(rij);
                                g += G1(rij, r_cutoff);
                            }
                            break;
                        }
                        // G2
                        case 2:
                        {
                            if(j < n_atoms){
                                if (i == j){
                                    break;
                                }
                                auto ri = cartesians[i];
                                auto rj = cartesians[j];
                                float rij = 0;
                                for (int dim = 0; dim < 3; dim++){
                                    rij += (ri[dim] - rj[dim]) * (ri[dim] - rj[dim]);
                                }
                                rij = sqrt(rij);
                                g += G2(rij, r_cutoff, eta, rs);
                            }
                            break;
                        }
                        // G3
                        case 3:
                        {
                            if(j < n_atoms){
                                if (i == j){
                                    break;
                                }
                                auto ri = cartesians[i];
                                auto rj = cartesians[j];
                                float rij = 0;
                                for (int dim = 0; dim < 3; dim++){
                                    rij += (ri[dim] - rj[dim]) * (ri[dim] - rj[dim]);
                                }
                                rij = sqrt(rij);
                                g += G3(rij, r_cutoff, k_param);
                            }
                            break;
                        }
                        // G4
                        case 4:
                        {
                            if (j < n_atoms){
                                for (int k = 0; k < n_atoms; k++){
                                    if (i == j || i == k || j == k){
                                        continue;
                                    }
                                    auto ri = cartesians[i];
                                    auto rj = cartesians[j];
                                    auto rk = cartesians[k];

                                    float rij = 0;
                                    float rik = 0;
                                    float rjk = 0;

                                    for (int dim = 0; dim < 3; dim++){
                                        rij += (ri[dim] - rj[dim]) * (ri[dim] - rj[dim]);
                                        rjk += (rk[dim] - rj[dim]) * (rk[dim] - rj[dim]);
                                        rik += (rk[dim] - ri[dim]) * (rk[dim] - ri[dim]);
                                    }

                                    rij = sqrt(rij);
                                    rik = sqrt(rik);
                                    rjk = sqrt(rjk);

                                    float cos_v = (rij * rij + rik * rik - rjk * rjk) / 2 / rij / rik;

                                    g += G4(rij, rik, rjk, r_cutoff, eta, lambda, xi, cos_v);
                                }
                            }
                            break;
                        }
                        // G5
                        case 5:
                        {
                            if (j < n_atoms){
                                for (int k = 0; k < n_atoms; k++){
                                    if (i == j || i == k || j == k){
                                        continue;
                                    }

                                    auto ri = cartesians[i];
                                    auto rj = cartesians[j];
                                    auto rk = cartesians[k];

                                    float rij = 0;
                                    float rik = 0;
                                    float rjk = 0;
                                    for (int dim = 0; dim < 3; dim++){
                                        rij += (ri[dim] - rj[dim]) * (ri[dim] - rj[dim]);
                                        rjk += (rk[dim] - rj[dim]) * (rk[dim] - rj[dim]);
                                        rik += (rk[dim] - ri[dim]) * (rk[dim] - ri[dim]);
                                    }
                                    rij = sqrt(rij);
                                    rik = sqrt(rik);
                                    rjk = sqrt(rjk);
                                    float cos_v = (rij * rij + rik * rik - rjk * rjk) / 2 / rij / rik;
                                    
                                    g += G5(rij, rik, rjk, r_cutoff, eta, lambda, xi, cos_v);
                                }
                            }
                            break;
                        }
                    }
                    // pass g and dg values of atom
                    g_total[i][g_type - 1] = g;
                }
    }

    __device__ float cutf(const float rij, const float rc){
        if (rij < rc){
            return 0.5 * (cos(M_PI * rij / rc) + 1);
        }
        return 0;
    }

    __device__ float dcutf(const float rij, const float rc){
        if (rij < rc){
            return 0.5 * (-M_PI * sin(M_PI * rij / rc) / rc);
        }
        return 0;
    }

    __device__ float G1(const float rij, const float rc){
        return cutf(rij, rc);
    }

    __device__ float G2(const float rij, const float rc, const float eta, const float rs){
        return exp(-eta * (rij - rs) * (rij - rs)) * cutf(rij, rc);
    }

    __device__ float G3(const float rij, const float rc, const float k){ 
        return cos(k * rij) * cutf(rij, rc);
    }

    __device__ float G4(const float rij, const float rik, const float rjk, const float rc,
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

    __device__ float G5(const float rij, const float rik, const float rjk, const float rc,
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