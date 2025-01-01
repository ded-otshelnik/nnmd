#include "cuda/symmetric_functions.hpp"

namespace cuda{

    __global__ void calculate_sf_kernel(
                    const at::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> cartesians,
                    at::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> g_total,
                    const int* features, float** params, int n_atoms, int n_features
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

    // @brief Calculates symmetry functions
    // @param cartesians: atomic positions
    // @param features: list of symmetry functions to calculate
    // @param params: list of parameters for each symmetry function
    // (length of list must be equal to length of features list
    // and all vectors must have the size equal to the number of symmetry functions params)
    // @return g_total: symmetry functions
    Tensor calculate_sf(const Tensor& cartesians, const vector<int>& features, const vector<vector<float>>& params){
        CHECK_INPUT(cartesians);
        
        torch::TensorOptions opts = torch::TensorOptions()
                                        .dtype(torch::kFloat)
                                        .device(torch::kCUDA);
        int N_atoms = cartesians.size(0);
        int N_features = features.size();
        
        // output g values
        Tensor g_total = torch::zeros({N_atoms, N_features}, opts);

        int threads = 512;
        dim3 blocks(N_atoms, N_features, N_atoms);

        // accessors to torch tensors for gpu
        auto cartesians_accessor = cartesians.packed_accessor32<float, 2, torch::RestrictPtrTraits>();
        auto g_total_accessor = g_total.packed_accessor32<float, 2, torch::RestrictPtrTraits>();

        // copy features and params to device
        int* features_d;
        cudaMalloc(&features_d, N_features * sizeof(int));
        cudaMemcpy(features_d, features.data(), N_features * sizeof(int), cudaMemcpyHostToDevice);

        float** params_d;
        float** params_h = (float**)malloc(N_features * sizeof(float*));
        for (int i = 0; i < N_features; i++){
            cudaMalloc(&params_h[i], params[i].size() * sizeof(float));
            cudaMemcpy(params_h[i], params[i].data(), params[i].size() * sizeof(float), cudaMemcpyHostToDevice);
        }
        cudaMalloc(&params_d, N_features * sizeof(float*));
        cudaMemcpy(params_d, params_h, N_features * sizeof(float*), cudaMemcpyHostToDevice);
        free(params_h);

        calculate_sf_kernel<<<blocks, threads>>>(
                cartesians_accessor, g_total_accessor,
                features_d, params_d, N_atoms, N_features
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
                    const int* features, float** params, int n_atoms, int n_features
    ){

            int i = blockIdx.x * blockDim.x + threadIdx.x;
            int feature_index = blockIdx.y;
            int j = blockIdx.z * blockDim.z + threadIdx.z;

            float g;
            if(i < n_atoms && feature_index < n_features){
                    switch (features[feature_index]){
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
                                g = G1(rij, params[feature_index][0]);
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
                                g = G2(rij, params[feature_index][0], params[feature_index][1], params[feature_index][2]);
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
                                g = G3(rij, params[feature_index][0], params[feature_index][1]);
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

                                    g = G4(rij, rik, rjk,
                                             params[feature_index][0], params[feature_index][1], 
                                             params[feature_index][2], params[feature_index][3], cos_v);
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
                                    
                                    g = G5(rij, rik, rjk,
                                             params[feature_index][0], params[feature_index][1], 
                                             params[feature_index][2], params[feature_index][3], cos_v);
                                }
                            }
                            break;
                        }
                    }
                    g_total[i][feature_index - 1] += g;
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

    __device__ float G3(const float rij, const float rc, const float kappa){ 
        return cos(kappa * rij) * cutf(rij, rc);
    }

    __device__ float G4(const float rij, const float rik, const float rjk, const float rc,
            const float eta, const float lambda, const float zeta,
            const float cos_v){ 
            
        float out_g;
        float expv = exp(-eta * (rij * rij + rik * rik + rjk * rjk)); 
        float cosv = 1 + lambda * cos_v;
        float powcos;
        if (fabs(cosv) < 10e-4){
            powcos = 0;
        }
        else{
            powcos = pow(cosv, zeta);
        }
        out_g = pow(2, 1 - zeta) * powcos * expv * \
                cutf(rij, rc) * cutf(rik, rc) * cutf(rjk, rc);
                
        return out_g;
    }

    __device__ float G5(const float rij, const float rik, const float rjk, const float rc,
            const float eta, const float lambda, const float zeta,
            const float cos_v){ 
            
        float out_g;
        float expv = exp(-eta * (rij * rij + rik * rik)); 
        float cosv = 1 + lambda * cos_v;
        float powcos;
        if (fabs(cosv) < 10e-4){
            powcos = 0;
        }
        else{
            powcos = pow(cosv, zeta);
        }
        out_g = pow(2, 1 - zeta) * powcos * expv * \
                cutf(rij, rc) * cutf(rik, rc);

        return out_g;
    }
}