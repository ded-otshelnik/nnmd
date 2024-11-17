#include <head.hpp>
#include "cuda/cuda_header.hpp"

namespace cuda{
    Tensor calculate_gdf(Tensor& refs, int num_refs, int num_targets,
                        int num_features, double sigma);
}