#pragma once

//cuda headers
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/cuda.h>
#include <thrust/device_vector.h>

// special macroses for torch tensors 
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)