# nnmd

Neural Networks for Molecular Dynamics (nnmd) - package that provides creation, training and usage
machine-learning potentials for MD simulations.
It uses Pytorch as ML package for both Python and C++ parts.

## Prerequisites

0. Install CUDA and CUDNN (compatible to CUDA)
if you need training on GPU
1. Install CMake (3.11 or more) and Ninja for building C++ extentions.
2. Install Pybind11 (the simpliest way - using pip).
3. (!!!) Build Pytorch from source and download or build from source Libtorch (with compatible CUDA version).
Make sure that you have compatible versions of C++ library and Python module.
4. Set next environment variables: 
    - PYBIND11_DIR
    - TORCH_LIB
    - Torch_DIR
5. If you build with CUDA support, export next environment variables:
    - CMAKE_CUDA_COMPILER
    - CUDA_CUDA_LIB
    - CUDACXX

## Installation

Recommendation: use virtual environments.

Go to repository folder and run command:
```python
pip install . -v
```
It

It installs required module with C++ extension to your python environment.

## Usage

After setup you can use package. 