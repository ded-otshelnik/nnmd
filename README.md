# nnmd

Train your own atomic potentials via neural networks!

## Prerequisites

0. Install CUDA (11.8 or more) and CUDNN (compatible to CUDA)
if you need training on GPU
1. Install CMake (3.1 or more) and Ninja for building C++ extentions.
2. Install Pybind11 (the simpliest way - using pip).
3. Build Pytorch from source and download Libtorch (with compatible CPU/CUDA version).
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

Go to nnmd_cpp/nnmd_cuda folders and run command:
```python
pip install .
```
Then go to root dir and run the same command

It installs required modules to your python environments.

## Usage

After setup you can use package. In samples directory there is a sample code called [main.py](https://github.com/ded-otshelnik/nnmd/blob/main/samples/main.py)
