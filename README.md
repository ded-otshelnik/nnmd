# nnmd

Train your own atomic potentials via neural networks!

## Prerequisites
0. Install CUDA (11.8 or more) and CUDNN (compatible to CUDA)
if you need training on GPU
1. Build Pytorch from source and download Libtorch (with compatible CPU/CUDA version).
Make sure that you have compatible versions of C++ library and Python module. 

## Installation

Go to nnmd_cpp/nnmd_cuda folders and run command
```
    python setup.py install
```

## Usage

After setup you can use package. In root there is a sample code main.py