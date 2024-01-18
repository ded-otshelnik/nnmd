import multiprocessing
import os
import platform
import subprocess
import sys
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

import glob
import os.path as osp
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension


ROOT_DIR = osp.dirname(osp.abspath(__file__))
include_dirs = [osp.join(ROOT_DIR, "include")]

sources = glob.glob('src/*.cpp', recursive=True) + glob.glob('src/*.cu', recursive=True)

TORCH_EXTENSION_NAME = "mdnn_cuda"

setup(
    name = TORCH_EXTENSION_NAME,
    # version = VERSION,
    version = "0.0.1",
    description = "Extension implementation with PyTorch C++ (Libtorch) and Python bindings",
    # long_description = README,
    author = "Andrey Budnikov",
    # install_requires = REQUIRES,
    include_package_data=True,
    packages=find_packages(exclude=["tests"]),
    package_data={"": ["*.so"]},
    test_suite="tests",
    ext_modules = [CUDAExtension(
        name = TORCH_EXTENSION_NAME,
        sources = sources,
        include_dirs = include_dirs,
        extra_compile_args = {'cxx': ['-O2'],
                              'nvcc': ['-O2', '-lcudnn']}
    )],
    cmdclass = {
        "build_ext": BuildExtension
    }
)