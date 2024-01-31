import glob
import os.path as osp
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

TORCH_EXTENSION_NAME = "nnmd"

setup(
    name = TORCH_EXTENSION_NAME,
    # version = VERSION,
    version = "0.0.0dev0",
    description = "Extension implementation with PyTorch C++ (Libtorch) and Python bindings",
    # long_description = README,
    author = "Andrey Budnikov",
    # install_requires = REQUIRES,
    include_package_data=True,
    packages=find_packages(exclude=["nnmd_cpp", "nnmd_cuda"]),
    package_data={"": ["*.so"]},
    test_suite="tests",
    ext_modules = [],
    cmdclass = {
        "build_ext": BuildExtension
    },
    platforms = ["linux_x86_64"]
)