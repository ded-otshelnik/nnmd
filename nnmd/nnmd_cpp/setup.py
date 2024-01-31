import glob
import os
import os.path as osp
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

TORCH_EXTENSION_NAME = "nnmd_cpp"

ROOT_DIR = osp.dirname(osp.abspath(__file__))
include_dirs = [osp.join(ROOT_DIR, "include")]

sources = glob.glob('src/*.cpp') + glob.glob('src/symm_func/*.cpp')

setup(
    name = TORCH_EXTENSION_NAME,
    # version = VERSION,
    version = "0.0.0.dev0",
    description = "Extension implementation with PyTorch C++ (Libtorch) and Python bindings",
    # long_description = README,
    author = "Andrey Budnikov",
    # install_requires = REQUIRES,
    include_package_data=True,
    packages=find_packages(exclude=["tests"]),
    package_data={"": ["*.so"]},
    test_suite="tests",
    ext_modules = [CppExtension(
        name = TORCH_EXTENSION_NAME,
        sources = sources,
        include_dirs = include_dirs,
        extra_compile_args = {'cxx': ['-O2']}
    )],
    cmdclass = {
        "build_ext": BuildExtension
    },
    platforms = ["linux_x86_64"]
)