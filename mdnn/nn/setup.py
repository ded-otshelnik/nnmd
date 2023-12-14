from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='atomic_nn_cpp',
      ext_modules=[cpp_extension.CppExtension('atomic_nn_cpp', ['atomic_nn.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})