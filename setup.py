import multiprocessing
import os
import platform
import subprocess
import sys

import glob
import os.path as osp
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

EXTENSION_NAME = "nnmd"

class CMakeBuild(BuildExtension):  #(build_ext):

    def __init__(self, *args, **kwargs):
        super(CMakeBuild, self).__init__(*args, **kwargs)
        self.python_exe = sys.executable
        self.pytorch_dir = None
        self.pybind11_dir = None
        self.cmake = None

    @property
    def cmake(self):
        if self._cmake is None:
            cmake_bin = os.getenv("CMAKE_EXECUTABLE", "cmake")
            cmake_bin = subprocess.check_output(["which", cmake_bin]).decode().strip()
            self.announce("CMAKE_EXECUTABLE: {}".format(cmake_bin))
            self._cmake = cmake_bin
        return self._cmake

    @cmake.setter
    def cmake(self, cmake):
        self._cmake = cmake

    def find_torch_dir(self):
        """
        Attempts finding precompiled :mod:`torch`.

        Searches with :envvar:`TORCH_DIR`, :envvar:`TORCH_LIBRARY` or reverts back to preinstalled package via ``pip``.
        """
        pytorch_dir = os.getenv("PYTORCH_DIR")
        if not pytorch_dir:
            pytorch_dir = os.getenv("TORCH_DIR")
        pytorch_lib = os.getenv("TORCH_LIBRARY")
        pytorch_lib_path = "lib/libtorch.so" if platform.system() != "Windows" else "lib/x64/torch.lib"
        if pytorch_dir and os.path.isdir(pytorch_dir) and os.path.isfile(os.path.join(pytorch_dir, pytorch_lib_path)):
            pytorch_lib = os.path.join(pytorch_dir, pytorch_lib_path)
        elif pytorch_lib and os.path.isfile(pytorch_lib) and os.path.isdir(pytorch_lib.replace(pytorch_lib_path, "")):
            pytorch_dir = pytorch_lib.replace(pytorch_lib_path, "")
        else:
            try:
                import torch  # noqa
                pytorch_dir = os.path.dirname(torch.__file__)
                pytorch_lib = os.path.join(pytorch_dir, pytorch_lib_path)
            except ImportError:
                sys.stderr.write("Pytorch is required to build this package\n")
                sys.exit(-1)
        if not os.path.isdir(pytorch_dir) or not os.path.isfile(pytorch_lib):
            sys.stderr.write("Pytorch is required to build this package. "
                             "Set TORCH_DIR for pre-compiled from sources, or install with pip.\n")
        self.announce("Found PyTorch dir: {}".format(pytorch_dir))
        return pytorch_dir

    def find_pybind_dir(self):
        pybind_dir = os.getenv("PYBIND11_DIR", "")
        if not os.path.isdir(pybind_dir):
            raise RuntimeError("Library pybind11 required but not valid: [{}]".format(pybind_dir))
        self.announce("Found PyBind11 dir: {}".format(pybind_dir))
        self.pybind11_dir = pybind_dir
        return self.pybind11_dir

    def run(self):
        try:
            _ = subprocess.check_output([self.cmake, "--version"])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(ext.name for ext in self.extensions))

        self.pytorch_dir = self.find_torch_dir()
        self.pybind11_dir = self.find_pybind_dir()
        for ext in self.extensions:
            self.build_cmake(ext)

    def build_cmake(self, ext):
        ext_path = self.get_ext_fullpath(ext.name)
        ext_dir = os.path.abspath(os.path.dirname(ext_path))
        build_dir = os.path.join(ext_dir, self.build_temp)
        self.announce("Extension Path: {}".format(ext_path))
        self.announce("Extension Dir:  {}".format(ext_dir))
        self.announce("Ext Build Path: {}".format(self.build_temp))
        self.announce("Ext Build Dir:  {}".format(build_dir))

        cmake_args = [
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={}".format(ext_dir),
            # "-DCMAKE_PREFIX_PATH={}".format(self.pytorch_dir),
            "-DPYTHON_EXECUTABLE:FILEPATH={}".format(self.python_exe),
            "-DWITH_PYTHON=ON",
            "-DWITH_TESTS=OFF",       # cannot be simultaneously with Python module
            "-DWITH_TEST_BENCH=OFF",  # cannot be simultaneously with Python module
            "-DTORCH_DIR={}".format(self.pytorch_dir),
            "-DPYBIND11_DIR={}".format(self.pybind11_dir),
            # "-DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=0",  # defined by Torch CXX FLAGS directly (must match)
        ]

        config = "Debug" if self.debug else "Release"
        build_args = ["--config", config]

        if platform.system() == "Darwin":
            cmake_args += ["-DCMAKE_OSX_DEPLOYMENT_TARGET=10.9"]

        if platform.system() == "Windows":
            cmake_args += ["-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}".format(config.upper(), ext_dir)]
            if sys.maxsize > 2 ** 32:
                cmake_args += ["-A", "x64"]
            build_args += ["--", "/m"]
        else:
            cmake_args += ["-DCMAKE_BUILD_TYPE=" + config]

        if not os.path.exists(build_dir):
            os.makedirs(build_dir)
        env = os.environ.copy()
        # configure/generate
        src = os.path.abspath(".")
        cmd = [self.cmake]
        if os.getenv("DISTUTILS_DEBUG") == "1":
            cmd += ["--log-level=DEBUG"]
        cmd += [src] + cmake_args
        self.announce("Configure with CMake:\n{}".format(cmd))
        subprocess.check_call(cmd, cwd=build_dir, env=env)
        # compile
        if not self.dry_run:
            jobs = os.getenv("CMAKE_JOBS", multiprocessing.cpu_count())
            cmd = [self.cmake, "--build", build_dir, "--", "-j{}".format(jobs)]
            subprocess.check_call(cmd, cwd=build_dir, env=env)


with open("README.md") as f:
    README = f.read()

with open("VERSION") as ver:
    VERSION = ver.readline().strip()

with open("requirements.txt") as r:
    REQUIRES = []
    for line in r.readlines():
        if line.startswith("#"):
            continue
        REQUIRES.append(line.strip())



setup(
    name = EXTENSION_NAME,
    version = VERSION,
    description = "Extension implementation with PyTorch C++ (Libtorch) and Python bindings",
    long_description = README,
    author = "Andrey Budnikov",
    # install_requires = REQUIRES,
    include_package_data=True,
    packages=find_packages(exclude=["test"]),
    package_data={"": ["*.so"]},
    test_suite="tests",
    ext_modules = [CUDAExtension(
        name = "_" + EXTENSION_NAME + "_cpp",
        sources = [],
        extra_compile_args = {}
    )],
    cmdclass = {
        "build_ext": CMakeBuild
    },
    platforms = ["linux_x86_64"]
)