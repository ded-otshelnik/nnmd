"""
Setup the Python/C++ extension for NNMD.

Example of setup.py for Python/C++ extension:
    https://github.com/crim-ca/crim-libtorch-extensions/blob/master/setup.py
    
Torch setup:
    https://pytorch.org/tutorials/advanced/cpp_extension.html#writing-a-c-extension
"""
import multiprocessing
import os
import platform
import subprocess
import sys

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

class CMakeBuild(BuildExtension):
    """Class provides CMake building of Python/C++ extensions
    with customization by CMakeLists.txt
    """

    def __init__(self, *args, **kwargs):
        super(CMakeBuild, self).__init__(*args, **kwargs)
        self.python_exe = sys.executable
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

    def run(self):
        try:
            _ = subprocess.check_output([self.cmake, "--version"])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(ext.name for ext in self.extensions))
        
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
            "-DPYTHON_EXECUTABLE:FILEPATH={}".format(self.python_exe),
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

setup(
    include_package_data=True,
    package_data={"": ["*.so"]},
    ext_modules = [CUDAExtension(
        name = "_nnmd_cpp",
        sources = [],
        extra_compile_args = {}
    )],
    cmdclass = {
        "build_ext": CMakeBuild
    },
    platforms = ["linux_x86_64"]
)