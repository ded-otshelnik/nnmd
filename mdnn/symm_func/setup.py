# Cython compile instructions

# Use python setup.py build_ext --inplace
# to compile

from __future__ import absolute_import, print_function

import os
import sys

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

source_files = ['symm_func.pyx',  'symmetric_functions.cpp']
extentions = [Extension('symm_func', source_files)]

setup(
    name='symm_func',
    ext_modules=cythonize(extentions)
)