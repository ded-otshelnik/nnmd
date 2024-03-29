// preprocessor directives
#pragma once

// torch headers and usings
#include <torch/extension.h>
#include <torch/script.h> 
#include <torch/csrc/utils/pybind.h>
#include <torch/optim/optimizer.h>
#include <torch/optim/sgd.h>
#include <torch/types.h>
using torch::Tensor;
using namespace torch::nn;

// pybind11 headers and usings
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/attr.h>
#include <pybind11/functional.h>
namespace py = pybind11;

// stl headers and usings
#include <vector>
using std::vector;

// IO headers/usings (will be removed later)
#include <iostream>
using std::cout, std::endl;
#include <string>
using std::to_string;

// time headers/usings
#include <chrono>
#include <ratio>
using namespace std::chrono;

#include <exception>