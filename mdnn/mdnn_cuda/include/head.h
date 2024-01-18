#ifndef HEAD_H
#define HEAD_H
// preprocessor directives
#pragma once

// torch headers
#include <torch/extension.h>
#include <torch/script.h> 
#include <torch/csrc/utils/pybind.h>
#include <torch/optim/optimizer.h>
#include <torch/optim/sgd.h>
#include <torch/types.h>
using torch::Tensor;
using namespace torch::nn;

// pybind11 headers
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/attr.h>
#include <pybind11/functional.h>
namespace py = pybind11;

// stl headers
#include <vector>
using std::vector;
#include <iostream>
using std::cout, std::endl;
#include <string>
using std::to_string;

#endif