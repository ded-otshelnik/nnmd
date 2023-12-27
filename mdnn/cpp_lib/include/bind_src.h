#ifndef BIND_SRC_H
#define BIND_SRC_H
#pragma once

#include <torch/extension.h>
#include <torch/script.h> 
#include <torch/csrc/utils/pybind.h>

#include <torch/optim/optimizer.h>
#include <torch/optim/sgd.h>
#include <torch/types.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
using torch::Tensor;

#include <vector>
using std::vector;

#include "symm_func/symmetric_functions.h"

using namespace torch::nn;

#endif