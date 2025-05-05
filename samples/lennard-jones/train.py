#!/usr/bin/env python
# -*- coding: utf-8 -*-

# example of nnmd package usage with lennard-jones potentials

import os
import shutil
import warnings

import torch
import numpy as np
import matplotlib.pyplot as plt

from nnmd.features import calculate_params
from nnmd.io import input_parser
from nnmd.nn import BPNN
from nnmd.nn.dataset import make_atomic_dataset
from nnmd.util import train_val_test_split

warnings.filterwarnings("ignore")
dtype = torch.float32

import torch

from nnmd.nn import BPNN
from nnmd.nn.dataset import make_atomic_dataset

print("Getting Lennard-Jones potential: ", end="")

input_data = input_parser("input/input.yaml")

print("done")

# parameters for symmetry functions

symm_funcs_data = input_data["atomic_data"]["symmetry_functions_set"]
print("Get info from traj simulation: done")

# convert train data to atomic dataset with symmetry functions
dataset = make_atomic_dataset(input_data['atomic_data'], symm_funcs_data)

# ~80% - train, ~10% - test and validation
train_val_test_ratio = (0.8, 0.1, 0.1)
train_dataset, val_dataset, test_dataset = train_val_test_split(
    dataset, train_val_test_ratio
)
print(f"Separate data to train and test datasets: done")

# train model
train = not input_data["neural_network"]["train"]
# save model params as files in <path> directory
save = not input_data["neural_network"]["save"]
# path to save model
path = input_data["neural_network"]["path"]

# Atomic NN input_size in hidden layers
input_sizes = input_data["atomic_data"]["n_atoms"]
output_sizes = [1]

print("Create an instance of NN and config its subnets:", end=" ")

net = BPNN(dtype=dtype)
net.config(input_data["neural_network"], input_sizes, output_sizes)

print("done")

try:
    if train:
        print("Training:", end=" ")
        net.fit(train_dataset, val_dataset, test_dataset)

    if save:
        if os.path.exists(path):
            shutil.rmtree(path, ignore_errors=True)
        os.mkdir(path)

        net.save_model(path)
        print("Saving model: done")

except KeyboardInterrupt:
    if os.path.exists("checkpoint"):
        shutil.rmtree("checkpoint", ignore_errors=True)
    os.mkdir("checkpoint")

    net.save_model("checkpoint")
    print("Training is stopped. Model is saved")

print("Visualize results:", end=" ")

print(input_data["atomic_data"]['reference_data'][0])
dataset = input_data["atomic_data"]["reference_data"]

# get cartesian coordinates for each species
cartesians = {
        spec: torch.tensor(
            np.array([data[spec]["positions"] for data in dataset]),
            dtype=torch.float32,
            device="cuda",
        )
        for spec in dataset[0].keys()
        if spec not in ["forces", "energy", "velocities"]
}
cell = torch.tensor(
    input_data["atomic_data"]["unit_cell"], dtype=torch.float32, device="cuda"
)

pred = net.predict(cartesians, cell, symm_funcs_data)
pred = pred.cpu().detach().numpy()

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].scatter(
    dataset["reference_data"]["energy"].cpu().detach().numpy(),
    pred[:, 0],
    s=1,
    c="blue",
    alpha=0.5,
)
ax[0].plot(
    [dataset["reference_data"]["energy"].min(), dataset["reference_data"]["energy"].max()],
    [dataset["reference_data"]["energy"].min(), dataset["reference_data"]["energy"].max()],
    c="red",
)
ax[0].set_xlabel("True energy")
ax[0].set_ylabel("Predicted energy")
ax[0].set_title("Energy prediction")
ax[1].scatter(
    dataset["reference_data"]["forces"].cpu().detach().numpy(),
    pred[:, 1],
    s=1,
    c="blue",
    alpha=0.5,
)
ax[1].plot(
    [dataset["reference_data"]["forces"].min(), dataset["reference_data"]["forces"].max()],
    [dataset["reference_data"]["forces"].min(), dataset["reference_data"]["forces"].max()],
    c="red",
)
ax[1].set_xlabel("True forces")
ax[1].set_ylabel("Predicted forces")
ax[1].set_title("Forces prediction")
plt.show()
