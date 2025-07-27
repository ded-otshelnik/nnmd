import os
import shutil
import warnings

import torch

from nnmd.io import input_parser
from nnmd.nn import BPNN, train_val_test_split
from nnmd.nn.dataset import TrainAtomicDataset
from nnmd.features import calculate_params

warnings.filterwarnings("ignore")
dtype = torch.float32

input_data = input_parser("input/input.yaml")
n_atoms = input_data["atomic_data"]["n_atoms"]
print("Getting info from gpaw file: done")

N1, N2, N3, N4, N5 = 0, 25, 0, 0, 25
r_cutoff = 38.0
params = calculate_params(r_cutoff, N1, N2, N3, N4, N5)
features = [1] * N1 + [2] * N2 + [3] * N3 + [4] * N4 + [5] * N5
input_data["atomic_data"]["symmetry_functions_set"]["Cu"] = {
    "params": params,
    "features": features,
}
input_data["neural_network"]["input_sizes"] = [
    N1 + N2 + N3 + N4 + N5
]  # Atomic NN input_size in hidden layers

# Atomic NN input_size in hidden layers
# convert train data to atomic dataset with symmetry functions
dataset = TrainAtomicDataset.make_atomic_dataset(input_data["atomic_data"])

# ~80% - train, ~10% - test and validation
train_val_test_ratio = (0.8, 0.1, 0.1)
train_dataset, val_dataset, test_dataset = train_val_test_split(
    dataset, train_val_test_ratio
)
print(f"Separate data to train and test datasets: done")

print("Create an instance of NN and config its subnets:", end=" ")

train = input_data["neural_network"]["train"]

# save model params as files in <path> directory
save = input_data["neural_network"]["save"]
path = input_data["neural_network"]["path"]

net = BPNN(dtype=dtype)
net.config(input_data["neural_network"])

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
