import warnings

import torch
import numpy as np
import matplotlib.pyplot as plt

from nnmd.io import input_parser
from nnmd.nn import BPNN
from nnmd.nn.dataset import make_atomic_dataset

import torch

from nnmd.nn import BPNN
from nnmd.nn.dataset import make_atomic_dataset

warnings.filterwarnings("ignore")
dtype = torch.float32

print("Getting Lennard-Jones potential: ", end="")

input_data = input_parser("input/input.yaml")

print("done")

# convert train data to atomic dataset with symmetry functions
dataset = make_atomic_dataset(input_data["atomic_data"], disable_tqdm=True)

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

net = BPNN(dtype=dtype)
net.config(input_data["neural_network"])

predicted_energy, predicted_forces = net.predict(
    cartesians, cell, input_data["atomic_data"]["symmetry_functions_set"]
)

actual_energy = np.array([data["energy"] for data in dataset], dtype=np.float32)
actual_forces = np.array([data["forces"] for data in dataset], dtype=np.float32)

distances = (
    torch.norm(cartesians["H"][:, 0, :] - cartesians["H"][:, 1, :], dim=-1, p=2)
    .cpu()
    .detach()
    .numpy()
)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].plot(
    distances,
    predicted_energy.sum(dim=-1).cpu().detach().numpy(),
    c="red",
    label="Predicted Energy",
)
ax[0].scatter(
    distances,
    (actual_energy - actual_energy.min()) / (actual_energy.max() - actual_energy.min()),
    label="Actual Energy",
)

ax[0].set_xlabel("Distance (Å)")
ax[0].set_ylabel("Energy")
ax[0].set_title("Energy vs Distance for Lennard-Jones Dimer")
ax[0].legend()
ax[0].grid()

ax[1].scatter(
    predicted_energy.sum(dim=-1).cpu().detach().numpy(),
    (actual_energy - actual_energy.min()) / (actual_energy.max() - actual_energy.min()),
    c="red",
    marker="x",
    label="Actual Energy",
)
ax[1].set_xlabel("Predicted Energy")
ax[1].set_ylabel("Actual Energy")
ax[1].set_title("Predicted vs Actual Energy")
ax[1].legend()
ax[1].grid()

plt.tight_layout()
plt.savefig("lennard_jones_energy_plot.png")
