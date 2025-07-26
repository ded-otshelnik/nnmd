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

# print("Getting Lennard-Jones potential: ", end="")

# input_data = input_parser("input/input.yaml")

# print("done")

# # convert train data to atomic dataset with symmetry functions
# dataset = make_atomic_dataset(input_data["atomic_data"], saved=True)

# dataset = input_data["atomic_data"]["reference_data"]

# # get cartesian coordinates for each species
# cartesians = {
#     spec: torch.tensor(
#         np.array([data[spec]["positions"] for data in dataset]),
#         dtype=torch.float32,
#         device="cuda",
#     )
#     for spec in dataset[0].keys()
#     if spec not in ["forces", "energy", "velocities"]
# }
# cell = torch.tensor(
#     input_data["atomic_data"]["unit_cell"], dtype=torch.float32, device="cuda"
# )

# net = BPNN(dtype=dtype)
# net.config(input_data["neural_network"])

# predicted_energy, predicted_forces = net.predict(
#     cartesians, cell, input_data["atomic_data"]["symmetry_functions_set"]
# )

# actual_energy = np.array([data["energy"] for data in dataset], dtype=np.float32)
# actual_forces = np.array([data["forces"] for data in dataset], dtype=np.float32)

# distances = (
#     torch.norm(cartesians["H"][:, 0, :] - cartesians["H"][:, 1, :], dim=-1, p=2)
#     .cpu()
#     .detach()
#     .numpy()
# )

# actual_energy = (actual_energy - actual_energy.min()) / (actual_energy.max() - actual_energy.min())
# predicted_energy = predicted_energy.cpu().detach().numpy()

# np.savez("lennard_jones_energy.npz", distances=distances, predicted_energy=predicted_energy, actual_energy=actual_energy)

data = np.load("lennard_jones_energy.npz")
distances = data["distances"]
predicted_energy = data["predicted_energy"]
actual_energy = data["actual_energy"]

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].plot(
    distances,
    predicted_energy,
    c="red",
    label="Предсказанная энергия",
)
ax[0].scatter(
    distances,
    actual_energy,
    label="Фактическая энергия",
)

ax[0].set_xlabel("Межатомное расстояние (Å)")
ax[0].set_ylabel("Энергия (eV)")
ax[0].set_title("Потенциал Леннарда-Джонса для димера")
ax[0].legend()
ax[0].grid()

ax[1].scatter(
    predicted_energy,
    actual_energy,
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
plt.savefig("lennard_jones_energy_plot_full.png")

fig, ax = plt.subplots(figsize=(15, 10))
ax.plot(
    distances[::100],
    predicted_energy[::100],
    c="red",
    label="Предсказанная энергия",
)
ax.scatter(
    distances[::100],
    actual_energy[::100],
    label="Фактическая энергия",
)

ax.set_xlabel("Межатомное расстояние (Å)", fontsize=20, color="black")
ax.set_ylabel("Энергия (eV)", fontsize=20, color="black")
ax.set_title("Потенциал Леннарда-Джонса для тримера", fontsize=20, color="black")
ax.legend(fontsize=16)
ax.tick_params(axis="both", which="major", labelsize=16)
ax.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))
ax.grid()
fig.tight_layout()
fig.savefig("lennard_jones_energy.png")
