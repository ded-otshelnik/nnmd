import warnings

import torch
import numpy as np
import matplotlib.pyplot as plt

from nnmd.io import input_parser
from nnmd.nn import BPNN

warnings.filterwarnings("ignore")
dtype = torch.float32

print("Getting atomic data: ", end="")

input_data = input_parser("input/input.yaml")

dataset = input_data["atomic_data"]["reference_data"]

print("done")

print("Parsing dataset: ", end="")
cell = torch.tensor(input_data["atomic_data"]["unit_cell"], dtype=torch.float32, device="cuda")
cartesians = {
    "Li": torch.tensor(
        np.array([data["Li"]["positions"] for data in dataset]),
        dtype=torch.float32,
        device="cuda",
    )
}

print("done")

print("Configuring neural network: ", end="")
net = BPNN(dtype=dtype)
net.config(input_data["neural_network"])

print("done")

print("Predicting energies and forces: ", end="")
predicted_energy, predicted_forces = net.predict(cartesians, cell, input_data["atomic_data"]["symmetry_functions_set"])

actual_energy = np.array([data["energy"] for data in dataset], dtype=np.float32)
actual_forces = np.array([data["forces"] for data in dataset], dtype=np.float32)

print("done")

np.save("predicted_energy.npy", predicted_energy.cpu().detach().numpy())
np.save("actual_energy.npy", actual_energy)

predicted_energy = np.load("predicted_energy.npy")
actual_energy = np.load("actual_energy.npy")

predicted_energy = torch.tensor(predicted_energy, dtype=torch.float32, device="cpu")
predicted_energy = predicted_energy.cpu().detach()

actual_energy = torch.tensor(actual_energy, dtype=torch.float32, device="cpu")

actual_energy = (actual_energy - actual_energy.min()) / (actual_energy.max() - actual_energy.min())
predicted_energy = (predicted_energy - predicted_energy.min()) / (predicted_energy.max() - predicted_energy.min())

for i in range(0, len(actual_energy), 1054):
    if i + 1054 > len(actual_energy):
        i -= 528
    plt.figure(figsize=(18, 12))
    plt.plot(
        range(i, i + 1054),
        predicted_energy[i:i + 1054],
        label="Предсказанная энергия",
        color='blue',
        marker="x"
    )

    plt.plot(
        range(i, i + 1054),
        actual_energy[i:i + 1054],
        label="Исходная энергия",
        color='red'
    )
    plt.xlabel("Индекс фрейма", fontsize=20)
    plt.ylabel("Энергия", fontsize=20)
    plt.title("Сравнение предсказанной и фактической энергии", fontsize=24)
    plt.xlim(i, i + 1054)
    plt.ylim(predicted_energy[i:i + 1054].min()-1e-3, predicted_energy[i:i + 1054].max()+1e-3)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.legend(fontsize=20)
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"pictures/energy_comparison_{i}.png")

plt.figure(figsize=(18, 12))
plt.plot(
        range(0, len(predicted_energy), 22),
        predicted_energy[::22],
        label="Предсказанная энергия",
        color='blue',
        marker="x"
)

plt.plot(
        range(0, len(actual_energy), 44),
        actual_energy[::44],
        label="Исходная энергия",
        color='red'
)
plt.xlabel("Индекс фрейма", fontsize=20)
plt.ylabel("Энергия", fontsize=20)
plt.title("Сравнение предсказанной и фактической энергии", fontsize=24)
plt.ylim(predicted_energy[::44].min()-1e-3, predicted_energy[::44].max()+1e-3)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.legend(fontsize=20)
plt.grid()
plt.tight_layout()
plt.savefig(f"pictures/energy_comparison_total.png")

plt.figure(figsize=(18, 12))
plt.scatter(
        actual_energy,
        predicted_energy,
        color='blue'
)
plt.xlabel("Исходная энергия", fontsize=20)
plt.ylabel("Предсказанная энергия", fontsize=20)
plt.xlim(actual_energy.min()-1e-3, actual_energy.max()+1e-3)
plt.ylim(predicted_energy.min()-1e-3, predicted_energy.max()+1e-3)
plt.title("Сравнение предсказанной и фактической энергии", fontsize=24)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.legend(fontsize=20)
plt.grid()
plt.tight_layout()
plt.savefig(f"pictures/energy_scatter.png")
