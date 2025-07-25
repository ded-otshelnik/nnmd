import warnings

import torch
import numpy as np
import matplotlib.pyplot as plt

from nnmd.io import input_parser
from nnmd.nn import BPNN
from nnmd.features import calculate_params

warnings.filterwarnings("ignore")
dtype = torch.float32

# print("Getting atomic data: ", end="")

# input_data = input_parser("input/input.yaml")

# dataset = input_data["atomic_data"]["reference_data"]

# print("done")

# print("Parsing dataset: ", end="")
# cell = torch.tensor(
#     input_data["atomic_data"]["unit_cell"], dtype=torch.float32, device="cuda"
# )
# cartesians = {
#     "Cu": torch.tensor(
#         np.array([data["Cu"]["positions"] for data in dataset]),
#         dtype=torch.float32,
#         device="cuda",
#     )
# }

# N1, N2, N3, N4, N5 = 0, 25, 0, 0, 25
# r_cutoff = 38.0
# params = calculate_params(r_cutoff, N1, N2, N3, N4, N5)
# features = [1] * N1 + [2] * N2 + [3] * N3 + [4] * N4 + [5] * N5
# input_data['atomic_data']['symmetry_functions_set']['Cu'] = {
#     "params": params,
#     "features": features
# }
# input_data["neural_network"]["input_sizes"] = [
#     N1 + N2 + N3 + N4 + N5
# ]

# print("done")

# print("Configuring neural network: ", end="")
# net = BPNN(dtype=dtype)
# net.config(input_data["neural_network"])

# print("done")

# print("Predicting energies and forces: ", end="")
# predicted_energy, predicted_forces = net.predict(
#     cartesians, cell, input_data["atomic_data"]["symmetry_functions_set"]
# )

# actual_energy = np.array([data["energy"] for data in dataset], dtype=np.float32)
# actual_forces = np.array([data["forces"] for data in dataset], dtype=np.float32)

# print("done")

# np.save("predicted_energy.npy", predicted_energy.cpu().detach().numpy())
# np.save("actual_energy.npy", actual_energy)

predicted_energy = np.load("predicted_energy.npy")
actual_energy = np.load("actual_energy.npy")

predicted_energy = torch.tensor(predicted_energy, dtype=torch.float32, device="cpu")
predicted_energy = predicted_energy.cpu().detach()

actual_energy = torch.tensor(actual_energy, dtype=torch.float32, device="cpu")

actual_energy = (actual_energy - actual_energy.min()) / (
    actual_energy.max() - actual_energy.min()
)

print(predicted_energy)

plt.figure(figsize=(18, 12))
plt.plot(
    range(len(predicted_energy)),
    predicted_energy,
    label="Предсказанная энергия",
    color="blue",
    marker="x",
)

plt.plot(
    range(len(actual_energy)),
    actual_energy,
    label="Исходная энергия",
    color="red",
)
plt.xlabel("Индекс фрейма", fontsize=20)
plt.ylabel("Энергия", fontsize=20)
plt.title("Сравнение предсказанной и фактической энергии", fontsize=24)
plt.tick_params(axis="both", which="major", labelsize=16)
plt.legend(fontsize=20)
plt.grid()
plt.tight_layout()
plt.savefig(f"pictures/energy_comparison_total.png")

plt.figure(figsize=(18, 12))
plt.scatter(actual_energy, predicted_energy, color="blue")
plt.xlabel("Исходная энергия", fontsize=20)
plt.ylabel("Предсказанная энергия", fontsize=20)
plt.title("Сравнение предсказанной и фактической энергии", fontsize=24)
plt.tick_params(axis="both", which="major", labelsize=16)
plt.legend(fontsize=20)
plt.grid()
plt.tight_layout()
plt.savefig(f"pictures/energy_scatter.png")
