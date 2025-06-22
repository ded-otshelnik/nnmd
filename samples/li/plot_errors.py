#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import traceback

import matplotlib.pyplot as plt
import numpy as np

data = {
    "train_e_nn": [],
    "val_e_nn": [],
    "train_f_nn": [],
    "val_f_nn": [],
    "train_total": [],
    "val_total": [],
}

with open("net_adamw.log", "r") as file:
    line = file.readline()
    while line:
        try:
            info = re.findall(r"\d+.\d+e[+-]+\d+", line)
            if len(info) == 0:
                line = file.readline()
                continue
            if line.find("training") != -1:
                e_rmse, f_rmse, total_rmse = info
                data["train_e_nn"].append(float(e_rmse))
                data["train_f_nn"].append(float(f_rmse))
                data["train_total"].append(float(total_rmse))
            elif line.find("validation") != -1:
                info = re.findall(r"\d+.\d+e[+-]+\d+", line)
                e_rmse, f_rmse, total_rmse = info
                data["val_e_nn"].append(float(e_rmse))
                data["val_f_nn"].append(float(f_rmse))
                data["val_total"].append(float(total_rmse))
            line = file.readline()

        except Exception:
            traceback.print_exc()
            exit(1)

shift = 5
data = {key: np.array(value)[shift::2] for key, value in data.items()}
n = len(data["train_e_nn"])

data["train_e_nn"] = [
    (
        data["train_e_nn"][i]
        if np.abs(data["train_e_nn"][i] - data["train_e_nn"][i - 1]) < 0.1
        else data["train_e_nn"][i - 1]
    )
    for i in range(len(data["train_e_nn"]))
]
data["val_e_nn"] = [
    (
        data["val_e_nn"][i]
        if np.abs(data["val_e_nn"][i] - data["val_e_nn"][i - 1]) < 0.1
        else data["val_e_nn"][i - 1]
    )
    for i in range(len(data["val_e_nn"]))
]
data["train_f_nn"] = [
    (
        data["train_f_nn"][i]
        if np.abs(data["train_f_nn"][i] - data["train_f_nn"][i - 1]) < 0.1
        else data["train_f_nn"][i - 1]
    )
    for i in range(len(data["train_f_nn"]))
]
data["val_f_nn"] = [
    (
        data["val_f_nn"][i]
        if np.abs(data["val_f_nn"][i] - data["val_f_nn"][i - 1]) < 0.1
        else data["val_f_nn"][i - 1]
    )
    for i in range(len(data["val_f_nn"]))
]

data["train_e_nn"] = np.array(data["train_e_nn"])
data["val_e_nn"] = np.array(data["val_e_nn"])
data["train_f_nn"] = np.array(data["train_f_nn"])
data["val_f_nn"] = np.array(data["val_f_nn"])

fig, ax = plt.subplots(1, 2, constrained_layout=True, figsize=(24, 8))
ax[0].plot(
    range(shift, 2 * (n + 2), 2),
    data["train_e_nn"],
    marker="o",
    label="MSE, потенциалы, обучающий датасет",
)
ax[0].plot(
    range(shift, 2 * (n + 2), 2),
    data["val_e_nn"],
    marker="o",
    label="MSE, потенциалы, валидационный датасет",
)
ax[0].set_xlim(shift - 2, 2 * (n + 2) + 1)
ax[0].set_ylim(data["val_e_nn"].min() - 1e-3, data["val_e_nn"].max() + 1e-3)
ax[0].set_xlabel("Эпохи", fontsize=20, color="black")
ax[0].set_ylabel("MSE (eV)", fontsize=20, color="black")
ax[0].tick_params(axis="both", which="major", labelsize=16)
ax[0].legend(fontsize=16)

ax[1].plot(
    range(shift, 2 * (n + 2), 2),
    data["train_f_nn"],
    marker="o",
    label="MSE, силы на атомах, обучающий датасет",
)
ax[1].plot(
    range(shift, 2 * (n + 2), 2),
    data["val_f_nn"],
    marker="o",
    label="MSE, силы на атомах, валидационный датасет",
)

ax[1].set_xlabel("Эпохи", fontsize=20, color="black")
ax[1].set_ylabel("MSE (eV/\u212b)", fontsize=20, color="black")
ax[1].set_xlim(shift - 2, 2 * (n + 2) + 1)
ax[1].set_ylim(data["val_f_nn"].min() - 1e-5, data["val_f_nn"].max() + 1e-5)
ax[1].tick_params(axis="both", which="major", labelsize=16)
ax[1].legend(fontsize=16)
fig.tight_layout()

fig.savefig("pictures/plot_errors.png")

fig, ax = plt.subplots(figsize=(10, 9))

ax.plot(
    range(shift, 2 * (n + 2), 2),
    data["train_e_nn"],
    marker="o",
    label="MSE, потенциалы, обучающий датасет",
)
ax.plot(
    range(shift, 2 * (n + 2), 2),
    data["val_e_nn"],
    marker="o",
    label="MSE, потенциалы, валидационный датасет",
)

ax.set_xlabel("Эпохи", fontsize=20, color="black")
ax.set_ylabel("MSE (eV)", fontsize=20, color="black")
ax.tick_params(axis="both", which="major", labelsize=16)
ax.legend(fontsize=18)

fig.tight_layout()
fig.savefig("pictures/plot_potentials.png")

fig, ax = plt.subplots(figsize=(10, 9))

ax.plot(
    range(shift, 2 * (n + 2), 2),
    data["train_f_nn"],
    marker="o",
    label="MSE, силы на атомах, обучающий датасет",
)
ax.plot(
    range(shift, 2 * (n + 2), 2),
    data["val_f_nn"],
    marker="o",
    label="MSE, силы на атомах, валидационный датасет",
)

ax.set_xlabel("Эпохи", fontsize=20, color="black")
ax.set_ylabel("MSE (eV/\u212b)", fontsize=20, color="black")
ax.tick_params(axis="both", which="major", labelsize=16)
ax.legend(fontsize=20)
fig.tight_layout()
fig.savefig("pictures/plot_forces.png")
