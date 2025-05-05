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

with open("net.log", "r") as file:
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

shift = 10
data = {key: np.array(value)[shift:100] for key, value in data.items()}

fig, ax = plt.subplots(2, constrained_layout=True, figsize=(18, 12))

ax[0].plot(
    range(shift, len(data["train_e_nn"]) + shift),
    data["train_e_nn"],
    marker="o",
    label="MSE, potentials, training set",
)
ax[0].plot(
    range(shift, len(data["val_e_nn"]) + shift),
    data["val_e_nn"],
    marker="o",
    label="MSE, potentials, validation set",
)

ax[0].set_xlabel("Iterations", fontsize=20, color="black")
ax[0].set_ylabel("MSE (eV)", fontsize=20, color="black")
ax[0].legend()

ax[1].plot(
    range(shift, len(data["train_f_nn"]) + shift),
    data["train_f_nn"],
    marker="o",
    label="MSE, atomic forces, training set",
)
ax[1].plot(
    range(shift, len(data["val_f_nn"]) + shift),
    data["val_f_nn"],
    marker="o",
    label="MSE, atomic forces, validation set",
)

ax[1].set_xlabel("Iterations", fontsize=20, color="black")
ax[1].set_ylabel("MSE (eV/\u212b)", fontsize=20, color="black")
ax[1].legend()

fig.savefig("plot_errors.png")

fig, ax = plt.subplots(figsize=(10, 9))

ax.plot(
    range(shift, len(data["train_e_nn"]) + shift),
    data["train_e_nn"],
    marker="o",
    label="MSE, potentials, training set",
)
ax.plot(
    range(shift, len(data["val_e_nn"]) + shift),
    data["val_e_nn"],
    marker="o",
    label="MSE, potentials, validation set",
)

ax.set_xlabel("Iterations", fontsize=20, color="black")
ax.set_ylabel("MSE (eV)", fontsize=20, color="black")
ax.legend()


fig.savefig("plot_potentials.png")

fig, ax = plt.subplots(figsize=(10, 9))

ax.plot(
    range(shift, len(data["train_f_nn"]) + shift),
    data["train_f_nn"],
    marker="o",
    label="MSE, atomic forces, training set",
)
ax.plot(
    range(shift, len(data["val_f_nn"]) + shift),
    data["val_f_nn"],
    marker="o",
    label="MSE, atomic forces, validation set",
)

ax.set_xlabel("Iterations", fontsize=20, color="black")
ax.set_ylabel("MSE (eV/\u212b)", fontsize=20, color="black")
ax.legend()

fig.savefig("plot_forces.png")
