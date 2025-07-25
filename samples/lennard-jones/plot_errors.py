#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import traceback

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd

data = {
    "train_e_nn": [],
    "val_e_nn": [],
    "train_f_nn": [],
    "val_f_nn": [],
    "train_total": [],
    "val_total": [],
}

with open("output/net_dimer.log", "r") as file:
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

shift = 30
data = {key: np.array(value)[shift:400] for key, value in data.items()}

fig, ax = plt.subplots(2, constrained_layout=True, figsize=(18, 12))

ax[0].plot(
    range(shift, len(data["train_e_nn"]) + shift),
    data["train_e_nn"],
    marker="o",
    label="MSE, потенциалы, обучающий датасет",
)
ax[0].plot(
    range(shift, len(data["val_e_nn"]) + shift),
    data["val_e_nn"],
    marker="o",
    label="MSE, потенциалы, валидационный датасет",
)

ax[0].set_xlabel("Эпохи", fontsize=20, color="black")
ax[0].set_ylabel("MSE (eV)", fontsize=20, color="black")
ax[0].tick_params(axis="both", which="major", labelsize=16)
ax[0].ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))
ax[0].legend(fontsize=16)

ax[1].plot(
    range(shift, len(data["train_f_nn"]) + shift),
    data["train_f_nn"],
    marker="o",
    label="MSE, атомные силы, обучающий датасет",
)
ax[1].plot(
    range(shift, len(data["val_f_nn"]) + shift),
    data["val_f_nn"],
    marker="o",
    label="MSE, атомные силы, валидационный датасет",
)

ax[1].set_xlabel("Эпохи", fontsize=20, color="black")
ax[1].set_ylabel("MSE (eV/\u212b)", fontsize=20, color="black")
ax[1].tick_params(axis="both", which="major", labelsize=16)
ax[1].ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))
ax[1].legend(fontsize=16)

fig.savefig("plot_errors_.png")

fig, ax = plt.subplots(figsize=(10, 9))

ax.plot(
    range(shift, len(data["train_e_nn"]) + shift),
    data["train_e_nn"],
    marker="o",
    label="MSE, потенциалы, обучающий датасет",
)
ax.plot(
    range(shift, len(data["val_e_nn"]) + shift),
    data["val_e_nn"],
    marker="o",
    label="MSE, потенциалы, валидационный датасет",
)

ax.set_xlabel("Эпохи", fontsize=20, color="black")
ax.set_ylabel("MSE (eV)", fontsize=20, color="black")
ax.legend(fontsize=16)
ax.tick_params(axis="both", which="major", labelsize=16)
ax.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))


fig.savefig("plot_potentials.png")

fig, ax = plt.subplots(figsize=(10, 9))

ax.plot(
    range(shift, len(data["train_f_nn"]) + shift),
    data["train_f_nn"],
    marker="o",
    label="MSE, атомные силы, обучающий датасет",
)
ax.plot(
    range(shift, len(data["val_f_nn"]) + shift),
    data["val_f_nn"],
    marker="o",
    label="MSE, атомные силы, валидационный датасет",
)

ax.set_xlabel("Эпохи", fontsize=20, color="black")
ax.set_ylabel("MSE (eV/\u212b)", fontsize=20, color="black")
ax.legend(fontsize=16)
ax.tick_params(axis="both", which="major", labelsize=16)
ax.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))

fig.savefig("plot_forces.png")
