#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import traceback

import numpy as np
import matplotlib.pyplot as plt

data = {
    "train_e_nn": [],
    "val_e_nn": [],
    "train_f_nn": [],
    "val_f_nn": [],
    "train_total": [],
    "val_total": []
}

with open("net.log",'r') as file:
    line = file.readline()
    while line:
        try:
            info = re.findall(r'\d+.\d+e[+-]+\d+', line)
            if len(info) == 0:
                line = file.readline()
                continue
            if (line.find("training") != -1):
                e_rmse, f_rmse, total_rmse = info
                data["train_e_nn"].append(float(e_rmse))
                data["train_f_nn"].append(float(f_rmse))
                data["train_total"].append(float(total_rmse))
            elif (line.find("validation") != -1):
                info = re.findall(r'\d+.\d+e[+-]+\d+', line)
                e_rmse, f_rmse, total_rmse = info
                data["val_e_nn"].append(float(e_rmse))
                data["val_f_nn"].append(float(f_rmse))
                data["val_total"].append(float(total_rmse))
            line = file.readline()    

        except Exception:
            traceback.print_exc()
            exit(1)

data = {key: np.array(value) for key, value in data.items()}

fig, ax = plt.subplots(constrained_layout = True, figsize = (10, 6))
ax.set_title("Среднеквадратичная ошибка (СКО) при обучении")

# the main plot
ax.plot(range(5, len(data['train_e_nn'])), data['train_e_nn'][5:], marker = 'o', label = "СКО, потенциалы, обучающая выборка")
ax.plot(range(5, len(data['val_e_nn'])), data['val_e_nn'][5:], marker = 'o', label = "СКО, потенциалы, валидационная выборка")

ax.set_xlabel("Итерации", fontsize = 15, color = 'black')
ax.set_ylabel("СКО (эВ/\u212b)", fontsize = 15, color = 'black')
ax.legend()

# save as png file
fig.savefig("plot_potentials.png")

fig, ax = plt.subplots(constrained_layout = True, figsize = (10, 6))
ax.set_title("Среднеквадратичная ошибка (СКО) при обучении")

# the main plot
ax.plot(range(len(data['train_f_nn'])), data['train_f_nn'], marker = 'o',label = "СКО, силы на атомах, обучающая выборка")
ax.plot(range(len(data['val_f_nn'])), data['val_f_nn'], marker = 'o',label = "СКО, силы на атомах, валидационная выборка")

ax.set_xlabel("Итерации", fontsize = 15, color = 'black')
ax.set_ylabel("СКО (эВ/\u212b)", fontsize = 15, color = 'black')
ax.legend()

# save as png file
fig.savefig("plot_forces.png")