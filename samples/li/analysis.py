#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import traceback

import matplotlib.pyplot as plt

iterations, e_nn_rmse, f_nn_rmse, total_nn_rmse = [], [], [], []

counter = 1
with open("net.log",'r') as file:
    line = file.readline()
    while line:
        try:
            # get numbers in scientific format (like 1.2e+3 = 1200.0)
            if (line.find("training") != -1):
                line = file.readline()    
                continue
            info = re.findall(r'\d+.\d+e[+-]+\d+', line)

            if len(info) != 0:
                e_rmse, f_rmse, total_rmse = info
                iterations.append(counter)
                e_nn_rmse.append(float(e_rmse))
                f_nn_rmse.append(float(f_rmse))
                total_nn_rmse.append(float(total_rmse))
                counter += 1

            line = file.readline()    

        except Exception:
            traceback.print_exc()
            exit(1)

fig, ax = plt.subplots(constrained_layout = True, figsize = (10, 6))
ax.set_title("Среднеквадратичная ошибка (СКО) при обучении")

# the main plot
ax.plot(iterations[50:], e_nn_rmse[50:], color = 'red', marker = 'o', label = "СКО, потенциалы")

ax.set_xlabel("Итерации", fontsize = 15, color = 'black')
ax.set_ylabel("СКО (эВ/\u212b)", fontsize = 15, color = 'black')
ax.legend()

# save as png file
fig.savefig("plot_potentials.png")

fig, ax = plt.subplots(constrained_layout = True, figsize = (10, 6))
ax.set_title("Среднеквадратичная ошибка (СКО) при обучении")

# the main plot
ax.plot(iterations[50:], f_nn_rmse[50:], color = 'red', marker = 'o',label = "СКО, силы на атомах")

ax.set_xlabel("Итерации", fontsize = 15, color = 'black')
ax.set_ylabel("СКО (эВ/\u212b)", fontsize = 15, color = 'black')
ax.legend()

# save as png file
fig.savefig("plot_forces.png")