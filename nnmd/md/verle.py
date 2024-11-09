import math
import numpy as np

import torch

from ..nn import Neural_Network

from tqdm import tqdm

class MDSimulation:
    def __init__(self, N_atoms: int, cartesians: torch.Tensor, nn: Neural_Network,
                  mass: float, rVan, symm_func_params: dict, L: float, T: float, dt: float,
                  h = float, v_initial = torch.Tensor, a_initial = torch.Tensor):
            self.log = open("md_simulation.log", "w+")

            self.cartesians = cartesians
            self.N_atoms = N_atoms
            self.nn = nn

            # velocity projection
            self.Vx = [v_initial_item[0] for v_initial_item in v_initial]
            self.Vy = [v_initial_item[1] for v_initial_item in v_initial]
            self.Vz = [v_initial_item[2] for v_initial_item in v_initial]
            #print(f"Velocities: {v_initial}", file = self.log)
            # acceleration projection
            self.ax = [a_initial_item[0] for a_initial_item in a_initial]
            self.ay = [a_initial_item[1] for a_initial_item in a_initial]
            self.az = [a_initial_item[2] for a_initial_item in a_initial]

            # derivatives by time
            self.dt = dt
            self.dt1 = dt / 2
            self.dt2 = dt ** 2 / 2

            # mass of atom
            self.mass = mass

            # Van der Waals radius
            self.rVan = rVan

            # size of the box
            self.L = L
            # temperature of system (in Kelvin)
            self.T = T

            # params of symmetric functions
            # r_cutoff - cutoff radius
            # eta - width of the Gaussian
            # k - power of the polynomial
            # rs - width of the Gaussian
            # lambda - power of the polynomial
            # xi - width of the Gaussian
            self.symm_func_params = symm_func_params

            # distance threshold between atoms
            self.L_threshold = -L / self.symm_func_params['r_cutoff']

            self.h = h

            #print(f"Initial correction of velocities", file = self.log)
            self.corr_veloc()
            #print(f"Initial calculation of acceleration", file = self.log)
            self.calc_acel()
            
            self.cartesians_history = [self.cartesians.clone()]
            self.forces_history = [a_initial.clone() * self.mass]
            
    def run_md_simulation(self, steps):
        for step in tqdm(range(steps)):
            #print(f"Step: {step}, start verle", file = self.log)
            self.verle()
            if (step + 1) % 100 == 0:
                #print(f"Step: {step}, correct velocities", file = self.log)
                self.corr_veloc()
            self.cartesians_history.append(self.cartesians.clone())
            acc = torch.as_tensor(list(zip(self.ax, self.ay, self.az)))
            #print(f"Acc: {acc}", file = self.log)
            self.forces_history.append(acc * self.mass)
            
    def corr_veloc(self):
        Vsx = 0
        Vsy = 0
        Vsz = 0

        for i in range(self.N_atoms):
            if (self.Vx[i] * self.Vx[i] + self.Vy[i] * self.Vy[i] + self.Vz[i] * self.Vz[i]) < 1e-8:
                continue
            D = torch.sqrt(1. / (self.Vx[i] * self.Vx[i] + self.Vy[i] * self.Vy[i] + self.Vz[i] * self.Vz[i]))
            #print(f"D{i}: ", D, file = self.log)
            self.Vx[i] *= D 
            self.Vy[i] *= D
            self.Vz[i] *= D
            Vsx += self.Vx[i]
            Vsy += self.Vy[i]
            Vsz += self.Vz[i]
            #print(f"Velocities ({i}): Vsx = {Vsx}, Vsy = {Vsy}, Vsz = {Vsz}", file = self.log)


        Vsx /= self.N_atoms
        Vsy /= self.N_atoms
        Vsz /= self.N_atoms

        for i in range(self.N_atoms):
            self.Vx[i] -= Vsx
            self.Vy[i] -= Vsy
            self.Vz[i] -= Vsz
            #print(f"Velocities ({i}): Vx = {self.Vx[i]}, Vy = {self.Vy[i]}, Vz = {self.Vz[i]}", file = self.log)
        

    def calc_acel(self):
        a = self.nn.predict(self.cartesians,
                            self.symm_func_params,
                            self.h).cpu() / self.mass

        for i in range(self.N_atoms - 1):
             for j in range(i + 1, self.N_atoms):
                dx = self.cartesians[j][0] - self.cartesians[i][0]
                if torch.abs(dx) > self.L_threshold:
                    dx -= np.sign(dx) * self.L
                if torch.abs(dx) > self.symm_func_params['r_cutoff']:
                    continue
                dy = self.cartesians[j][1] - self.cartesians[i][1]
                if torch.abs(dy) > self.L_threshold:
                    dy -= np.sign(dy) * self.L
                if torch.abs(dx) > self.symm_func_params['r_cutoff']:
                    continue
                dz = self.cartesians[j][2] - self.cartesians[i][2]
                if torch.abs(dz) > self.L_threshold:
                    dz -= np.sign(dz) * self.L
                if torch.abs(dz) > self.symm_func_params['r_cutoff']:
                    continue
                r = torch.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
                if r > self.symm_func_params['r_cutoff']:
                    continue
                Nx = dx / r
                Ny = dy / r
                Nz = dz / r
                #print(f"({i},{j}), r: {r}, dx: {dx}, dy: {dy}, dz: {dz},", file = self.log)
                if r < self.rVan:
                    #! check this
                    #print(f"Velocities ({i}): Vx = {self.Vx[i]}, Vy = {self.Vy[i]}, Vz = {self.Vz[i]}", file = self.log)
                    #print(f"Velocities ({j}): Vx = {self.Vx[j]}, Vy = {self.Vy[j]}, Vz = {self.Vz[j]}", file = self.log)
                    #print(f"({i},{j}), (self.Vx[j] - self.Vx[i]): {(self.Vx[j] - self.Vx[i])}, \
                    #       self.Vy[j] - self.Vy[i]: {self.Vy[j] - self.Vy[i]}, \
                    #       self.Vz[j] - self.Vz[i]: {self.Vz[j] - self.Vz[i]}, ", file = self.log)
                    proj = Nx * (self.Vx[j] - self.Vx[i]) + Ny * (
                        self.Vy[j] - self.Vy[i]) + Nz * (self.Vz[j] - self.Vz[i])
                    dr = (self.rVan - r) / 2
                    self.cartesians[i][0] -= Nx * dr
                    self.cartesians[i][1] -= Ny * dr
                    self.cartesians[i][2] -= Nz * dr
                    self.cartesians[j][0] += Nx * dr
                    self.cartesians[j][1] += Ny * dr
                    self.cartesians[j][2] += Nz * dr

                    #print(f"({i},{j}), Nx: {Nx}, Ny: {Ny}, Nz: {Nz}", file = self.log)
                    #print(f"proj ({i},{j}): {proj}", file = self.log)
                    self.Vx[i] += Nx * proj
                    self.Vy[i] += Ny * proj
                    self.Vz[i] += Nz * proj
                    self.Vx[j] -= Nx * proj
                    self.Vy[j] -= Ny * proj
                    self.Vz[j] -= Nz * proj
                    #print(f"Velocities ({i},{j}): {self.Vx}, {self.Vy}, {self.Vz}", file = self.log)
                else:
                    self.ax[i] -= a[i][0] * Nx
                    self.ay[i] -= a[i][1] * Ny
                    self.az[i] -= a[i][2] * Nz
                    self.ax[j] += a[j][0] * Nx
                    self.ay[j] += a[j][1] * Ny
                    self.az[j] += a[j][2] * Nz
        #print("Carts: ", self.cartesians, file = self.log)
        #print("Nx: ", Nx, file = self.log)

    def verle(self):
        #print("Carts: ", self.cartesians, file = self.log)
        #print("Vx: ", self.Vx, file = self.log)
        #print("ax: ", self.ax, file = self.log)
        for i in range(self.N_atoms):
            self.cartesians[i][0] += self.Vx[i] * self.dt + self.ax[i] * self.dt2
            self.cartesians[i][1] += self.Vy[i] * self.dt + self.ay[i] * self.dt2
            self.cartesians[i][2] += self.Vz[i] * self.dt + self.az[i] * self.dt2
            if self.cartesians[i][0] < 0: self.cartesians[i][0] += self.L
            if self.cartesians[i][0] > self.L: self.cartesians[i][0] -= self.L
            if self.cartesians[i][1] < 0: self.cartesians[i][1] += self.L
            if self.cartesians[i][1] > self.L: self.cartesians[i][1] -= self.L
            if self.cartesians[i][2] < 0: self.cartesians[i][2] += self.L
            if self.cartesians[i][2] > self.L: self.cartesians[i][2] -= self.L
            self.Vx[i] += self.ax[i] * self.dt1
            self.Vy[i] += self.ay[i] * self.dt1
            self.Vz[i] += self.az[i] * self.dt1
        #print(self.cartesians, file = self.log)
        self.calc_acel()

        for i in range(self.N_atoms):
            self.Vx[i] += self.ax[i] * self.dt1
            self.Vy[i] += self.ay[i] * self.dt1
            self.Vz[i] += self.az[i] * self.dt1