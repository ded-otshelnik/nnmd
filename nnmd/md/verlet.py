import math
import numpy as np

import torch

from ..nn import HDNN

from tqdm import tqdm

class MDSimulation:
    def __init__(self, N_atoms: int, cartesians: np.ndarray, nn: HDNN,
                  mass: float, rVan: float, symm_func_params: dict, L: float, T: float, dt: float,
                  h = float, v_initial = np.ndarray, a_initial = np.ndarray):
            """Initializes MD simulation system

            Args:
                N_atoms (int): number of atoms
                cartesians (np.ndarray): initial positions
                nn (HDNN): neural network
                mass (float): mass of atom
                rVan (float): van der Waals radius
                symm_func_params (dict): params of symmetric functions
                L (float): size of the box
                T (float): temperature of system (in Kelvin)
                dt (float): step of verlet integration
                h (float, optional): step of forces calculation. Defaults to float.
                v_initial (np.ndarray, optional): initial velocities. Defaults to torch.Tensor.
                a_initial (np.ndarray, optional): initial accelerations. Defaults to torch.Tensor.
            """

            self.cartesians = cartesians
            self.N_atoms = N_atoms
            self.nn = nn

            # velocity projection
            self.Vx = v_initial[:, 0]
            self.Vy = v_initial[:, 1]
            self.Vz = v_initial[:, 2]
            
            # acceleration projection
            self.ax = a_initial[:, 0]
            self.ay = a_initial[:, 1]
            self.az = a_initial[:, 2]

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
            
            self.cartesians_history = [self.cartesians.copy()]
            self.forces_history = [a_initial.copy() * self.mass]
            self.velocities_history = [v_initial.copy()]
            
    def run_md_simulation(self, steps):
        for step in tqdm(range(steps)):
            self.verlet()
            if (step + 1) % 100 == 0:
                self.corr_veloc()
            self.cartesians_history.append(self.cartesians.copy())
            self.forces_history.append(np.array(list(zip(self.ax, self.ay, self.az))) * self.mass)
            self.velocities_history.append(list(zip(self.Vx, self.Vy, self.Vz)))
            
    def corr_veloc(self):
        Vsx = 0
        Vsy = 0
        Vsz = 0

        for i in range(self.N_atoms):
            if (self.Vx[i] * self.Vx[i] + self.Vy[i] * self.Vy[i] + self.Vz[i] * self.Vz[i]) < 1e-4:
                continue
            D = np.sqrt(1. / (self.Vx[i] * self.Vx[i] + self.Vy[i] * self.Vy[i] + self.Vz[i] * self.Vz[i]))
            self.Vx[i] *= D 
            self.Vy[i] *= D
            self.Vz[i] *= D
            Vsx += self.Vx[i]
            Vsy += self.Vy[i]
            Vsz += self.Vz[i]

        Vsx /= self.N_atoms
        Vsy /= self.N_atoms
        Vsz /= self.N_atoms

        for i in range(self.N_atoms):
            self.Vx[i] -= Vsx
            self.Vy[i] -= Vsy
            self.Vz[i] -= Vsz
        

    def calc_acel(self):
        _, f = self.nn.predict(torch.tensor(self.cartesians, dtype = torch.float32),
                                self.symm_func_params,
                                self.h)
        a = f.cpu().numpy() / self.mass

        for i in range(self.N_atoms - 1):
             for j in range(i + 1, self.N_atoms):
                dx = self.cartesians[j][0] - self.cartesians[i][0]
                if np.abs(dx) > self.L_threshold:
                    dx -= np.sign(dx) * self.L
                if np.abs(dx) > self.symm_func_params['r_cutoff']:
                    continue
                dy = self.cartesians[j][1] - self.cartesians[i][1]
                if np.abs(dy) > self.L_threshold:
                    dy -= np.sign(dy) * self.L
                if np.abs(dx) > self.symm_func_params['r_cutoff']:
                    continue
                dz = self.cartesians[j][2] - self.cartesians[i][2]
                if np.abs(dz) > self.L_threshold:
                    dz -= np.sign(dz) * self.L
                if np.abs(dz) > self.symm_func_params['r_cutoff']:
                    continue
                r = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
                if r > self.symm_func_params['r_cutoff']:
                    continue
                Nx = dx / r
                Ny = dy / r
                Nz = dz / r
                
                if r < self.rVan:
                    proj = Nx * (self.Vx[j] - self.Vx[i]) + \
                           Ny * (self.Vy[j] - self.Vy[i]) + \
                           Nz * (self.Vz[j] - self.Vz[i])
                    dr = (self.rVan - r) / 2
                    self.cartesians[i][0] -= Nx * dr
                    self.cartesians[i][1] -= Ny * dr
                    self.cartesians[i][2] -= Nz * dr
                    self.cartesians[j][0] += Nx * dr
                    self.cartesians[j][1] += Ny * dr
                    self.cartesians[j][2] += Nz * dr

                    self.Vx[i] += Nx * proj
                    self.Vy[i] += Ny * proj
                    self.Vz[i] += Nz * proj
                    self.Vx[j] -= Nx * proj
                    self.Vy[j] -= Ny * proj
                    self.Vz[j] -= Nz * proj
                else:
                    self.ax[i] -= a[i][0] * Nx
                    self.ay[i] -= a[i][1] * Ny
                    self.az[i] -= a[i][2] * Nz
                    self.ax[j] += a[j][0] * Nx
                    self.ay[j] += a[j][1] * Ny
                    self.az[j] += a[j][2] * Nz

    def verlet(self):
        for i in range(self.N_atoms):
            self.cartesians[i][0] += self.Vx[i] * self.dt + self.ax[i] * self.dt2
            self.cartesians[i][1] += self.Vy[i] * self.dt + self.ay[i] * self.dt2
            self.cartesians[i][2] += self.Vz[i] * self.dt + self.az[i] * self.dt2
            if self.cartesians[i][0] < -self.L: self.cartesians[i][0] += self.L
            if self.cartesians[i][0] > self.L: self.cartesians[i][0] -= self.L
            if self.cartesians[i][1] < -self.L: self.cartesians[i][1] += self.L
            if self.cartesians[i][1] > self.L: self.cartesians[i][1] -= self.L
            if self.cartesians[i][2] < -self.L: self.cartesians[i][2] += self.L
            if self.cartesians[i][2] > self.L: self.cartesians[i][2] -= self.L
            self.Vx[i] += self.ax[i] * self.dt1
            self.Vy[i] += self.ay[i] * self.dt1
            self.Vz[i] += self.az[i] * self.dt1
        self.calc_acel()

        for i in range(self.N_atoms):
            self.Vx[i] += self.ax[i] * self.dt1
            self.Vy[i] += self.ay[i] * self.dt1
            self.Vz[i] += self.az[i] * self.dt1