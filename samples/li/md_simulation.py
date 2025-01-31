# MD simulation of Li atoms with neural network

import torch

from nnmd.md import NNMD_calc
from nnmd.features import calculate_sf, params_for_G2, params_for_G4

from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
import ase.units as units

from ase.io import Trajectory

device = torch.device('cuda')
dtype = torch.float64

# params of symmetric functions
r_cutoff = 5.28
    
params_g2 = [
            [r_cutoff, 0.001, 0.0],
            [r_cutoff, 0.01, 0.0],
            [r_cutoff, 0.03, 0.0],
            [r_cutoff, 0.05, 0.0],
            [r_cutoff, 0.7, 0.0],
            [r_cutoff, 0.1, 0.0],
            [r_cutoff, 0.2, 0.0],
            [r_cutoff, 0.4, 0.0],
            [r_cutoff, 0.5, 0.0],
            [r_cutoff, 0.7, 0.0],
            [r_cutoff, 0.9, 0.0],
            [r_cutoff, 1.0, 0.0],
]
params_g4 = [
            [r_cutoff, 0.01, 4, -1],
            [r_cutoff, 0.01, 4, 1]
]

n_radial = len(params_g2)
n_angular = len(params_g4)

features = [2] * n_radial + [4] * n_angular
params = params_g2 + params_g4
symm_funcs_data = {'features': features, 'params': params}

sample_traj = Trajectory('input/Li_crystal_27.traj')
start = int(len(sample_traj) * 0.8)

class NN:
    def __init__(self):
        from nnmd.nn import AtomicNN
        self.model = AtomicNN(input_size = n_angular + n_radial, hidden_sizes = [64, 64]).float()
        self.model.load_state_dict(torch.load("models/atomic_nn_Li.pth"))

    def predict(self, cartesians, cell, symm_func_params):
        carts = cartesians.unsqueeze(0)
        carts.requires_grad = True
        g, dg = calculate_sf(carts, cell, symm_func_params)
        g = g.squeeze()
        dg = dg.squeeze()

        g.requires_grad = True
        energy = self.model(g)
        de = torch.autograd.grad(energy.sum(), g, create_graph = True)[0]
        forces = -torch.einsum('ij,ijk->ik', de, dg)
        return energy, forces

nn = NN()

# ASE calculator
calc = NNMD_calc(model = nn, to_eV = 1.0, properties = ['energy', 'forces'], symm_funcs_data = symm_funcs_data)
atoms = sample_traj[start]
atoms.calc = calc

traj_file = f'Test_md_Li_27.traj'

# Moving to the MD part
timestep = 1  # time step of the simulation (in fs)

# Integrator for the equations of motion, timestep depends on system
dyn = VelocityVerlet(atoms, timestep * units.fs)
#MaxwellBoltzmannDistribution(atoms, temperature_K = 300)

# Saving the positions of all atoms after every time step
with Trajectory(traj_file, 'w', atoms) as traj:
    dyn.attach(traj.write, interval = 1)
    # Running the simulation for timesteps
    dyn.run(600)