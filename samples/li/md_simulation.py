# MD simulation of Li atoms with neural network

import torch

from nnmd.md import NNMD_calc
from nnmd.features import calculate_sf, params_for_G2, params_for_G4

from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
import ase.units as units
import ase.data as ad

from ase.io import Trajectory

device = torch.device('cuda')
dtype = torch.float64

# params of symmetric functions
r_cutoff_g2 = 4.0
r_cutoff_g4 = 4.0
n_radial = 4
n_angular = 12

features = [2] * n_radial + [4] * n_angular
params = params_for_G2(n_radial, r_cutoff_g2) + params_for_G4(n_angular, r_cutoff_g4)
symm_funcs_data = {'features': features, 'params': params}

# mass of atom (in atomic mass units)
m_atom = ad.atomic_masses[ad.atomic_numbers['Li']]
# temperature of system (in Kelvin)
T = 300.0
# size of the box
L = 9.0
# Van der Waals radius
rVan = 1.44
dt = 10e-3

sample_traj = Trajectory('input/Li_crystal_27.traj')
start = int(len(sample_traj) * 0.8)

class NN:
    def __init__(self):
        from nnmd.nn import AtomicNN
        self.model = AtomicNN(input_size = n_angular + n_radial, hidden_size = [48, 48]).float()
        self.model.load_state_dict(torch.load("models/atomic_nn_Li.pth"))

    def forward(self, cartesians, symm_func_params, h = None):
        carts = cartesians.unsqueeze(0)
        carts.requires_grad = True
        g, dg = calculate_sf(carts, symm_func_params)
        g = g.squeeze()
        dg = dg.squeeze()

        energy = self.model(g)
        de = torch.autograd.grad(energy.sum(), g, create_graph = True)[0]
        forces = -torch.einsum('ijk,ij->ik', dg, de)
        return energy, forces

nn = NN()

# ASE calculator
calc = NNMD_calc(model = nn, to_eV = 1.0, properties = ['energy', 'forces'], symm_funcs_data = symm_funcs_data)
atoms = sample_traj[0]
atoms.calc = calc

traj_file = f'Test_md_Li_27.traj'

# Moving to the MD part
timestep = 1  # time step of the simulation (in fs)

# Integrator for the equations of motion, timestep depends on system
dyn = VelocityVerlet(atoms, timestep * units.fs)
MaxwellBoltzmannDistribution(atoms, temperature_K = 300)

# Saving the positions of all atoms after every time step
with Trajectory(traj_file, 'w', atoms) as traj:
    dyn.attach(traj.write, interval = 1)
    # Running the simulation for timesteps
    dyn.run(600)