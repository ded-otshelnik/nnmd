from mdnn.nn.atomic_nn import AtomicNN

from mdnn.util.params_parser import parser

import numpy as np

def expit_mod(x, A, b, c):
    """Modified sigmoid function

    Args:
        x: array of values 
        A: normalization constant
        b, c: constants what determine the function shape
    """
    return A * x / (1 + np.exp(- b * (x - c)))


file = 'Cu111.txt'
n_atoms, cartesians, forces, energies = parser(file)

hidden_nodes = 30
rc = 12.0

params = []

for n_atoms_struct, struct, e_struct, f_struct in zip(n_atoms, cartesians, energies, forces):
    net = AtomicNN(n_atoms_struct, log=open('nn.txt','w+'), hidden_nodes=hidden_nodes, r_cutoff=rc)
    net.fit(struct, e_struct, f_struct)
    params.append(net.get_params())