from mdnn.nn.neural_network import NeuralNetwork

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
# n_struct, n_atoms, r_cutoff, hidden_nodes, learning_rate, epochs, mu
net = NeuralNetwork(len(cartesians), n_atoms[0], rc, 30)
net.fit(cartesians, energies, forces)