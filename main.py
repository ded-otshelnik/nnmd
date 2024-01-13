import os
import shutil
from mdnn.nn.neural_network import Neural_Network

from mdnn.util.params_parser import parser

file = './samples/Cu111.txt'
n_atoms, cartesians, forces, energies = parser(file)

hidden_nodes = [n_atoms[0], 15, 10]
rc = 12.0

eta, rs, k, _lambda, xi = 0.01, 0.5, 1, -1, 3
# n_struct, n_atoms, r_cutoff, hidden_nodes, learning_rate, epochs, mu - params

net = Neural_Network(len(cartesians), n_atoms[0], rc, hidden_nodes)
net.compile(cartesians, eta, rs, k, _lambda, xi, load_models = False, path = 'models')

net.fit(energies, forces)
if os.path.exists('models'):
    shutil.rmtree('models', ignore_errors=True)
os.mkdir('models')
net.save_model()