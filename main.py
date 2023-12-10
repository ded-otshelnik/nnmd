from mdnn.nn.neural_network import Neural_Network

from mdnn.util.params_parser import parser

file = 'Cu111.txt'
n_atoms, cartesians, forces, energies = parser(file)

hidden_nodes = [n_atoms[0], 15, 10]
rc = 12.0

eta, rs, k, _lambda, xi = 0.01, 0.5, 1, -1, 3
# n_struct, n_atoms, r_cutoff, hidden_nodes, learning_rate, epochs, mu
net = Neural_Network(len(cartesians), n_atoms[0], rc, hidden_nodes)

net.compile(cartesians, eta, rs, k, _lambda, xi)
net.fit(energies, forces)