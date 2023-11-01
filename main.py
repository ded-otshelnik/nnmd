from mdnn.nn.atomic_nn import AtomicNN
from mdnn.nn.neural_network import NeuralNetwork

from mdnn.util.params_parser import parser
from mdnn.symmetry_functions.sf import calculate_sf, G_TYPE
from mdnn.symmetry_functions.pair_g import PairG

from tqdm import tqdm

file = 'example.txt'
n_atoms, cartesians, forces = parser(file)

rc = 3.0
eta = 0.1
rs = 0
k = 1
lambda_ = 1
xi = 1

# net = NeuralNetwork(n_atoms=n_atoms, hidden_nodes=[30, 30], n_iter=4)
# net.fit(cartesians)
