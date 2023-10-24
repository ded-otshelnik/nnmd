import numpy as np

from collections.abc import Iterable

from ..util.gdf import gdf
from ..symmetry_functions.sf import calculate_sf, G_TYPE
from ..symmetry_functions.pair_g import PairG

def expit(x):
    """Sigmoid function

    Args:
        x: array of values 
    """
    return 1 / (1 + np.exp(-x))

def expit_mod(x, A, b, c):
    """Modified sigmoid function

    Args:
        x: array of values 
        A: normalization constant
        b, c: constants what determine the function shape
    """
    return A * x / (1 + np.exp(- b * (x - c)))

class NeuralNetwork:
    def __init__(self, n_atoms, g_init, g_iter,
                hidden_nodes,
                eta=0.05, n_iter=1, activation=expit):
        """Create neural network instance

        Args:
            n_atoms: 
            g_init:
            g_iter:
            input_nodes (int): input neurons amount
            hidden_nodes (int | Iterable): hidden neurons amount
            output_nodes (int): output neurons amount
            eta (float, optional): training speed. Defaults to 0.05.
            n_iter (int, optional): epochs number. Defaults to 1.
            activation (optional): activation func on the hidden layer. Defaults to sigmoid function
            
        """

        # TODO: config init parameters of nn
        # number of atoms
        self.n_atoms = n_atoms

        # input neurons amount - g and dg
        self.inodes = 2
        # output neurons amount - energy
        self.onodes = 1
        # weights between input and hidden layers
        sigma = 0.001
        self.wih = np.array(gdf(n_atoms, g_init, g_iter, sigma))

        # hidden layers configuration
        self.hnodes = hidden_nodes
        self.whh = None
        # if 2 or more hidden layers
        if isinstance(self.hnodes, list | np.ndarray | Iterable):
            self.whh = []
            for i in range(len(self.hnodes) - 1):
                # weights between hidden layers
                self.whh.append(np.random.normal(0.0, pow(self.hnodes[i + 1], -0.5), (self.hnodes[i + 1], self.hnodes[i])))
            # weights between last hidden and output layers
            self.who = np.random.normal(0.0 , pow(self.onodes, -0.5), (self.onodes, self.hnodes[len(self.hnodes) - 1]))
        # if only 1 hidden layer
        elif isinstance(self.hnodes, int):
            # weights between hidden and output layers
            self.who = np.random.normal(0.0 , pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        else:
            raise ValueError("Hidden layers configuration must be consided by integer value \
                              or an iterable object of integers")

        # training speed
        self.eta = eta
        # activation function
        self.activation_function = activation 
        # epochs number
        self.n_iter = n_iter

    def train(self, cartesians_train, E_train, eps, r_cutoff):
        # parameters in symmetric functions: eta, rs, k, lambda, xi 
        eta, rs, k, lambda_, xi  = 1, 1, 1, 1, 1, 1
        loss = 10e1
        while loss > eps:
            for cartesians, energy_dft in zip(cartesians_train, E_train):
                energy_nn = 0
                for cartesian in cartesians:
                    pair = PairG(0, 0)
                    pair += calculate_sf(cartesian, cartesians, g_type=1,
                                        r_cutoff=r_cutoff)
                    pair += calculate_sf(cartesian, cartesians, g_type=2,
                                        r_cutoff=r_cutoff, eta=eta, rs = rs)
                    pair += calculate_sf(cartesian, cartesians, g_type=3,
                                        r_cutoff=r_cutoff, k = k)
                    pair += calculate_sf(cartesian, cartesians, g_type=4,
                                        r_cutoff=r_cutoff, eta = eta, xi = xi, lambda_ = lambda_)
                    pair += calculate_sf(cartesian, cartesians, g_type=5,
                                        r_cutoff=r_cutoff, eta = eta, xi = xi, lambda_ = lambda_)
                    for _ in range(self.n_iter):
                        hidden_inputs = np.dot(self.wih, pair.T) 
                
        pass

    def predict():
        pass       
            

