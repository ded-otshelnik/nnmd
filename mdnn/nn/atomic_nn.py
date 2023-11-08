from math import isclose
import numpy as np

from collections.abc import Iterable

from ..symmetry_functions.sf import calculate_sf, G_TYPE
from ..util.gdf import gdf

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

class AtomicNN(object):

    def __init__(self, n_atoms, hidden_nodes, r_cutoff, learning_rate = 0.001, n_iter = 100, mu=0):
        # number of atoms in each structure
        self.N = n_atoms
        # number of structures
        self.M = len(n_atoms)
        self.learning_rate = learning_rate
        # number of used symmetry functions
        self.descriptors_amount = 5
        # number of training epochs
        self.n_iter = n_iter
        # coeff of force importance in loss
        self.mu = mu

        # parameters of symmetric functions
        self.r_cutoff, self.eta, self.k, self.rs, self.k, self.lambda_, self.xi = r_cutoff, 0.001, 1, -1, 1

        # input neurons amount - count of cartesians
        self.inodes = n_atoms
        # output neurons amount - energy
        self.onodes = 5
        # hidden layers configuration
        self.hnodes = hidden_nodes

        self.whh = None
        # if 2 or more hidden layers
        if isinstance(self.hnodes, list | np.ndarray | Iterable):
            self.wih = np.random.normal(0.0 , pow(self.onodes, -0.5), (self.hnodes[0], self.inodes))
            self.whh = []
            for i in range(len(self.hnodes) - 1):
                # weights between hidden layers
                self.whh.append(np.random.normal(0.0, pow(self.hnodes[i + 1], -0.5), (self.hnodes[i + 1], self.hnodes[i])))
            # weights between last hidden and output layers
            self.who = np.random.normal(0.0 , pow(self.onodes, -0.5), (self.onodes, self.hnodes[len(self.hnodes) - 1]))
        # if only 1 hidden layer
        elif isinstance(self.hnodes, int) and self.hnodes > 0:
            self.wih = np.random.normal(0.0 , pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
            # weights between hidden and output layers
            self.who = np.random.normal(0.0 , pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        else:
            raise ValueError("Hidden layers configuration must be consided by positive integer value \
                              or an iterable object of positive integers")

        # training speed
        self.learning_rate = learning_rate
        # activation function
        self.activation_function = expit
        # epochs number
        self.n_iter = n_iter

    def loss(self, e_dft, e_nnp, f_dft, f_nnp):
        """RMSE(Energy)^2 + (mu/3) * RMSE(Force)^2

        Args:
            mu (_type_): _description_
            e_dft (_type_): _description_
            e_nnp (_type_): _description_
            f_dft (_type_): _description_
            f_nnp (_type_): _description_
            M (_type_): _description_
            N (_type_): _description_
        """
        return (1 / self.M) * sum([((e_dft_i - e_nnp_i) / Ni) ** 2 for e_dft_i, e_nnp_i, Ni in zip(e_dft, e_nnp, self.N)]) \
            + (self.mu / (3 * sum(self.N))) \
            * sum([ sum([(fij_dft - fij_nnp) ** 2 for fij_dft, fij_nnp in zip(fi_dft, fi_nnp)]) for fi_dft, fi_nnp in zip(f_dft, f_nnp)])

    def calculate_energy(self, cartesians)-> list[float]:
        energy = []
        derivatives = []
        for struct in cartesians:
            e_struct = []
            de_struct = []
            for point in struct:
                e_point = []
                de_point = []
                for i in [1, 2, 3, 4, 5]:
                    gi = calculate_sf(point, cartesians[0], G_TYPE(i),
                                            r_cutoff=self.r_cutoff, eta=self.eta, rs=self.rs, k=self.k, lambda_=self.lambda_, xi=self.xi)
                    e_point.append(gi.g)
                    de_point.append(gi.dg)
                e_struct.append(e_point) 
                de_struct.append(de_point) 


            min_ = [min(e_struct[:][i]) for i in range(5)]
            max_ = [max(e_struct[:][i]) for i in range(5)]
            scaled_energy = sum([sum([(2 * g[i] - min_[i]) / (max_[i] - min_[i]) - 1 for i in range(5)]) for g in e_struct])
            energy.append(scaled_energy)

            min_ = [min(de_struct[:][i]) for i in range(5)]
            max_ = [max(de_struct[:][i]) for i in range(5)]
            scaled_derivative = sum([sum([(2 * g[i] - min_[i]) / (max_[i] - min_[i]) - 1 for i in range(5)]) for g in de_struct])
            derivatives.append(scaled_derivative)
        
        return energy, derivatives
        
    def calculate_forces(cartesians):
        h = 0.01
        pass


    def fit(self, cartesians, e_train, f_train, eps):
        
        e_nnp = self.calculate_energy(cartesians)
        f_nnp = self.calculate_forces(cartesians)
        while not np.isclose(e_nnp, e_train, atol=eps, rtol=0).all():
                pass
            
        pass
            

    def predict(self, point, cartesians):
        pair = calculate_sf(point, cartesians, G_TYPE(1))
        for i in range(1, 5):
            pair += calculate_sf(point, cartesians, G_TYPE(i + 1), self.r_cutoff,
                            self.eta, self.rs, self.k, self.lambda_, self.xi)
        return pair.g