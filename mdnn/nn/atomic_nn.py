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

        # input neurons amount - count of cartesians in each structure
        self.inodes = n_atoms
        # output neurons amount - parameters of descriptors
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

        # log file
        self.log = open('nn.txt','w+')

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
        """Calculates energy for each structure.

        Args:
            cartesians: coordinates of atoms in each structure
        """
        # energy array
        energy = []
        # derivatives array
        derivatives = []

        # loop by structures
        for struct in cartesians:
            e_struct = []
            de_struct = []
            # loop by atom
            for atom in struct:
                # atom descriptors and its derivatives 
                e_atom = []
                de_atom = []
                # calculate descriptors for atom with parameters
                for i in [1, 2, 3, 4, 5]:
                    gi = calculate_sf(atom, cartesians[0], G_TYPE(i),
                                            r_cutoff=self.r_cutoff, eta=self.eta, rs=self.rs, k=self.k, lambda_=self.lambda_, xi=self.xi)
                    e_atom.append(gi.g)
                    de_atom.append(gi.dg)
                e_struct.append(e_atom) 
                de_struct.append(de_atom) 

            # scale of energies
            min_ = [min(e_struct[:][i]) for i in range(5)]
            max_ = [max(e_struct[:][i]) for i in range(5)]
            scaled_energy = sum([sum([(2 * g[i] - min_[i]) / (max_[i] - min_[i]) - 1 for i in range(5)]) for g in e_struct])
            energy.append(scaled_energy)

            # scale of energies derivatives 
            min_ = [min(de_struct[:][i]) for i in range(5)]
            max_ = [max(de_struct[:][i]) for i in range(5)]
            scaled_derivative = sum([sum([(2 * dg[i] - min_[i]) / (max_[i] - min_[i]) - 1 for i in range(5)]) for dg in de_struct])
            derivatives.append(scaled_derivative)
        
        return energy, derivatives
        
    def calculate_forces(cartesians):
        h = 0.01
        pass


    def fit(self, cartesians, e_train, f_train, eps):
        """_summary_

        Args:
            cartesians (_type_): _description_
            e_train (_type_): _description_
            f_train (_type_): _description_
            eps (_type_): _description_
        """
        X = np.array(cartesians, ndmin=2).T
        y = np.array(e_train)

        for _ in range(self.n_iter):
            hidden_input_hist = []
            hidden_output_hist = []

            hidden_inputs = np.dot(self.wih, X)
            hidden_outputs = self.activation_function(hidden_inputs)

            hidden_input_hist.append(hidden_inputs)
            hidden_output_hist.append(hidden_outputs)

            for i in range(len(self.whh)):
                hidden_inputs = np.dot(self.whh[i], hidden_outputs)
                hidden_outputs = self.activation_function(hidden_inputs)

                hidden_input_hist.append(hidden_inputs)
                hidden_output_hist.append(hidden_outputs)

            final_inputs = np.dot(self.who, hidden_outputs)
            final_outputs = self.activation_function(final_inputs)

            self.eta, self.k, self.rs, self.k, self.lambda_, self.xi = final_outputs
            self.lambda_ = np.where(self.lambda_ < 0, -1, 1)

            e_nnp, de_nnp = self.calculate_energy(cartesians)
            # f_nnp = self.calculate_forces(cartesians)
            loss = self.loss(e_train, e_nnp, f_train, np.full((len(f_train), len(f_train[0])), 0))
            
            # TODO: config weights correction   
            # ? get loss of net by derivatives by parameters      

            # get errors by gradient 

            # update weights between hidden and output layers
            # self.who += self.eta * np.dot(output_errors * final_outputs * (1.0 - final_outputs), np.transpose(hidden_outputs))

            # some update in hidden layers
            # for ...

            # update weights between input and hidden layers
            # self.wih += self.eta * np.dot(hidden_errors * hidden_outputs * (1.0 - hidden_outputs), np.transpose(X))  

    def predict(self, point, cartesians):
        pass