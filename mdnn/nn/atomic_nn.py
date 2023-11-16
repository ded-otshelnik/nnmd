import copy
import numpy as np

from collections.abc import Iterable
from collections import deque

from scipy.special import softmax

from ..symmetry_functions.sf import calculate_dg_by_params, calculate_sf, G_TYPE

def expit(x):
    """Sigmoid function

    Args:
        x: array of values 
    """
    return 1. / (1. + np.exp(-x))

def expit_mod(x, A, b, c):
    """Modified sigmoid function

    Args:
        x: array of values 
        A: normalization constant
        b, c: constants what determine the function shape
    """
    return A * x / (1 + np.exp(- b * (x - c)))

class AtomicNN(object):

    def __init__(self, n_atoms, hidden_nodes, log, r_cutoff, learning_rate = 0.1, n_iter = 100, mu=0):
        # number of atoms in each structure
        self.N = n_atoms

        self.learning_rate = learning_rate
        # number of used symmetry functions
        self.descriptors_amount = 5
        # number of training epochs
        self.n_iter = n_iter
        # coeff of force importance in loss
        self.mu = mu
        # cutoff radius
        self.r_cutoff = r_cutoff

        # input neurons amount - count of cartesians in structure
        self.inodes = 3 * n_atoms
        # output neurons amount - parameters of descriptors
        self.onodes = 5
        # hidden layers configuration
        self.hnodes = hidden_nodes

        self.whh = None
        # if 2 or more hidden layers
        if isinstance(self.hnodes, list | np.ndarray | Iterable):
            self.wih = np.random.normal(0.0 , pow(self.hnodes[0], -1.0), (self.hnodes[0], self.inodes))
            self.whh = []
            for i in range(1, len(self.hnodes)):
                # weights between hidden layers
                self.whh.append(np.random.normal(0.0, pow(self.hnodes[i], -1.0), (self.hnodes[i], self.hnodes[i - 1])))
            # weights between last hidden and output layers
            self.who = np.random.normal(0.0, pow(self.onodes, -1.0), (self.onodes, self.hnodes[len(self.hnodes) - 1]))
        # if only 1 hidden layer
        elif isinstance(self.hnodes, int) and self.hnodes > 0:
            self.wih = np.random.normal(0.0, pow(self.hnodes, -1.0), (self.hnodes, self.inodes))
            # weights between hidden and output layers
            self.who = np.random.normal(0.0, pow(self.onodes, -0.25), (self.onodes, self.hnodes))
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
        self.log = log

    def loss(self, iter, e_dft, e_nnp, f_dft, f_nnp):
        """Gets loss of neural network: RMSE(Energy)^2 + (mu/3) * RMSE(Force)^2

        Args:
            mu: importance coefficient of force
            e_dft: dft energy
            e_nnp: energy that is calculated by network
            f_dft: dft energy
            f_nnp: force that is calculated by network
            N: number of atoms in structure
        """
        E_loss =((e_dft - e_nnp) / self.N) ** 2 
        # F_loss = (self.mu / (3 * sum(self.N))) \
        #     * sum([ sum([(fij_dft - fij_nnp) ** 2 for fij_dft, fij_nnp in zip(fi_dft, fi_nnp)]) for fi_dft, fi_nnp in zip(f_dft, f_nnp)])
        self.log.write(f'iter: {iter}, E = {e_nnp}, RMSE = {E_loss}\n') 
        print(f'iter: {iter}, E = {e_nnp}, RMSE = {E_loss}, params = {self.eta}, {self.rs}, {self.k}, {self.lambda_}, {self.xi}')
        return E_loss

    def preprocess_g1(self, cartesians):
        """Computes G1 values on set of atoms.

        Args:
            cartesians: coordinates of atoms
        """
        self.g1 = []
        for atom in cartesians:
            self.g1.append(calculate_sf(atom, cartesians, G_TYPE(1), r_cutoff=self.r_cutoff))

    def calculate_g(self, cartesians):
        """Calculates g and dg(by atoms and by parameters) values for each atom in structure.

        Args:
            cartesians: coordinates of atoms
        """
        # descriptors array
        g = []
        # derivatives by radius array
        derivatives_g_r = []
        # derivatives by parameters array
        # 0 - dg2_eta,  1 - dg2_rs,      2 - dg3_k, 
        # 3 - dg_4_eta, 4 - dg_4_lambda, 5 - dg_4_xi
        # 6 - dg_5_eta, 7 - dg_5_lambda, 8 - dg_5_xi 
        derivatives_g_params = {"dg2_eta": 0, "dg2_rs": 0,    "dg3_k": 0,
                                "dg4_eta": 0, "dg4_lambda": 0, "dg4_xi": 0,
                                "dg5_eta": 0, "dg5_lambda": 0, "dg5_xi": 0}
        
        # loop by atom
        for i, atom in enumerate(cartesians):
            # atom descriptors and its derivatives 
            g_atom = [self.g1[i].g]
            dg_atom = self.g1[i].dg
            # calculate descriptors for atom with parameters
            for g_type in [2, 3, 4, 5]:
                gi = calculate_sf(atom, cartesians, G_TYPE(g_type),
                                r_cutoff=self.r_cutoff, eta=self.eta, rs=self.rs, k=self.k, lambda_=self.lambda_, xi=self.xi)
                g_atom.append(gi.g)
                for i in range(3):
                    dg_atom[i] += gi.dg[i]

            g.append(g_atom)
            derivatives_g_r.append(dg_atom)
        
        g_min = [min([g_item[i] for g_item in g]) for i in range(self.descriptors_amount)]
        g_max = [max([g_item[i] for g_item in g]) for i in range(self.descriptors_amount)]

        g = [[(g_item[i] - g_min[i]) / (g_min[i] - g_max[i]) for i in range(self.descriptors_amount)] for g_item in g]

        dg_min = [min([dg_item[i] for dg_item in derivatives_g_r]) for i in range(3)]
        dg_max = [max([dg_item[i] for dg_item in derivatives_g_r]) for i in range(3)]

        derivatives_g_r = [[(dg_item[i] - dg_min[i]) / (dg_min[i] - dg_max[i]) for i in range(3)] for dg_item in derivatives_g_r]

        for i, atom in enumerate(cartesians):
            for g_type in [2, 3, 4, 5]:
                output_dg_params = calculate_dg_by_params(g[i][g_type - 1], atom, cartesians, G_TYPE(g_type),
                                                          self.r_cutoff, self.eta, self.rs, self.k, self.lambda_, self.xi)
                match g_type:
                    case 2:
                        derivatives_g_params['dg2_eta'] += output_dg_params[0]
                        derivatives_g_params['dg2_rs'] += output_dg_params[1]
                    case 3: 
                        derivatives_g_params['dg3_k'] += output_dg_params
                    case 4: 
                        derivatives_g_params['dg4_eta'] += output_dg_params[0]
                        derivatives_g_params['dg4_lambda'] += output_dg_params[1]
                        derivatives_g_params['dg4_xi'] += output_dg_params[2]
                    case 5: 
                        derivatives_g_params['dg5_eta'] += output_dg_params[0]
                        derivatives_g_params['dg5_lambda'] += output_dg_params[1]
                        derivatives_g_params['dg5_xi'] += output_dg_params[2]  

        return g, derivatives_g_r, derivatives_g_params
        
    def calculate_forces(cartesians):
        h = 0.01
        pass
    
    def fit(self, cartesians, e_train, f_train):
        """Trainig method of network.

        Args:
            cartesians: coordinates of atoms
            e_train: dft energy
            f_train: dft forces
        """
        self.prev_g, self.curr_g = None, None
        self.preprocess_g1(cartesians)
        X = np.array(cartesians).ravel().T

        for iter in range(self.n_iter):

            hidden_inputs = np.dot(self.wih, X)
            hidden_outputs = self.activation_function(hidden_inputs)
        
            if not isinstance(self.hnodes, int):
                hist = deque()
                for i in range(len(self.whh)):
                    hist.append(hidden_outputs)
                    hidden_inputs = np.dot(self.whh[i], hidden_outputs)
                    hidden_outputs = self.activation_function(hidden_inputs)

            final_inputs = np.dot(self.who, hidden_outputs)
            
            final_outputs = self.activation_function(final_inputs)

            self.eta, self.rs, self.k, self.lambda_, self.xi = final_outputs
            self.lambda_ = -1 if self.lambda_ < 0 else 1
               
            g, _, dg_params = self.calculate_g(cartesians)
            
            # TODO: implement calculations of forces
            # forces = self.calculate_forces()
            
            g_temp = [[g_atom[i] for i in range(1, 5)] for g_atom in g]

            if self.curr_g is None:
                self.curr_g = copy.deepcopy(g_temp)
            else:
                self.prev_g = copy.deepcopy(self.curr_g)
                self.curr_g = copy.deepcopy(g_temp)

            e_nnp = sum([sum(g_atom) for g_atom in g])
            self.loss(iter + 1, e_train, e_nnp, None, None)
            # self.loss(iter, e_train, e_nnp, f_train, f_nnp)

            # grad e = g_curr - g_prev with partial derivatives by g descriptors  
            # if first iteration 
            if self.prev_g is None:
                # set gradient as values of g2 - g5
                grad_e = [sum(g_atom[j] for g_atom in self.curr_g) for j in range(self.descriptors_amount - 1)]
            else:
                # set gradient as difference values of g2 - g5
                grad_e = [sum([curr_g_atom[j] - prev_g_atom[j] for curr_g_atom, prev_g_atom in zip(self.curr_g, self.prev_g)])
                        for j in range(self.descriptors_amount - 1)]
            print(grad_e)
            print(dg_params)
            # i.e. grad_e = gradient by descriptors
            # Note: g1 = const for fixed atoms and g1 has no parameters so we don't use it
            de_eta = grad_e[0] * dg_params['dg2_eta'] + grad_e[2] * dg_params['dg4_eta'] + grad_e[3] * dg_params['dg5_eta']
            de_rs = grad_e[0] * dg_params['dg2_rs']
            de_k = grad_e[1] * dg_params['dg3_k']
            de_lambda = grad_e[2] * dg_params['dg4_lambda'] + grad_e[3] * dg_params['dg5_lambda']
            de_xi = grad_e[2] * dg_params['dg4_xi'] + grad_e[3] * dg_params['dg5_xi']
            output_errors = np.array([[de_eta, de_rs, de_k, de_lambda, de_xi]])

            # update weights between hidden and output layers
            self.who += self.learning_rate * np.dot(output_errors * final_outputs * (1.0 - final_outputs), np.transpose(hidden_outputs))
            hidden_errors = np.dot(self.who.T, output_errors)
            # some update in hidden layers
            if self.whh is not None:
                while len(hist) != 0:
                    hidden_outputs_prev = hist.pop()
                    self.whh[i] += self.learning_rate * np.dot(hidden_errors * hidden_outputs * (1.0 - hidden_outputs), np.transpose(hidden_outputs_prev))
                    hidden_errors = np.dot(self.whh.T, hidden_errors)
                    hidden_outputs = hidden_outputs_prev
            # update weights between input and hidden layers
            self.wih += self.learning_rate * np.dot(hidden_errors * hidden_outputs * (1.0 - hidden_outputs), np.transpose(X))  

    def get_params(self):
        return self.eta, self.rs, self.k, self.lambda_, self.xi