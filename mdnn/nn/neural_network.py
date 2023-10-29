import numpy as np

from ..util.gdf import gdf
from ..symmetry_functions.sf import calculate_sf, G_TYPE
from ..symmetry_functions.pair_g import PairG

from .atomic_nn import AtomicNN

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
                eta_nn=0.05, n_iter=1, mu=0.6):
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
            mu (float, optional): relative importance of forces in loss function
            
        """
        # number of atoms
        self.n_atoms = n_atoms
        self.hnodes = hidden_nodes
        self.eta_nn = eta_nn
        self.n_iter = n_iter
        self.mu = mu
        
        self.atomic_nn = AtomicNN(eta_nn, n_iter, hidden_nodes)

        # weights of input g in atomic neural network
        sigma = 0.001
        self.wih = np.array(gdf(n_atoms, g_init, g_iter, sigma))

    def calculate_energy(self, cartesians, eta, rs, k, lambda_, xi, r_cutoff, e_train):
        for cartesian, energy in zip(cartesians, e_train):
            pair = PairG(0, 0)
            pair += calculate_sf(cartesian, cartesians, g_type = G_TYPE.G1,
                                r_cutoff = r_cutoff)
            pair += calculate_sf(cartesian, cartesians, g_type = G_TYPE.G2,
                                r_cutoff = r_cutoff, eta = eta, rs = rs)
            pair += calculate_sf(cartesian, cartesians, g_type = G_TYPE.G3,
                                r_cutoff = r_cutoff, k = k)
            pair += calculate_sf(cartesian, cartesians, g_type = G_TYPE.G4,
                                r_cutoff = r_cutoff, eta = eta, xi = xi, lambda_ = lambda_)
            pair += calculate_sf(cartesian, cartesians, g_type = G_TYPE.G5,
                                r_cutoff = r_cutoff, eta = eta, xi = xi, lambda_ = lambda_)
            self.atomic_nn.fit(pair, energy)
            
    def loss(self, mu, e_dft, e_nnp, f_dft, f_nnp, M, N):
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
        return (1 / M) * sum([((e_dft_i - e_nnp_i) / Ni) ** 2 for e_dft_i, e_nnp_i, Ni in zip(e_dft, e_nnp, N)]) \
            + (mu / (3 * sum(N))) * sum([ sum([(fij_dft - fij_nnp) ** 2 for fij_dft, fij_nnp in zip(fi_dft, fi_nnp)]) for fi_dft, fi_nnp in zip(f_dft, f_nnp)])
                

    def train(self, cartesians_train, E_train, F_train, eps, r_cutoff):
        # parameters in symmetric functions: eta, rs, k, lambda, xi 
        self.eta, self.rs, self.k, self.lambda_, self.xi  = 1, 1, 1, 1, 1, 1
        
        loss = 10e1
        while loss > eps:
            e_nnp = []
            e_dft = []
            f_nnp = []
            f_dft = []
            for cartesians, energy_dft in zip(cartesians_train, E_train):
                energy_nnp = self.calculate_energy(self, cartesians)
                


            loss = self.loss(self.mu, e)
                
                
        pass

    def predict():
        pass       
            
    