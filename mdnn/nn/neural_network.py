from copy import deepcopy
import torch
from torch import nn

from mdnn.nn.atomic_nn import AtomicNN

from mdnn.symm_func import symm_func

class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, target, predictions):
        return torch.sqrt(self.mse(target, predictions))

class Neural_Network(nn.Module):

    def __init__(self, n_struct, n_atoms, r_cutoff, hidden_nodes, 
                       input_nodes = 5, learning_rate = 0.5, epochs = 1000, h = 0.01, mu = 3) -> None:
        super().__init__()

        # params related to MD configuration
        self.n_struct = n_struct
        self.n_atoms = n_atoms
        self.r_cutoff = r_cutoff
        self.h = h
        self.mu = mu

        # params related to atomic nn
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.criterion = nn.MSELoss()

    def preprocess_g(self, cartesians, eta, rs, k, _lambda, xi):

        self.g = []
        self.dg = []
        
        self.eta, self.rs, self.k, self._lambda, self.xi = eta, rs, k, _lambda, xi

        for struct in cartesians:
            g_struct, dg_struct = symm_func.calculate_g(struct, self.r_cutoff, eta, rs, k, _lambda, xi)
            self.g.append(g_struct)
            self.dg.append(dg_struct)

        self.g = torch.tensor(self.g, requires_grad=True)
        self.dg = torch.tensor(self.dg)
        self.cartesians = torch.as_tensor(cartesians)

    def compile(self, cartesians, eta, rs, k, _lambda, xi):
        self.preprocess_g(cartesians, eta, rs, k, _lambda, xi)

        self.atomic_nn_set = []
        self.nn_optims = []
        for _ in range(self.n_atoms):
            #TODO: maybe convert model to C++ 
            nn = AtomicNN(self.r_cutoff, self.hidden_nodes, self.input_nodes,
                                               self.learning_rate)
            self.atomic_nn_set.append(nn)
            self.nn_optims.append(torch.optim.Adam(nn.parameters(), lr = self.learning_rate))
    
    def fit(self, e_dft, f_dft):
        self.e_dft = torch.tensor(e_dft)
        self.f_dft = torch.tensor(f_dft)
        
        for epoch in range(self.epochs):
            self._train_loop(epoch)

    def _train_loop(self, epoch):
        e_nn = torch.zeros((self.n_struct, self.n_atoms))
        f_nn = torch.zeros((self.n_struct, self.n_atoms, 3))
        
        for struct_index in range(self.n_struct):
            for i in range(self.n_atoms):
                e_nn[struct_index, i] = self.atomic_nn_set[i](self.g[struct_index][i])

            # f_nn[struct_index] = self.calculate_forces(e_nn[struct_index], self.atomic_nn_set,
            #                                            self.cartesians[struct_index])

        loss = self.loss(epoch, e_nn, self.e_dft, f_nn, self.f_dft)
        loss.backward()
        for i in range(self.n_atoms):
            self.nn_optims[i].step()
            self.nn_optims[i].zero_grad()
             
    def loss(self, epoch, e_nn: torch.Tensor, e_dft: torch.Tensor, f_nn: torch.Tensor, f_dft: torch.Tensor):
        E_loss = self.criterion(e_nn.sum(dim=1), e_dft) / (self.n_struct * self.n_atoms)
        # F_loss = self.criterion(f_nn, f_dft) * self.mu / 3
        print(f"iter: {epoch}, RMSE E = {E_loss}")
        # print(f"iter: {epoch}, RMSE E = {E_loss}, RMSE F = {F_loss}")
        # loss = E_loss + F_loss
        loss = E_loss
        return loss

    def calculate_forces(self, e_nn, atomic_nn, cartesians):
        #TODO: to implement forces calculation (mb C++ ?)
        cartesians_copy = torch.clone(cartesians)
        dim = len(cartesians[0])

        force = torch.zeros((len(cartesians), dim))

        for atom in range(len(cartesians)):
            for i in range(dim):
                cartesians_copy[atom][i] += self.h
                g_new, dg_new = symm_func.calculate_g(cartesians, self.r_cutoff,
                                                                  self.eta, self.rs, self.k,
                                                                  self._lambda, self.xi)
                g_new, dg_new = torch.as_tensor(g_new), torch.as_tensor(dg_new)

                e_new = atomic_nn[atom](g_new[atom])

                force[atom][i] = torch.sub(force[atom][i], ((e_new - e_nn[atom]) * dg_new[atom][i]))
                cartesians_copy[atom][i] -= self.h
                
        return force