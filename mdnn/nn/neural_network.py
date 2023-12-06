import torch
import torch.nn as nn

from mdnn.symm_func import symm_func

class NeuralNetwork(nn.Module):
    def __init__(self, n_struct, n_atoms, r_cutoff, hidden_nodes, learning_rate = 0.05, epochs=1000, mu=3):
        super().__init__()

        self.r_cutoff = r_cutoff
        self.n_struct = n_struct
        self.n_atoms = n_atoms
        self.mu = mu
        self.h = 0.01

        self.epochs = epochs

        self.flatten = nn.Flatten()
        self.model = nn.Sequential()
        if isinstance(hidden_nodes, list):
            self.model.append(nn.Linear(3 * n_atoms, hidden_nodes[0]))
            for curr, prev in zip(range(1, len(hidden_nodes)), range(len(hidden_nodes) - 1)):
                self.model.append(nn.Linear(hidden_nodes[prev], hidden_nodes[curr]))
                self.model.append(nn.Sigmoid())
            self.model.append(nn.Linear(hidden_nodes[-1], 5))
        else:
            self.model.append(nn.Linear(3 * n_atoms, hidden_nodes))
            self.model.append(nn.Sigmoid())
            self.model.append(nn.Linear(hidden_nodes, 5))
        self.model.append(nn.ReLU())

        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.log = open('log.out', 'w+', encoding='utf-8')

    def loss(self, iter, cartesians, pred, e_dft, f_dft):
        """Gets loss of neural network: RMSE(Energy)^2 + (mu/3) * RMSE(Force)^2

        Args:
            mu: importance coefficient of force
            e_dft: dft energy
            e_nnp: energy that is calculated by network
            f_dft: dft energy
            f_nnp: force that is calculated by network
            N: number of atoms in structure
        """
        self.eta, self.rs,  self.k, self._lambda, self.xi = pred

        e_nn = torch.tensor(symm_func.calculate_atoms_energy(cartesians, self.r_cutoff,
                                          self.eta, self.rs,  self.k, self._lambda, self.xi))
        f_nn = torch.tensor(symm_func.calculate_force(cartesians, self.r_cutoff,
                                          self.eta, self.rs,  self.k, self._lambda, self.xi, self.h))

        E_loss = ((torch.sub(e_nn, e_dft)) / self.n_atoms).pow(2).sum()
        F_loss = (torch.sub(f_nn, f_dft).pow(2)).sum() * self.mu / (3 * self.n_atoms)
        loss = E_loss + F_loss
        self.log.write(f'iter: {iter + 1}, RMSE E = {E_loss}, RMSE F = {F_loss}\n') 
        print(f'iter: {iter + 1}, RMSE E = {E_loss}, RMSE F = {F_loss}')
        return loss

    def forward(self, x):
        logits = self.model(x)
        return logits
    
    def train_loop(self, iter, cartesians, e_dft, f_dft):
        size = len(cartesians)
        for batch, struct in enumerate(cartesians, start=1):
            input = torch.flatten(struct)

            pred = self.model(input)
            pred[3] = torch.sign(pred[3])

            loss = self.loss(iter, struct, pred, e_dft[batch], f_dft[batch])

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if batch % 5 == 0:
                loss, current = loss.item(), (batch + 1) * len(struct)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def fit(self, cartesians, e_dft, f_dft, device=None):
        input = torch.as_tensor(cartesians, device=device)
        e_dft = torch.as_tensor(e_dft, device=device)
        f_dft = torch.as_tensor(f_dft, device=device)

        for epoch in range(self.epochs):
            self.train_loop(epoch, input, e_dft, f_dft)