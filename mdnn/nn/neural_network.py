import torch
import mdnn_cpp
class Neural_Network(torch.nn.Module):
    """Class implement high-dimentional NN for system of atoms. \n
    For each atom defined special Atomic NN which provide machine-trained potentials.
    """
    def __init__(self, n_struct, n_atoms, r_cutoff, hidden_nodes, 
                       input_nodes = 5, learning_rate = 0.5, epochs = 1000, h = 1, mu = 3) -> None:
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
        self.criterion = torch.nn.MSELoss()
        
        # self.log = open('log.out', 'w+', encoding='utf-8')
        self.double()
    
    def preprocess_g(self, cartesians, eta: float, rs: float, k: float, _lambda: int, xi: float):
        # arrays of g and their derivatives
        self.g = []
        self.dg = []
        # params of symmetric functions
        self.eta, self.rs, self.k, self._lambda, self.xi = eta, rs, k, _lambda, xi

        self.cartesians = torch.tensor(cartesians, dtype=torch.double)
        # calculate symmetric functions values for each struct of atoms
        for struct in self.cartesians:
            dg_struct = torch.zeros(self.n_atoms, 5, 3, dtype=torch.double)
            g_struct = mdnn_cpp.calculate_sf(struct, self.r_cutoff, eta, rs, k, _lambda, xi, dg_struct)
            self.g.append(g_struct)
            self.dg.append(dg_struct)
        
        # g values - inputs of Atomic NNs
        # so we need to store gradient for backpropagation
        self.g = torch.stack(self.g).to(torch.float)
        self.g.requires_grad = True
        self.dg = torch.stack(self.dg).to(torch.double)

    def compile(self, cartesians, eta, rs, k, _lambda, xi):
        # pre-define g values
        self.preprocess_g(cartesians, eta, rs, k, _lambda, xi)

        self.atomic_nn_set = []
        self.nn_optims = []
        # for each atom we make Atomic NN and NN optimizer
        for _ in range(self.n_atoms):
            nn = mdnn_cpp.AtomicNN(self.input_nodes, self.hidden_nodes)
            # nn = AtomicNN(self.input_nodes, self.hidden_nodes)
            self.atomic_nn_set.append(nn)
            self.nn_optims.append(torch.optim.Adam(nn.parameters(), lr = self.learning_rate))
    
    def fit(self, e_dft, f_dft):
        self.e_dft = torch.tensor(e_dft, dtype=torch.double)
        self.f_dft = torch.tensor(f_dft, dtype=torch.double)
        
        for epoch in range(self.epochs):
            self._train_loop(epoch)

    def _train_loop(self, epoch):
        e_nn = torch.zeros((self.n_struct, self.n_atoms), dtype=torch.double)
        f_nn = torch.zeros((self.n_struct, self.n_atoms, 3), dtype=torch.double)
        for struct_index in range(self.n_struct):
            # calculate energies by NN
            for atom in range(self.n_atoms):
                nn = self.atomic_nn_set[atom]
                e_nn[struct_index][atom] = nn(self.g[struct_index][atom])
            # calculate forces per struct
            temp = mdnn_cpp.calculate_forces(self.cartesians[struct_index], e_nn[struct_index], self.atomic_nn_set,
                                                         self.r_cutoff, self.h, 
                                                         self.eta, self.rs, self.k,
                                                         self._lambda, self.xi)
            f_nn[struct_index] = temp

        # get loss
        loss = self.loss(epoch, e_nn, self.e_dft, f_nn, self.f_dft)
        # run backpropagation
        loss.backward()
        for i in range(self.n_atoms):
            optim = self.nn_optims[i]
            optim.step()
            optim.zero_grad(set_to_none=True)
             
    def loss(self, epoch, e_nn: torch.Tensor, e_dft: torch.Tensor, f_nn: torch.Tensor, f_dft: torch.Tensor):
        E_loss = self.criterion(e_nn.sum(dim=1), e_dft) / (self.n_struct * self.n_atoms)
        F_loss = self.criterion(f_nn, f_dft) * self.mu / 3
        # print(f"iter: {epoch + 1}, RMSE E = {E_loss}")
        print(f"iter: {epoch + 1}, RMSE E = {E_loss}, RMSE F = {F_loss}")
        loss = E_loss + F_loss
        # loss = E_loss
        return loss