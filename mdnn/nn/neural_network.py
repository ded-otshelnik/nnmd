import torch
import mdnn_cpp
from mdnn.nn.atomic_nn import AtomicNN

class Neural_Network(torch.nn.Module):
    """Class implement high-dimentional NN for system of atoms. \n
    For each atom it defines special Atomic NN which provide machine-trained potentials.
    """
    def __init__(self, n_struct: int, n_atoms: int, r_cutoff: float, hidden_nodes: list[int], 
                       input_nodes = 5, learning_rate = 0.1, epochs = 1000, h = 1, mu = 30) -> None:
        """Initiats neural network.

        Args:
            n_struct (int): number of atoms structs (atomic systems in certain time moment)
            n_atoms (int): number of atoms
            r_cutoff (float): cutoff radius
            hidden_nodes (list[int]): configuration of AtomicNNs internal layers
            input_nodes (int, optional): configuration of AtomicNNs input layer. Defaults to 5.
            learning_rate (float, optional): Defaults to 0.5.
            epochs (int, optional): number of training epochs. Defaults to 1000.
            h (int, optional): step of coordinate-wise moving (used in forces caclulations). Defaults to 1.
            mu (int, optional): coefficient of forces importance in error. Defaults to 3.
        """
        super().__init__()

        # params related to MD configuration
        self.n_struct = n_struct
        self.n_atoms = n_atoms
        self.r_cutoff = r_cutoff
        self.h = h
        self.mu = mu

        # params related to atomic nn and its config
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.criterion = torch.nn.MSELoss()
        
        self._log = open('log.out', 'w+', encoding='utf-8')
        self._train = False
    
    def _preprocess_g(self, cartesians: torch.Tensor, eta: float, rs: float, k: float, _lambda: int, xi: float):
        """Calculates symmetric functions for each structs of atoms with specified parameters.

        Args:
            cartesians (torch.Tensor): structs of atoms (atomic system in certain time moment)
            eta (float): parameter of symmetric functions
            rs (float): parameter of symmetric functions
            k (float): parameter of symmetric functions
            _lambda (int): parameter of symmetric functions
            xi (float): parameter of symmetric functions
        """
        # arrays of g and their derivatives
        g = []
        dg = []
        # params of symmetric functions
        self.eta, self.rs, self.k, self._lambda, self.xi = eta, rs, k, _lambda, xi

        # calculate symmetric functions values for each struct of atoms and its derivatives
        for struct in cartesians:
            dg_struct = torch.zeros(self.n_atoms, 5, 3, dtype=torch.double)
            g_struct = mdnn_cpp.calculate_sf(struct, self.r_cutoff, eta, rs, k, _lambda, xi, dg_struct)
            g.append(g_struct)
            dg.append(dg_struct)
        
        # g values - inputs of Atomic NNs
        # so we need to store gradient for backpropagation
        
        g = torch.stack(g).to(torch.double)
        g.requires_grad = True
        dg = torch.stack(dg).to(torch.double)
        return g, dg

    def compile(self, cartesians, eta, rs, k, _lambda, xi, load_models = False, path: str = None):
        """Configurates parameters related to calculations and AtomicNNs

        Args:
            cartesians: structs of atoms (atomic system in certain time moment)
            eta (float): parameter of symmetric functions
            rs (float): parameter of symmetric functions
            k (float): parameter of symmetric functions
            _lambda (int): parameter of symmetric functions
            xi (float): parameter of symmetric functions
            load_models (bool, optional): load pre-trained models or not. Defaults to False.
            path (str, optional): path to pre-trained models. Defaults to None.
        """
        # pre-define g values
        self.cartesians = torch.tensor(cartesians, dtype=torch.double)
        self.g, self.dg = self._preprocess_g(self.cartesians, eta, rs, k, _lambda, xi)

        # sets of atomic nn and their optimizers
        self.atomic_nn_set = []
        self.nn_optims = []
        # for each atom make Atomic NN and its optimizer
        for i in range(self.n_atoms):
            nn = AtomicNN(self.input_nodes, self.hidden_nodes)
            # if need only use pre-trained models
            if load_models:
                # load from path
                nn.load_state_dict(torch.load(path + f"/atomic_nn_{i}.pt"))
            nn.double()
            self.atomic_nn_set.append(nn)
            optim = torch.optim.Adam(nn.parameters(), lr = self.learning_rate)
            optim.zero_grad(set_to_none=True)
            self.nn_optims.append(optim)
    
    def fit(self, e_dft, f_dft):
        """Train method of neural network.

        Args:
            e_dft: target energy
            f_dft: target forces
        """
        self._train = True
        # data preparation
        self.e_dft = torch.tensor(e_dft, dtype=torch.double)
        self.f_dft = torch.tensor(f_dft, dtype=torch.double)
        
        # run training
        for epoch in range(self.epochs):
            self._train_loop(epoch)
        self._train = False

    def _train_loop(self, epoch):
        """Train loop method.

        Args:
            epoch: current training epoch
        """
        e_nn = torch.zeros((self.n_struct, self.n_atoms), dtype=torch.double)
        f_nn = torch.zeros((self.n_struct, self.n_atoms, 3), dtype=torch.double)
        # loop by struct
        for struct_index in range(self.n_struct):
            # calculate energy by NN for each atom
            for atom in range(self.n_atoms):
                nn = self.atomic_nn_set[atom]
                e_nn[struct_index][atom] = nn(self.g[struct_index][atom])

        # calculate forces per struct
        f_nn = mdnn_cpp.calculate_forces(self.cartesians, e_nn, self.g,
                                         self.atomic_nn_set, self.r_cutoff,
                                         self.h, self.eta, self.rs,
                                         self.k, self._lambda, self.xi)

        # get loss
        loss = self.loss(e_nn, self.e_dft, f_nn, self.f_dft, epoch)

        # run backpropagation
        loss.backward()
        
        # get optimizers and do optimization step
        for i in range(self.n_atoms):
            optim = self.nn_optims[i]
            optim.step()
            optim.zero_grad(set_to_none=True)
             
    def loss(self, e_nn: torch.Tensor, e_dft: torch.Tensor, f_nn: torch.Tensor, f_dft: torch.Tensor, epoch: int = None):
        """Get loss of training by criterion.

        Args:
            epoch: current training epoch
            e_nn (torch.Tensor): calculated energy
            e_dft (torch.Tensor): target energy
            f_nn (torch.Tensor): calculated forces
            f_dft (torch.Tensor): target forces
        """
        E_loss = self.criterion(e_nn.sum(dim=1), e_dft) / (self.n_struct * self.n_atoms)
        F_loss = self.criterion(f_nn, f_dft) * self.mu / 3
        loss = E_loss + F_loss
        info = (f"iter: {epoch + 1},"  if self._train else "")+ f" RMSE E = {E_loss}, RMSE F = {F_loss}, total = {loss}"
        print(info)
        if self._train:
            print(info, file=self._log)
        return loss
    
    def predict(self, cartesians):
        """Calculates energy and forces for structs of atoms

        Args:
            cartesians: structs of atoms (atomic system in certain time moment)
        """
        cartesians_ = torch.tensor(cartesians, dtype=torch.double).unsqueeze(0)
        n_structs = len(cartesians_)
        g, _ = self._preprocess_g(cartesians_, self.eta, self.rs, self.k, self._lambda, self.xi)
        e_nn = torch.zeros((n_structs, self.n_atoms), dtype=torch.double)
        f_nn = torch.zeros((n_structs, self.n_atoms, 3), dtype=torch.double)
        with torch.no_grad():
            for struct_index in range(n_structs):
                # calculate energies by NN
                for atom in range(self.n_atoms):
                    nn = self.atomic_nn_set[atom]
                    e_nn[struct_index][atom] = nn(g[struct_index][atom])
                # calculate forces per struct
            f_nn = mdnn_cpp.calculate_forces(cartesians_, e_nn, g,
                                             self.atomic_nn_set, self.r_cutoff,
                                             self.h, self.eta, self.rs,
                                             self.k, self._lambda, self.xi)
        if n_structs == 1:
            f_nn = f_nn.squeeze()
        return e_nn, f_nn
            

    def save_model(self):
        """Saves trained models to files.
        It makes possible to load pre-trained nets for calculations
        """
        for i, nn in enumerate(self.atomic_nn_set):
            assert isinstance(nn, torch.nn.Module)
            torch.save(nn.state_dict(), f'models/atomic_nn_{i}.pt')