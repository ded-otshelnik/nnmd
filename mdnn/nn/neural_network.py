# must be imported first
# for compatibility with custom extensions
import torch

# C++ extention
import mdnn_cpp
# C++/CUDA extention
import mdnn_cuda

# Atomic neural network
from mdnn.nn.atomic_nn import AtomicNN

import time

cuda = True
mdnn, device = (mdnn_cuda, torch.device('cuda')) if torch.cuda.is_available() and cuda else (mdnn_cpp, torch.device('cpu'))

class Neural_Network(torch.nn.Module):
    """Class implement high-dimentional NN for system of atoms. \n
    For each atom it defines special Atomic NN which provide machine-trained potentials.
    """
    def __init__(self, n_struct: int, n_atoms: int, hidden_nodes: list[int],
                       input_nodes: int = 5, learning_rate: float = 0.1,
                       epochs: int = 2, h: float = 1, mu: float = 30) -> None:
        """Initializes a neural network instance.

        Args:
            n_struct (int): number of atoms structs (atomic systems in certain time moment)
            n_atoms (int): number of atoms
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
        self.h = h
        self.mu = mu

        # params related to atomic nn and its config
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.criterion = torch.nn.MSELoss().to(device=device)
        
        self._log = open('log.out', 'w+', encoding='utf-8')
        self._train = False
    
    def _preprocess_g(self, cartesians: torch.Tensor, r_cutoff: float, eta: float, rs: float, k: float, _lambda: int, xi: float):
        """Calculates symmetric functions for each structs of atoms with specified parameters.

        Args:
            cartesians (torch.Tensor): structs of atoms (atomic systems in certain time moment)
            r_cutoff (float): cutoff radius
            eta (float): parameter of symmetric functions
            rs (float): parameter of symmetric functions
            k (float): parameter of symmetric functions
            _lambda (int): parameter of symmetric functions
            xi (float): parameter of symmetric functions
            
        """
        # arrays of g and their derivatives
        g = []
        # params of symmetric functions
        self.r_cutoff = r_cutoff
        self.eta, self.rs, self.k, self._lambda, self.xi = eta, rs, k, _lambda, xi
        
        # calculate symmetric functions values for each struct of atoms and its derivatives
        for struct in cartesians:
            g_struct = mdnn.calculate_sf(struct, self.r_cutoff, eta, rs, k, _lambda, xi)
            g.append(g_struct)
        
        # g values - inputs of Atomic NNs
        # so we need to store gradient for backpropagation
        g = torch.stack(g).to(device=device, dtype=torch.float32)
        g.requires_grad = True
        return g

    def compile(self, cartesians: list, r_cutoff: float, eta: float, rs: float, k: float, _lambda: float,
                              xi: float, load_models: bool = False, path: str = None):
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
        self.cartesians = torch.tensor(cartesians, device=device, dtype=torch.float32)
        
        # pre-define g values
        self.g = self._preprocess_g(self.cartesians, r_cutoff, eta, rs, k, _lambda, xi)

        # sets of atomic nn and their optimizers
        self.atomic_nn_set = []
        self.nn_optims = []

        # for each atom make Atomic NN and its optimizer
        for i in range(self.n_atoms):
            # create Atomic NN instance for i-th atom
            nn = AtomicNN(self.input_nodes, self.hidden_nodes)
            # if needs to use pre-trained models only
            if load_models:
                # load from path
                nn.load_state_dict(torch.load(path + f"/atomic_nn_{i}.pt"))         
            nn = nn.to(device=device)
            self.atomic_nn_set.append(nn)

            # create Atomic NN optimizer instance for i-th atom
            optim = torch.optim.Adam(nn.parameters(), lr = self.learning_rate)
            self.nn_optims.append(optim)
    
    def fit(self, e_dft: list, f_dft: list):
        """Train method of neural network.

        Args:
            e_dft: target energy
            f_dft: target forces
        """
        # data preparation
        self.e_dft = torch.tensor(e_dft, device=device, dtype=torch.float32)
        self.f_dft = torch.tensor(f_dft, device=device, dtype=torch.float32)

        # run training
        for epoch in range(self.epochs):
            start = time.time()
            self._train_loop(epoch)
            end = time.time()
            print("CUDA train:" if cuda else "CPU train:", (end - start), end="\n")

    def _train_loop(self, epoch: int):
        """Train loop method. 

        Args:
            epoch: current training epoch
        """
        e_nn = torch.empty((self.n_struct, self.n_atoms), device=device, dtype=torch.float32)
        
        start = time.time()

        # loop by struct
        for struct_index in range(self.n_struct):
            # calculate energy by NN for each atom
            for atom in range(self.n_atoms):
                nn = self.atomic_nn_set[atom]
                e_nn[struct_index][atom] = nn(self.g[struct_index][atom])

        end = time.time()
        print("Energies:", (end - start))

        start = time.time()
                
        # calculate forces per struct
        f_nn = mdnn.calculate_forces(self.cartesians, e_nn, self.g,
                                     self.atomic_nn_set, self.r_cutoff,
                                     self.h, self.eta, self.rs,
                                     self.k, self._lambda, self.xi)
        
        end = time.time()
        print("Forces:", (end - start))

        start = time.time()

        # get loss
        loss = self.loss(e_nn, self.e_dft, f_nn, self.f_dft, epoch)

        end = time.time()
        print("Get loss:", (end - start))

        start = time.time()

        # run backpropagation
        loss.backward()
        end = time.time()
        print("Backpropagation:", (end - start))

        start = time.time()
        # get optimizers and do optimization step
        for i in range(self.n_atoms):
            optim = self.nn_optims[i]
            optim.step()
            optim.zero_grad(set_to_none=True)
        end = time.time()
        print("Optimizers:", (end - start))
             
    def loss(self, e_nn: torch.Tensor, e_dft: torch.Tensor,
                   f_nn: torch.Tensor, f_dft: torch.Tensor, epoch: int = None) -> torch.Tensor:
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
        info = f"iter: {epoch + 1}: RMSE E = {E_loss}, RMSE F = {F_loss}, total = {loss}"
        print(info)
        print(info, file=self._log)
        return loss
    
    def predict(self, cartesians: list):
        """Calculates energy and forces for structs of atoms

        Args:
            cartesians: structs of atoms (atomic system in certain time moment)
        """
        # data preparing
        cartesians_ = torch.tensor(cartesians, device=device, dtype=torch.float32)
        n_structs = len(cartesians_)
        g = self._preprocess_g(cartesians_,self.r_cutoff, self.eta, self.rs, self.k, self._lambda, self.xi)
        e_nn = torch.empty((n_structs, self.n_atoms), device=device, dtype=torch.float32)

        # disable gradient computation
        with torch.no_grad():
            # calculate energies using NN
            for struct_index in range(n_structs):
                for atom in range(self.n_atoms):
                    nn = self.atomic_nn_set[atom]
                    e_nn[struct_index][atom] = nn(g[struct_index][atom])

            # calculate forces per struct
            f_nn = mdnn.calculate_forces(cartesians_, e_nn, g,
                                            self.atomic_nn_set, self.r_cutoff,
                                            self.h, self.eta, self.rs,
                                            self.k, self._lambda, self.xi)
        
        return e_nn, f_nn
            

    def save_model(self):
        """Saves trained models to files.
        It makes possible to load pre-trained nets for calculations
        """
        for i, nn in enumerate(self.atomic_nn_set):
            torch.save(nn.state_dict(), f'models/atomic_nn_{i}.pt')