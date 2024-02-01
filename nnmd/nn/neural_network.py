# must be imported first
# for compatibility with custom extensions
import torch

# C++/CUDA extention
import nnmd_cpp as nnmd_cpp

# Atomic neural network
from .atomic_nn import AtomicNN

class Neural_Network(torch.nn.Module):
    """Class implement high-dimentional NN for system of atoms. \n
    For each atom it defines special Atomic NN which provide machine-trained potentials.
    """
    def __init__(self, hidden_nodes: list[int],
                       use_cuda: bool = True, input_nodes: int = 5, learning_rate: float = 0.2,
                       epochs: int = 2, h: float = 1, mu: float = 30) -> None:
        """Initializes a neural network instance.

        Args:
            hidden_nodes (list[int]): configuration of AtomicNNs internal layers
            input_nodes (int, optional): configuration of AtomicNNs input layer. Defaults to 5.
            learning_rate (float, optional): Defaults to 0.5.
            epochs (int, optional): number of training epochs. Defaults to 1000.
            h (int, optional): step of coordinate-wise moving (used in forces caclulations). Defaults to 1.
            mu (int, optional): coefficient of forces importance in error. Defaults to 3.
        """
        super().__init__()

        # params related to MD configuration
        self.h = h
        self.mu = mu

        # params related to atomic nn and its config
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.learning_rate = learning_rate
        self.epochs = epochs

        # flag of cuda usage 
        self.use_cuda = use_cuda
        # if we can use cuda
        if torch.cuda.is_available() and use_cuda: 
            # set cuda c++ module as computational
            # and GPU as device
            self._nnmd = nnmd_cpp.cuda
            self.device = torch.device('cuda')
        else:
            # set pure c++ module as computational
            # and CPU as device
            self._nnmd = nnmd_cpp.cpu
            self.device = torch.device('cpu')

        self.criterion = torch.nn.MSELoss().to(device=self.device)
        
        # output file
        self.log = open('log.out', 'w+', encoding='utf-8')
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
            g_struct = self._nnmd.calculate_sf(struct, r_cutoff, eta, rs, k, _lambda, xi)
            g.append(g_struct)
        
        # g values - inputs of Atomic NNs
        # so we need to store gradient for backpropagation
        g = torch.stack(g).to(device=self.device, dtype=torch.float32)
        g.requires_grad = True
        return g
    
    def _describe_env(self):
        from importlib.metadata import version

        module_name = "nnmd"

        info = f"""NNMD v{version(module_name)} 

Neural network parameters:
    device type: {self.device.type},
    epochs total: {self.epochs},
    optimizer: Adam
    learning rate: {self.learning_rate},

Atomic NNs parameters:
    input size (number of descriptors): {self.input_nodes},
    hidden layers configuration: {self.hidden_nodes}
        
Symmetric functions parameters:
    cutoff radius rc = {self.r_cutoff},
    eta = {self.eta},
    rs = {self.rs},
    k = {self.k},
    lambda = {self._lambda},
    xi = {self.xi},
"""
        print(info, file=self.log)


    def compile(self, cartesians: list, n_structs: int, n_atoms: int, r_cutoff: float, eta: float, rs: float, k: float, _lambda: float,
                              xi: float, load_models: bool = False, path: str = None):
        """Configurates parameters related to calculations and AtomicNNs

        Args:
            cartesians: structs of atoms (atomic system in certain time moment)
            n_structs (int): number of atoms structs
            n_atoms (int): number of atoms
            eta (float): parameter of symmetric functions
            rs (float): parameter of symmetric functions
            k (float): parameter of symmetric functions
            _lambda (int): parameter of symmetric functions
            xi (float): parameter of symmetric functions
            load_models (bool, optional): load pre-trained models or not. Defaults to False.
            path (str, optional): path to pre-trained models. Defaults to None.
        """
        self.cartesians = torch.tensor(cartesians, device=self.device, dtype=torch.float32)

        # characteristics of training set
        self.n_structs = n_structs
        self.n_atoms = n_atoms

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
            nn = nn.to(device=self.device)
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
        self.e_dft = torch.tensor(e_dft, device=self.device, dtype=torch.float32)
        self.f_dft = torch.tensor(f_dft, device=self.device, dtype=torch.float32)
        
        # save info to log file
        self._describe_env()

        # run training
        self._train = True
        for epoch in range(self.epochs):
            self._train_loop(epoch)
        self._train = False

    def _train_loop(self, epoch: int):
        """Train loop method. 

        Args:
            epoch: current training epoch
        """
        e_nn = torch.empty((self.n_structs, self.n_atoms),
                            device=self.device,
                            dtype=torch.float32)

        # loop by struct
        for struct_index in range(self.n_structs):
            # calculate energy by NN for each atom
            for atom in range(self.n_atoms):
                nn = self.atomic_nn_set[atom]
                e_nn[struct_index][atom] = nn(self.g[struct_index][atom])
                
        # calculate forces per struct
        f_nn = self._nnmd.calculate_forces(self.cartesians, e_nn, self.g,
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
        E_loss = self.criterion(e_nn.sum(dim=1), e_dft) / (self.n_structs * self.n_atoms)
        F_loss = self.criterion(f_nn, f_dft) * self.mu / 3
        loss = E_loss + F_loss
        # make info message and write to log file
        if self._train:
            info = f"iter: {epoch + 1}: RMSE E = {E_loss}, RMSE F = {F_loss}, total = {loss}"
            print(info, file=self.log)
        return loss
    
    def predict(self, cartesians: list):
        """Calculates energy and forces for structs of atoms

        Args:
            cartesians: structs of atoms (atomic system in certain time moment)
        """
        # data preparing
        cartesians_ = torch.tensor(cartesians, device=self.device, dtype=torch.float32)
        n_structs = len(cartesians_)
        g = self._preprocess_g(cartesians_,self.r_cutoff, self.eta, self.rs, self.k, self._lambda, self.xi)
        e_nn = torch.empty((n_structs, self.n_atoms), device=self.device, dtype=torch.float32)

        # disable gradient computation (we don't train NN)
        with torch.no_grad():
            # calculate energies using NN
            for struct_index in range(n_structs):
                for atom in range(self.n_atoms):
                    nn = self.atomic_nn_set[atom]
                    e_nn[struct_index][atom] = nn(g[struct_index][atom])

            # calculate forces per struct
            f_nn = self._nnmd.calculate_forces(cartesians_, e_nn, g,
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