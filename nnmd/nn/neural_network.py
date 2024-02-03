# must be imported first
# for compatibility with custom extensions
import torch
torch.manual_seed(0)

from torch.func import stack_module_state
from torch.func import functional_call
from torch.utils.data import DataLoader


# C++/CUDA extention
import nnmd_cpp

from . import AtomicNN

from ..util._logger import Logger
from ._dataset import TrainAtomicDataset

import time
import copy

class Neural_Network(torch.nn.Module):
    # amount of nn instances
    _count: int = 0

    """Class implement high-dimentional NN for system of atoms. \n
    For each atom it defines special Atomic NN which provide machine-trained potentials.
    """
    def __init__(self, hidden_nodes: list[int], use_cuda: bool,
                       input_nodes: int = 5, learning_rate: float = 0.2,
                       epochs: int = 100, h: float = 1, mu: float = 30) -> None:
        """Initializes a neural network instance.

        Args:
            hidden_nodes (list[int]): configuration of AtomicNNs internal layers
            use_cuda (bool): enable/disable CUDA support
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

        # set flag of cuda usage 
        self.use_cuda = use_cuda
        # if we can use cuda
        if self.use_cuda: 
            # set cuda c++ module as computational
            self._nnmd = nnmd_cpp.cuda
            # set GPU as device
            self.device = torch.device('cuda')
        else:
            # set cpu c++ module as computational
            self._nnmd = nnmd_cpp.cpu
            # set CPU as device
            self.device = torch.device('cpu')

        # loss function
        self.criterion = torch.nn.MSELoss().to(device=self.device)
        
        # output files
        self.net_log = Logger.get_logger("net train info", "net.log")
        self.time_log = Logger.get_logger("net time info", "time.log")
    
    def __del__(self):
        Neural_Network._count -= 1

    def _calculate_g(self, cartesians: torch.Tensor, r_cutoff: float, eta: float, rs: float, k: float, _lambda: int, xi: float):
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
        self.net_log.info(info)


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
        self.g = self._calculate_g(self.cartesians, r_cutoff, eta, rs, k, _lambda, xi)

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
    
    def fit(self, e_dft: list, f_dft: list, batch_size: int):
        """Train method of neural network.

        Args:
            e_dft: target energy
            f_dft: target forces
            batch_size: size of batch
        """

        # data preparation
        self.e_dft = torch.tensor(e_dft, device=self.device, dtype=torch.float32)
        self.f_dft = torch.tensor(f_dft, device=self.device, dtype=torch.float32)
        
        dataset = TrainAtomicDataset(self.cartesians, self.g, self.e_dft, self.f_dft)
        train_loader = DataLoader(dataset, batch_size = batch_size)
        # save package/environment info to log file
        self._describe_env()

        # run training
        for epoch in range(self.epochs):
            if self.use_cuda:
                torch.cuda.synchronize()
            start = time.time()
            self._train_loop(epoch, train_loader, batch_size)
            if self.use_cuda:
                torch.cuda.synchronize()
            self.time_log.info(f"epoch {epoch} elapsed: {(time.time() - start):.3f} s")
            print(f"epoch {epoch}: done")

    def _train_loop(self, epoch: int, train_loader: DataLoader, batch_size: int):
        """Train loop method. 

        Args:
            epoch (int): current training epoch
            train_loader (DataLoader): loads data into the pipeline for training
            batch_size (int): size of batch
        """
        # make models ensemble for passing whole inputs to whole models set
        params, buffers = stack_module_state(self.atomic_nn_set)

        # make an "anchor"
        base_model = copy.deepcopy(self.atomic_nn_set[0])
        base_model = base_model.to('meta')

        def vectorised_model(params, buffers, x):
            """Call whole models set"""
            return functional_call(base_model, (params, buffers), (x,))

        loss = torch.zeros(1, device = self.device)
        for batch, data in enumerate(train_loader):
            cartesians, g, e_dft, f_dft = data
            if self.use_cuda:
                torch.cuda.synchronize()
            start = time.time()

            # make vmap to get energies (permutes are used for dimensions aligning)
            e_nn = torch.vmap(vectorised_model)(params, buffers, g.permute(1, 0, 2)) \
                                    .permute(1, 0, 2) \
                                    .squeeze() \
                                    .contiguous() 
            if self.use_cuda:
                torch.cuda.synchronize()
            end = time.time()
            self.time_log.info(f"{epoch}, {batch}, energies: {(end - start):.3f} s")
            if self.use_cuda:
                torch.cuda.synchronize()
            start = time.time()     
            # calculate forces per struct
            f_nn = self._nnmd.calculate_forces(cartesians, e_nn, g,
                                        self.atomic_nn_set, self.r_cutoff,
                                        self.h, self.eta, self.rs,
                                        self.k, self._lambda, self.xi)
            if self.use_cuda:
                torch.cuda.synchronize()
            end = time.time()
            self.time_log.info(f"{epoch}, {batch}, forces: {(end - start):.3f} s")
            # get loss
            batch_loss = self.loss(e_nn, e_dft, f_nn, f_dft, batch_size)
            if self.use_cuda:
                torch.cuda.synchronize()
            start = time.time()
            # run backpropagation
            batch_loss.backward()
            if self.use_cuda:
                torch.cuda.synchronize()
            end = time.time()
            self.time_log.info(f"{epoch}, {batch}, backpropagation: {(end - start):.3f} s")
            loss += batch_loss
        loss /= len(train_loader)
        if self.use_cuda:
            torch.cuda.synchronize()
        start = time.time()
        # get optimizers and do optimization step
        for i in range(self.n_atoms):
            optim = self.nn_optims[i]
            optim.step()
            optim.zero_grad(set_to_none=True)
        if self.use_cuda:
            torch.cuda.synchronize()
        end = time.time()
        self.time_log.info(f"Optimizers: {(end - start):.3f} s")

        # make info message and write to log file if train mode is set
        self.net_log.info(f"iter: {epoch + 1}: RMSE total = {loss.squeeze().cpu().detach().numpy():.3f}")
             
    def loss(self, e_nn: torch.Tensor, e_dft: torch.Tensor,
                   f_nn: torch.Tensor, f_dft: torch.Tensor, batch_size: int = None) -> torch.Tensor:
        """Get loss of training by criterion.

        Args:
            epoch: current training epoch
            e_nn (torch.Tensor): calculated energy
            e_dft (torch.Tensor): target energy
            f_nn (torch.Tensor): calculated forces
            f_dft (torch.Tensor): target forces
        """
        E_loss = self.criterion(e_nn.sum(dim = 1), e_dft)
        F_loss = self.criterion(f_nn, f_dft) * self.mu
        loss = E_loss + F_loss
        return loss
    
    def predict(self, cartesians: list):
        """Calculates energy and forces for structs of atoms

        Args:
            cartesians: structs of atoms (atomic system in certain time moment)
        """
        # data preparing: convert to tensors, calculate g values, etc.
        cartesians_ = torch.tensor(cartesians, device=self.device, dtype=torch.float32)
        n_structs = len(cartesians_)
        e_nn = torch.empty((n_structs, self.n_atoms), device=self.device, dtype=torch.float32)
        g = self._calculate_g(cartesians_,self.r_cutoff, self.eta, self.rs, self.k, self._lambda, self.xi)

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
            

    def save_model(self, path: str):
        """Saves trained models to files.
        It makes possible to load pre-trained nets for calculations
        """
        for i, nn in enumerate(self.atomic_nn_set):
            torch.save(nn.state_dict(), f'{path}/atomic_nn_{i}.pt')