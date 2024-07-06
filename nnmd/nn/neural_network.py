#TODO:
# 1. total refactoring & clean-up
# 3. parameters of all methods must be checked

# must be imported before C++ extention
# for compatibility
import torch
torch.manual_seed(0)
from torch.utils.data import DataLoader

# C++/CUDA extention
import nnmd_cpp

from . import AtomicNN
from .dataset import AtomicDataset
from ..util._logger import Logger

import time

class RMSELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        
    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))

class Neural_Network(torch.nn.Module):
    """Class implement high-dimentional NN for system of atoms. \n
    For each atom it defines special Atomic NN which provide machine-trained potentials.
    """
    def __init__(self) -> None:
        super().__init__()
    
    def _describe_training(self, dataset, batch_size, epochs):
        from importlib.metadata import version

        module_name = "nnmd"

        info = f"""NNMD v{version(module_name)} 

Neural network parameters:
    device type: {self.device.type},
    training epochs: {epochs},
    optimizer: Adam
    learning rate: {self.learning_rate},

Atomic NNs parameters:
    input size (number of descriptors): {self.input_nodes},
    hidden layers configuration: {self.hidden_nodes}
    batch size: {batch_size}
        
Symmetric functions parameters:
    cutoff radius = {dataset.symm_func_params['r_cutoff']},
    eta = {dataset.symm_func_params['eta']},
    rs = {dataset.symm_func_params['rs']},
    k = {dataset.symm_func_params['k']},
    lambda = {dataset.symm_func_params['lambda']},
    xi = {dataset.symm_func_params['xi']},
"""
        self.net_log.info(info)

    def config(self, hidden_nodes: list[int], use_cuda: bool, n_atoms: int,
                     dtype = torch.float32, input_nodes: int = 5, learning_rate: float = 0.2,
                     mu: float = 3, load_models: bool = False, **kwargs):
        """Configurates parameters related to calculations and AtomicNNs

        Args:
            hidden_nodes (list[int]): configuration of AtomicNNs internal layers
            use_cuda (bool): enable/disable CUDA support
            n_atoms (int): number of atoms
            dtype: type of data. Defaults to torch.float32
            input_nodes (int, optional): configuration of AtomicNNs input layer. Defaults to 5.
            learning_rate (float, optional): Defaults to 0.2.
            mu (int, optional): coefficient of forces importance in error. Defaults to 3.
            load_models (bool, optional): load pre-trained models or not. Defaults to False.
            path (str, optional): path to pre-trained models. Defaults to None.
        """
        
        # params related to atomic nn and its config
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes

        self.learning_rate = learning_rate
        self.mu = mu

        self.use_cuda = use_cuda
        # if CUDA is needed
        if self.use_cuda: 
            # set GPU c++ module as computational
            self._nnmd = nnmd_cpp.cuda
            # set GPU as device
            self.device = torch.device('cuda')
        # otherwise
        else:
            # set CPU c++ module as computational
            self._nnmd = nnmd_cpp.cpu
            # set CPU as device
            self.device = torch.device('cpu')

        self.dtype = dtype

        # loss function
        self.criterion = RMSELoss().to(device = self.device)

        # amount of subnets = amount of atoms 
        self.n_atoms = n_atoms

        # sets of atomic nn, their optimizers and schedulers of learning rate
        self.atomic_nn_set = []
        self.nn_optims = []

        # for each atom make Atomic NN and its optimizer
        for i in range(self.n_atoms):
            # create Atomic NN instance for i-th atom
            nn = AtomicNN(input_nodes = self.input_nodes,
                          hidden_nodes = self.hidden_nodes)
            # if using pre-trained models is needed only
            if load_models:
                # load params to current AtomicNN instance
                nn.load_state_dict(torch.load(kwargs['path'] + f"/atomic_nn_{i}.pt"))         
            nn = nn.to(device = self.device)
            self.atomic_nn_set.append(nn)

            # Atomic NN optimizer
            optim = torch.optim.Adam(nn.parameters(), lr = self.learning_rate)
            self.nn_optims.append(optim)
    
    def fit(self, dataset: AtomicDataset, batch_size: int, epochs: int):
        """Train method of neural network.

        Args:
            dataset (AtomicDataset): input dataset with data about atoms
            batch_size (int): size of batch
            epochs (int): amount of training epochs
        """
        num_batches = len(dataset.cartesians) // batch_size if len(dataset.cartesians) > batch_size else 1
        train_loader = DataLoader(dataset, batch_size = batch_size)
        symm_funcs_params = dataset.symm_func_params

        # output log files
        self.net_log = Logger.get_logger("net train info", "net.log")
        self.time_log = Logger.get_logger("net time info", "time.log")

        # save package/environment/nn info to log file
        self._describe_training(dataset, batch_size, epochs)

        # run training
        for epoch in range(epochs):
            start = time.time()
            print(f"epoch {epoch + 1}:" , end = ' ')

            self._train_loop(epoch, train_loader, num_batches, symm_funcs_params, dataset.h)

            self.time_log.info(f"epoch {epoch + 1} elapsed: {(time.time() - start):.3f} s")
            print("done")

    def _train_loop(self, epoch: int, train_loader: DataLoader, num_batches: int,
                    symm_func_params: dict[str, float], h: float):
        """Train loop method. 

        Args:
            epoch (int): current training epoch
            train_loader (DataLoader): loads data into the pipeline for training
            batch_size (int): size of batch
            symm_func_params (dict[str, float]): parameters of symmetric functions
            h (float): step of coordinate-wise moving (used in forces caclulations).
        """
        # epoch losses: energies, forces and total
        e_loss = torch.zeros(1, device = self.device).squeeze()
        f_loss = torch.zeros(1, device = self.device).squeeze()
        loss = torch.zeros(1, device = self.device).squeeze()

        for batch, data in enumerate(train_loader):
            # input data in batch:
            # positions (necessary in forces calculation),
            # g values (nn inputs),
            # energies and forces (nn targets)
            cartesians, g, e_dft, f_dft, = data
            e_nn = torch.empty((len(cartesians), self.n_atoms),
                                device = self.device, dtype = self.dtype)
            
            start = time.time()
            # calculate energies using NN
            for struct_index in range(len(cartesians)):
                for atom in range(self.n_atoms):
                    nn = self.atomic_nn_set[atom]
                    e_nn[struct_index][atom] = nn(g[struct_index][atom])

            # calculate forces per batch
            f_nn = self._nnmd.calculate_forces(cartesians, e_nn, g, self.atomic_nn_set,
                                               symm_func_params['r_cutoff'],
                                               symm_func_params['eta'],
                                               symm_func_params['rs'],
                                               symm_func_params['k'],
                                               symm_func_params['lambda'],
                                               symm_func_params['xi'], h)
            end = time.time()
            self.time_log.info(f"Epoch {epoch}, batch {batch}, forward: {(end - start):.5f} s")

            batch_loss, batch_e_loss, batch_f_loss = self.loss(e_nn, e_dft, f_nn, f_dft)

            start = time.time()
            batch_loss.backward()
            end = time.time()
            self.time_log.info(f"Epoch {epoch}, batch {batch}, backpropagation: {(end - start):.3f} s")

            loss += batch_loss
            e_loss += batch_e_loss
            f_loss += batch_f_loss

        # normalize total epoch losses
        f_loss /= num_batches
        e_loss /= num_batches
        loss /= num_batches

        start = time.time()
        for i in range(self.n_atoms):
            optim = self.nn_optims[i]
            optim.step()

        # reset params grad to None
        for model in self.atomic_nn_set:
            for param in model.parameters():
                param.grad = None

        end = time.time()
        self.time_log.info(f"Optimizers: {(end - start):.3f} s")

        # convert tensors to numpy types for logging
        e_loss = e_loss.cpu().detach().numpy()
        f_loss = f_loss.cpu().detach().numpy()
        loss = loss.cpu().detach().numpy()

        # make info message about losses and write to log file
        loss_info = f"iter: {epoch + 1}: RMSE E = {e_loss:e}, RMSE F = {f_loss:e}, RMSE total = {loss:e} eV"
        self.net_log.info(loss_info)
             
    def loss(self, e_nn: torch.Tensor, e_dft: torch.Tensor,
                   f_nn: torch.Tensor, f_dft: torch.Tensor) -> torch.Tensor:
        """Gets loss of calculations.

        Args:
            e_nn (torch.Tensor): calculated energy
            e_dft (torch.Tensor): target energy
            f_nn (torch.Tensor): calculated forces
            f_dft (torch.Tensor): target forces
        """
        E_loss = self.criterion(e_nn.sum(dim = 1), e_dft)
        F_loss = self.criterion(f_nn, f_dft) * self.mu / 3
        loss = E_loss + F_loss
        return loss, E_loss, F_loss

    def predict(self, dataset: AtomicDataset):
        """Calculates energy and forces for structs of atoms

        Args:
            dataset (nnmd.nn.AtomicDataset): input dataset with data about atoms
        """
        # data preparing: convert to tensors, calculate g values, etc.
        n_structs = len(dataset.cartesians)
        e_nn = torch.empty((n_structs, self.n_atoms), device = self.device, dtype = self.dtype)

        # disable gradient computation (we don't train NN)
        with torch.no_grad():
            # calculate energies using NN
            for struct_index in range(n_structs):
                for atom in range(self.n_atoms):
                    nn = self.atomic_nn_set[atom]
                    e_nn[struct_index][atom] = nn(dataset.g[struct_index][atom])

            f_nn = self._nnmd.calculate_forces(dataset.cartesians, e_nn, dataset.g, self.atomic_nn_set,
                                               dataset.symm_func_params['r_cutoff'],
                                               dataset.symm_func_params['eta'],
                                               dataset.symm_func_params['rs'],
                                               dataset.symm_func_params['k'],
                                               dataset.symm_func_params['lambda'],
                                               dataset.symm_func_params['xi'], dataset.h)
        
        return e_nn, f_nn
            
    def save_model(self, path: str):
        """Saves trained models to files.
        It makes possible to load pre-trained nets for calculations
        """
        for i, nn in enumerate(self.atomic_nn_set):
            torch.save(nn.state_dict(), f'{path}/atomic_nn_{i}.pt')