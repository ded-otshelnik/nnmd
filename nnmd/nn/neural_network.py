#TODO:
# 1. total refactoring & clean-up
# 3. parameters of all methods must be checked

# must be imported before C++ extention
# for compatibility
import torch
from torch import nn
torch.manual_seed(0)
from torch.utils.data import DataLoader

# C++/CUDA extention
import nnmd_cpp

from . import AtomicNN
from .dataset import AtomicDataset
from ..util._logger import Logger
from ..util.calculate_g import calculate_g

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
        """Init method of Neural_Network class
        """
        super().__init__()
    
    def _describe_training(self, dataset, batch_size, epochs):
        from importlib.metadata import version

        module_name = "nnmd"

        self.net_log.info(f"NNMD v{version(module_name)}\n")
        self.net_log.info(f"---Neural network parameters---")
        self.net_log.info(f"device:                        {self.device}")
        self.net_log.info(f"training epochs:               {epochs}")
        self.net_log.info(f"batch size:                    {batch_size}")
        self.net_log.info(f"optimizer:                     {self.optim.__class__.__name__}")
        self.net_log.info(f"loss function:                 {self.criterion.__class__.__name__}")
        self.net_log.info(f"loss function coefficient:     {self.mu}")
        self.net_log.info(f"learning rate:                 {self.learning_rate}")
        self.net_log.info(f"scheduler:                     {self.sched.__class__.__name__}")
        self.net_log.info(f"L2 regularization coefficient: {self.l2_regularization}\n")
        self.net_log.info(f"----Atomic NNs parameters----")
        self.net_log.info(f"input size (number of descriptors): {self.input_nodes}")
        self.net_log.info(f"hidden layers sizes:                {self.hidden_nodes}")
        self.net_log.info(f"mu:                                 {self.mu}\n")
        self.net_log.info(f"----Symmetry functions parameters----")
        self.net_log.info(f"cutoff radius: {dataset.symm_func_params['r_cutoff']}")
        self.net_log.info(f"eta:           {dataset.symm_func_params['eta']}")
        self.net_log.info(f"rs:            {dataset.symm_func_params['rs']}")
        self.net_log.info(f"k:             {dataset.symm_func_params['k']}")
        self.net_log.info(f"lambda:        {dataset.symm_func_params['lambda']}")
        self.net_log.info(f"xi:            {dataset.symm_func_params['xi']}\n")
        self.net_log.info(f"Training sample size: {len(dataset)}")


    def config(self, hidden_nodes: list[int], use_cuda: bool, n_atoms: int,
                     dtype = torch.float32, input_nodes: int = 5, learning_rate: float = 10e-4,
                     l2_regularization = 10e-4, mu: float = 3, load_models: bool = False, **kwargs):
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
        self.l2_regularization = l2_regularization
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

        # create Atomic NN instance for i-th atom
        self.net = AtomicNN(input_nodes = self.input_nodes,
                          hidden_nodes = self.hidden_nodes)
            
        for m in self.net.model:
            # Xavier weights initialization works better on linear layers
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

        # if using pre-trained models is needed
        if load_models:
            # load params to current AtomicNN instance
            self.net.load_state_dict(torch.load(kwargs['path'] + f"/atomic_nn.pt"))         
        self.net = self.net.to(device = self.device)

        # Atomic NN optimizer
        self.optim = torch.optim.Adam(self.net.parameters(), lr = self.learning_rate, weight_decay = self.l2_regularization)
        # scheduler for Atomic NN optimizer
        self.sched = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim, factor = 0.9, patience = 15)
    
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
        self.net.train()
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
            
            start = time.time()
            # calculate energies using NN
            # and gradients by G values using Autograd Engine
            e_nn = self.net(g)
            dE = torch.autograd.grad(torch.sum(e_nn), g, create_graph = True)[0]

            f_nn = self._nnmd.calculate_forces(cartesians, g, dE,
                                               symm_func_params['r_cutoff'],
                                               symm_func_params['eta'],
                                               symm_func_params['rs'],
                                               symm_func_params['k'],
                                               symm_func_params['lambda'],
                                               symm_func_params['xi'],
                                               h)
            
            end = time.time()
            self.time_log.info(f"Epoch {epoch}, batch {batch}, forward: {(end - start):.5f} s")

            batch_loss, batch_e_loss, batch_f_loss = self.loss(e_nn.squeeze(2), e_dft, f_nn, f_dft)

            start = time.time()
            batch_loss.backward(retain_graph = True)
            end = time.time()
            self.time_log.info(f"Epoch {epoch}, batch {batch}, backpropagation: {(end - start):.3f} s")

            self.optim.step()
            self.sched.step(batch_loss)

            # reset params grad to None
            for param in self.net.parameters():
                param.grad = None

            loss += batch_loss
            e_loss += batch_e_loss
            f_loss += batch_f_loss

        # normalize total epoch losses
        f_loss /= num_batches
        e_loss /= num_batches
        loss /= num_batches

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
        F_loss = self.criterion(f_nn, f_dft) * (self.mu / 3)
        loss = E_loss + F_loss
        return loss, E_loss, F_loss

    def predict(self, cartesians: torch.Tensor, symm_func_params: dict[str, float],
                 h: float) -> tuple[torch.Tensor]:
        """Calculates energy and forces for structs of atoms

        Args:
            cartesians (torch.Tensor): positions of atoms
            symm_func_params (dict[str, float]): parameters of symmetric functions
            h (float): step of coordinate-wise moving (used in forces caclulations).
        """
        self.net.eval()
        cartesians = cartesians.unsqueeze(0).to(device = self.device)
        e_nn = torch.empty(self.n_atoms, device = self.device, dtype = self.dtype)
        # disable gradient computation (we don't train NN)
        g = calculate_g(cartesians, self.device, symm_func_params).squeeze(0)
        g.requires_grad = True
        # calculate energies using NN
        e_nn = self.net(g)
        dE = torch.autograd.grad(torch.sum(e_nn), g, retain_graph = True)[0]
        with torch.no_grad():
            #f_nn = self._nnmd.calculate_forces(cartesians, e_nn.unsqueeze(0), g.unsqueeze(0), self.atomic_nn_set,
            f_nn = self._nnmd.calculate_forces(cartesians, g.unsqueeze(0), dE,
                                               symm_func_params['r_cutoff'],
                                               symm_func_params['eta'],
                                               symm_func_params['rs'],
                                               symm_func_params['k'],
                                               symm_func_params['lambda'],
                                               symm_func_params['xi'], h).squeeze(0)
        return e_nn, f_nn
            
    def save_model(self, path: str):
        """Saves trained models to files.
        It makes possible to load pre-trained nets for calculations
        """
        torch.save(self.net.state_dict(), f'{path}/atomic_nn.pt')