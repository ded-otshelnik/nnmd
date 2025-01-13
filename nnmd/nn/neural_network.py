#TODO:
# 1. total refactoring & clean-up
# 2. all methods except __init__ must be covered by tests
# 3. parameters of all methods must be checked

import math

# PyTorch module
# must be imported before C++ extention
# for compatibility
import torch
# set random seed for reproducibility
torch.manual_seed(0)
from torch import nn
from torch.utils.data import DataLoader

from . import AtomicNN
from .dataset import TrainAtomicDataset
from ..util._logger import Logger
from ..features import calculate_sf

import os
DEBUG = os.environ.get('DEBUG', True)

if DEBUG:
    import time
    from tqdm import tqdm

class RMSELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        
    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))

class HDNN(torch.nn.Module):
    """Class implement high-dimentional NN for system of atoms. \n
    For each atom it defines special Atomic NN which provide machine-trained potentials.
    """
    def __init__(self, dtype = torch.float32, use_cuda: bool = True) -> None:
        """Initialize HDNN instance.

        Args:
            dtype (torch.dtype): type of data in tensors
            use_cuda (bool): flag to use CUDA for calculations
        """
        super().__init__()
        self.dtype = dtype
        self.use_cuda = use_cuda
    
    def _describe_training(self, train_dataset_size, val_dataset_size, batch_size, epochs):
        from importlib.metadata import version

        module_name = "nnmd"

        self.net_log.info(f"NNMD v{version(module_name)}\n")
        self.net_log.info(f"---Neural network parameters---")
        self.net_log.info(f"device:                        {self.device}")
        self.net_log.info(f"training epochs:               {epochs}")
        self.net_log.info(f"batch size:                    {batch_size}")
        self.net_log.info(f"optimizer:                     {self.optim[0].__class__.__name__}")
        self.net_log.info(f"loss function:                 {self.criterion.__class__.__name__}")
        self.net_log.info(f"loss function coefficient:     {self.mu}")
        self.net_log.info(f"learning rate:                 {self.learning_rate}")
        self.net_log.info(f"scheduler:                     {self.sched[0].__class__.__name__}")
        self.net_log.info(f"L2 regularization coefficient: {self.l2_regularization}\n")
        self.net_log.info(f"----Atomic NNs parameters----")
        self.net_log.info(f"input size (number of descriptors): {self.input_size}")
        self.net_log.info(f"hidden layers sizes:                {self.hidden_sizes}")
        self.net_log.info(f"Training sample size:   {train_dataset_size}")
        self.net_log.info(f"Validation sample size: {val_dataset_size}")


    def config(self, neural_net_data: dict, input_size: int, path: str = None):
        """Configures HDNN instance.

        Args:
            neural_net_data (dict): data about high-dimentional neural network and its atomic subnets
            input_size (int): size of input data, number of descriptors
            path (str): path to pre-trained models. If not None and parameter in neural_network_data ("load_models") is True,
            models will be loaded
        """
        self.neural_net_data = neural_net_data

        # amount of atom species in dataset
        self.num_species = len(neural_net_data['atom_species'])

        # params related to atomic nn and its config
        self.input_size = input_size
        self.hidden_sizes = neural_net_data['hidden_sizes']

        # params related to training
        self.learning_rate = neural_net_data['learning_rate']
        self.l2_regularization = neural_net_data['l2_regularization']

        # a coefficient of forces importance in error
        self.mu = neural_net_data['mu']

        # target device
        self.device = torch.device('cuda') if self.use_cuda else torch.device('cpu')

        # create Atomic NNs for each atom species
        self.atomic_nets = nn.ModuleList()
        for i in range(self.num_species):
            self.atomic_nets.append(AtomicNN(input_size = self.input_size,
                                             hidden_size = self.hidden_sizes[i]))
            # initialize weights of Atomic NNs
            for m in self.atomic_nets[i].model:
                # Xavier weights initialization works better on linear layers
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    nn.init.constant_(m.bias, 0)
            self.atomic_nets[i] = self.atomic_nets[i].to(device = self.device)
            if neural_net_data['load_models']:
                path2model = "".join([path, f"atomic_nn_{neural_net_data['atom_species'][i]}.pt"])
                self.atomic_nets[i].load_state_dict(torch.load(path2model))

        # loss function
        self.criterion = RMSELoss().to(device = self.device)

        # Atomic NN optimizer
        self.optim = [torch.optim.Adam(net.parameters(),
                                        lr = self.learning_rate,
                                        weight_decay = self.l2_regularization) for net in self.atomic_nets]
    
    def loss(self, e_nn: torch.Tensor, energies: torch.Tensor,
                   f_nn: torch.Tensor, forces: torch.Tensor) -> torch.Tensor:
        """Gets loss of calculations.

        Args:
            e_nn (torch.Tensor): calculated energy
            energy (torch.Tensor): target energy
            f_nn (torch.Tensor): calculated forces
            forces (torch.Tensor): target forces
        """
        E_loss = self.criterion(e_nn, energies)
        F_loss = self.criterion(f_nn, forces) * (self.mu / 3)
        loss = E_loss + F_loss
        return loss, E_loss, F_loss

    def fit(self, train_dataset: TrainAtomicDataset,
             val_dataset: TrainAtomicDataset,
             test_dataset: TrainAtomicDataset):
        """Train method of neural network.

        Args:
            dataset (AtomicDataset): input dataset with data about atoms
            batch_size (int): size of batch
            epochs (int): amount of training epochs
        """
        # training parameters
        batch_size = self.neural_net_data['batch_size']
        epochs = self.neural_net_data['epochs']

        train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
        val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = True)
        test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True)

        # scheduler for Atomic NN optimizer
        self.sched = [torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode = 'min',
                                                                factor = 0.8,
                                                                patience = 10, 
                                                                min_lr = 1e-5) for optim in self.optim]
        
        # output log file
        self.net_log = Logger.get_logger("Net training info", "net.log")

        # save package/environment/nn info to log file
        self._describe_training(len(train_dataset), len(val_dataset), batch_size, epochs)

        # current learning rate
        curr_lr = self.optim[0].param_groups[0]['lr']
        
        # format of loss info message
        loss_info = "RMSE E = {e_loss:e}, RMSE F = {f_loss:e}, RMSE total = {loss:e} eV"

        # run training
        for epoch in range(epochs):
            e_loss, f_loss, loss = self._train_loop(epoch, train_loader)
            self.net_log.info(f"Epoch {epoch + 1}: training: " + loss_info.format(e_loss = e_loss.item(),
                                                                                    f_loss = f_loss.item(), 
                                                                                    loss = loss.item()))

            e_loss, f_loss, loss = self._validate(epoch, val_loader)
            
            self.net_log.info(f"Epoch {epoch + 1}: validation: " + loss_info.format(e_loss = e_loss.item(),
                                                                                    f_loss = f_loss.item(), 
                                                                                    loss = loss.item()))

            # if scheduler is used
            # update learning rate by rule
            for sched in self.sched:
                if isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    sched.step(loss)
                else:
                    sched.step()

            # print the learning rate at the end of the epoch
            # if it was changed
            if self.optim[0].param_groups[0]['lr'] != curr_lr:
                curr_lr = self.optim[0].param_groups[0]['lr']
                self.net_log.info(f"Learning rate changed to {curr_lr}")

        e_loss, f_loss, loss = self._test(test_loader)
        self.net_log.info(f"Testing: " + loss_info.format(e_loss = e_loss.item(),
                                                          f_loss = f_loss.item(), 
                                                          loss = loss.item()))


    def _train_loop(self, epoch: int, train_loader: DataLoader):
        """Train loop method. 

        Args:
            epoch (int): current training epoch
            train_loader (DataLoader): loads data into the pipeline for training
            batch_size (int): size of batch
            symm_func_params (dict[str, float]): parameters of symmetric functions
            h (float): step of coordinate-wise moving (used in forces caclulations).
        """
        # set model to training mode
        for net in self.atomic_nets:
            net.train()

        # epoch losses: energies, forces and total
        e_loss = torch.zeros(1, device = self.device).squeeze()
        f_loss = torch.zeros(1, device = self.device).squeeze()
        loss = torch.zeros(1, device = self.device).squeeze()

        for data in tqdm(train_loader, desc = f"Epoch {epoch + 1}, Training", total = len(train_loader)):
            # reset params grad to None
            for net in self.atomic_nets:
                for param in net.parameters():
                    param.grad = None
                
            # input data in batch:
            # g values (nn inputs),
            # dG values (for forces calculation),
            # energies and forces (nn targets)
            g, dG, energy, forces = data

            # calculate energies using NN
            # and gradients of G values using Autograd Engine
            e_nn = self.atomic_nets[0](g)
            dE = torch.autograd.grad(e_nn.sum(), g, retain_graph = True)[0]

            # calculate forces using gradients of E values
            # and G values gradients by Einstein summation rule
            f_nn = -torch.einsum('ijk,ijkl->ijl', dE, dG)

            batch_loss, batch_e_loss, batch_f_loss = self.loss(e_nn.sum(dim = 1).squeeze(1), energy, f_nn, forces)
            batch_loss.backward(retain_graph = True)

            for optim in self.optim:
                optim.step()

            loss += batch_loss
            e_loss += batch_e_loss
            f_loss += batch_f_loss

        # normalize total epoch losses
        f_loss /= len(train_loader)
        e_loss /= len(train_loader)
        loss /= len(train_loader)

        return e_loss, f_loss, loss

    def _validate(self, epoch: int, val_loader: DataLoader) -> torch.Tensor:
        """Train loop method. 

        Args:
            epoch (int): current training epoch
            train_loader (DataLoader): loads data into the pipeline for training
            
        
        """
        for net in self.atomic_nets:
            net.eval()

        # epoch losses: energies, forces and total
        e_loss = torch.zeros(1, device = self.device).squeeze()
        f_loss = torch.zeros(1, device = self.device).squeeze()
        loss = torch.zeros(1, device = self.device).squeeze()

        for data in tqdm(val_loader, desc = f"Epoch {epoch + 1}, Validation"):
            # input data in batch:
            # positions (necessary in forces calculation),
            # g values (nn inputs),
            # energies and forces (nn targets)
            g, dG, energy, forces = data

            # calculate energies using NN
            # and gradients by G values using Autograd Engine
            e_nn = self.atomic_nets[0](g)
            dE = torch.autograd.grad(torch.sum(e_nn), g, retain_graph = True)[0]

            # calculate forces using gradients of E values
            # and G values gradients by Einstein summation rule
            with torch.no_grad():
                f_nn = -torch.einsum('ijk,ijkl->ijl', dE, dG)
            
            batch_loss, batch_e_loss, batch_f_loss = self.loss(e_nn.sum(dim = 1).squeeze(1), energy, f_nn, forces)

            loss += batch_loss
            e_loss += batch_e_loss
            f_loss += batch_f_loss

        # normalize total epoch losses
        f_loss /= len(val_loader)
        e_loss /= len(val_loader)
        loss /= len(val_loader)

        return e_loss, f_loss, loss

    def _test(self, test_loader: DataLoader):
        for net in self.atomic_nets:
            net.eval()

        # epoch losses: energies, forces and total
        e_loss = torch.zeros(1, device = self.device).squeeze()
        f_loss = torch.zeros(1, device = self.device).squeeze()
        loss = torch.zeros(1, device = self.device).squeeze()

        for data in tqdm(test_loader, desc = f"Testing"):
            # input data in batch:
            # positions (necessary in forces calculation),
            # g values (nn inputs),
            # energies and forces (nn targets)
            g, dG, energy, forces = data

            # calculate energies using NN
            # and gradients by G values using Autograd Engine
            e_nn = self.atomic_nets[0](g)
            dE = torch.autograd.grad(torch.sum(e_nn), g, retain_graph = True)[0]

            # calculate forces using gradients of E values
            # and G values gradients by Einstein summation rule
            with torch.no_grad():
                f_nn = -torch.einsum('ijk,ijkl->ijl', dE, dG)
            
            batch_loss, batch_e_loss, batch_f_loss = self.loss(e_nn.sum(dim = 1).squeeze(1), energy, f_nn, forces)

            loss += batch_loss
            e_loss += batch_e_loss
            f_loss += batch_f_loss

        # normalize total epoch losses
        f_loss /= len(test_loader)
        e_loss /= len(test_loader)
        loss /= len(test_loader)    

        return e_loss, f_loss, loss

    def predict(self, cartesians: torch.Tensor, symm_func_params: dict[str, float]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculates energy and forces for structs of atoms

        Args:
            cartesians (torch.Tensor): positions of atoms
            symm_func_params (dict[str, float]): parameters of symmetric functions
        """
        self.net.eval()
        cartesians = cartesians.unsqueeze(0).to(device = self.device)
        
        g, dg = calculate_sf(cartesians, symm_func_params)
        g = g.squeeze(0)
        dg = dg.squeeze(0)
        g.requires_grad = True

        # calculate energies using NN
        energies = self.atomic_nets[0](g)
        de = torch.autograd.grad(torch.sum(energies), g, retain_graph = True)[0]
        with torch.no_grad():
            forces = -torch.einsum('ijk,ijkl->ijl', de, dg)
        return energies, forces
            
    def save_model(self, path: str):
        """Saves trained model to file.
        It makes possible to load pre-trained net for calculations
        """
        for i, net in enumerate(self.atomic_nets):
            torch.save(net.state_dict(), path + f"_{self.neural_net_data['atom_species'][i]}")