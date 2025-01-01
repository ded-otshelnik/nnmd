#TODO:
# 1. total refactoring & clean-up
# 3. parameters of all methods must be checked

# must be imported before C++ extention
# for compatibility
import math
import torch
from torch import nn
torch.manual_seed(0)
from torch.utils.data import DataLoader

# C++/CUDA extention
import nnmd_cpp

from . import AtomicNN
from .dataset import AtomicDataset, TrainAtomicDataset
from ..util._logger import Logger
from ..util.calculate_g import calculate_g

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
    def __init__(self) -> None:
        """Init method of HDNN class
        """
        super().__init__()
    
    def _describe_training(self, symm_func_params, train_dataset_size, val_dataset_size, batch_size, epochs):
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
        self.net_log.info(f"cutoff radius: {symm_func_params['r_cutoff']}")
        self.net_log.info(f"eta:           {symm_func_params['eta']}")
        self.net_log.info(f"rs:            {symm_func_params['rs']}")
        self.net_log.info(f"kappa:         {symm_func_params['kappa']}")
        self.net_log.info(f"lambda:        {symm_func_params['lambda']}")
        self.net_log.info(f"zeta:          {symm_func_params['zeta']}\n")
        self.net_log.info(f"Training sample size:   {train_dataset_size}")
        self.net_log.info(f"Validation sample size: {val_dataset_size}")


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
            self.net.load_state_dict(torch.load(kwargs['path']))         
        self.net = self.net.to(device = self.device)

        # Atomic NN optimizer
        self.optim = torch.optim.Adam(self.net.parameters(), lr = self.learning_rate, weight_decay = self.l2_regularization)
        
        # output log files
        self.net_log = Logger.get_logger("net train info", "net.log")
        self.time_log = Logger.get_logger("net time info", "time.log")

    def _get_lr(self):
        for param_group in self.optim.param_groups:
            return param_group['lr']
        
    def fit(self, train_dataset: TrainAtomicDataset,
             val_dataset: TrainAtomicDataset,
             batch_size: int, epochs: int):
        """Train method of neural network.

        Args:
            dataset (AtomicDataset): input dataset with data about atoms
            batch_size (int): size of batch
            epochs (int): amount of training epochs
        """
        train_num_batches = math.floor(len(train_dataset) / batch_size) if len(train_dataset) > batch_size else 1
        val_num_batches = math.floor(len(val_dataset) / batch_size) if len(val_dataset) > batch_size else 1

        train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, drop_last = True)
        val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = True, drop_last = True)

        # scheduler for Atomic NN optimizer
        self.sched = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim, mode = 'min',
                                                               factor = 0.8, patience = train_num_batches * 10, min_lr = 1e-5)
        # save package/environment/nn info to log file
        self._describe_training(train_dataset.symm_func_params, len(train_dataset), len(val_dataset), batch_size, epochs)

        # run training
        curr_lr = self._get_lr()
        for epoch in range(epochs):
            start = time.time()
            self._train_loop(epoch, train_loader, train_num_batches)
            self._validate(epoch, val_loader, val_num_batches)
            # Print the learning rate at the end of each epoch
            if self._get_lr() != curr_lr:
                curr_lr = self._get_lr()
                self.net_log.info(f"Learning rate changed to {curr_lr}")
                self.time_log.info(f"epoch {epoch + 1} elapsed: {(time.time() - start):.3f} s")

    def _train_loop(self, epoch: int, train_loader: DataLoader, num_batches: int):
        """Train loop method. 

        Args:
            epoch (int): current training epoch
            train_loader (DataLoader): loads data into the pipeline for training
            batch_size (int): size of batch
            symm_func_params (dict[str, float]): parameters of symmetric functions
            h (float): step of coordinate-wise moving (used in forces caclulations).
        """
        # set model to training mode
        self.net.train()
        # epoch losses: energies, forces and total
        e_loss = torch.zeros(1, device = self.device).squeeze()
        f_loss = torch.zeros(1, device = self.device).squeeze()
        loss = torch.zeros(1, device = self.device).squeeze()

        self.time_log.info(f"Epoch {epoch + 1}\nTraining")

        for batch, data in tqdm(enumerate(train_loader), desc = f"Epoch {epoch + 1}, Training"):
            # input data in batch:
            # g values (nn inputs),
            # dG values (for forces calculation),
            # energies and forces (nn targets)
            # reset params grad to None
            for param in self.net.parameters():
                param.grad = None
            g, dG, energy, forces = data
            
            start = time.time()

            # calculate energies using NN
            # and gradients by G values using Autograd Engine
            # print(g)
            e_nn = self.net(g)
            # if batch == 0:
            #     print("e_nn: ", e_nn)
            #     print(e_nn.sum(dim = 1).squeeze(1))
            dE = torch.autograd.grad(torch.sum(e_nn), g, retain_graph = True)[0]
            # if batch == 0:
            #     print("dE: ", dE)

            f_nn = torch.mul(torch.sum(dG, dim = 2), torch.sum(dE, dim = 2).unsqueeze(2))
            # if batch == 0:
            #     print("f_nn: ", f_nn)

            end = time.time()
            self.time_log.info(f"Epoch {epoch + 1}, batch {batch + 1}, forward: {(end - start):.5f} s")
            batch_loss, batch_e_loss, batch_f_loss = self.loss(e_nn.sum(dim = 1).squeeze(1), energy, f_nn, forces)

            start = time.time()
            batch_loss.backward(retain_graph = True)
            end = time.time()
            self.time_log.info(f"Epoch {epoch + 1}, batch {batch + 1}, backpropagation: {(end - start):.3f} s")

            self.optim.step()
            if self.sched:
                self.sched.step(batch_loss)

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
        loss_info = f"RMSE E = {e_loss:e}, RMSE F = {f_loss:e}, RMSE total = {loss:e} eV"
        self.net_log.info(f"Epoch {epoch + 1}, training:   " + loss_info)

    def _validate(self, epoch: int, val_loader: DataLoader, num_batches: int):
        """Train loop method. 

        Args:
            epoch (int): current training epoch
            train_loader (DataLoader): loads data into the pipeline for training
            batch_size (int): size of batch
            symm_func_params (dict[str, float]): parameters of symmetric functions
            h (float): step of coordinate-wise moving (used in forces caclulations).
        """
        self.net.eval()
        # epoch losses: energies, forces and total
        e_loss = torch.zeros(1, device = self.device).squeeze()
        f_loss = torch.zeros(1, device = self.device).squeeze()
        loss = torch.zeros(1, device = self.device).squeeze()

        self.time_log.info(f"Epoch {epoch + 1}\nValidation")
        for batch, data in tqdm(enumerate(val_loader), desc = f"Epoch {epoch + 1}, Validation"):
            # input data in batch:
            # positions (necessary in forces calculation),
            # g values (nn inputs),
            # energies and forces (nn targets)
            g, dG, energy, forces = data
            
            start = time.time()

            # calculate energies using NN
            # and gradients by G values using Autograd Engine
            e_nn = self.net(g)
            dE = torch.autograd.grad(torch.sum(e_nn), g, retain_graph = True)[0]
            f_nn = torch.mul(torch.sum(dG, dim = 2), torch.sum(dE, dim = 2).unsqueeze(2))
            
            end = time.time()
            self.time_log.info(f"Epoch {epoch + 1}, batch {batch}, forward: {(end - start):.5f} s")
            
            batch_loss, batch_e_loss, batch_f_loss = self.loss(e_nn.sum(dim = 1).squeeze(1), energy, f_nn, forces)

            self.time_log.info(f"Epoch {epoch + 1}, batch {batch}, backpropagation: {(end - start):.3f} s")

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
        loss_info = f"RMSE E = {e_loss:e}, RMSE F = {f_loss:e}, RMSE total = {loss:e} eV"
        self.net_log.info(f"Epoch {epoch + 1}: validation: " + loss_info)
             
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
        print("E_loss: ", E_loss)
        F_loss = self.criterion(f_nn, forces) * (self.mu / 3)
        print("F_loss: ", F_loss)
        loss = E_loss + F_loss
        print("loss: ", loss)
        return loss, E_loss, F_loss

    def predict(self, cartesians: torch.Tensor, symm_func_params: dict[str, float]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculates energy and forces for structs of atoms

        Args:
            cartesians (torch.Tensor): positions of atoms
            symm_func_params (dict[str, float]): parameters of symmetric functions
        """
        self.net.eval()
        cartesians = cartesians.unsqueeze(0).to(device = self.device)
        e_nn = torch.empty(self.n_atoms, device = self.device, dtype = self.dtype)
        
        g, dG = calculate_g(cartesians, self.device, symm_func_params)
        g = g.squeeze(0)
        dG = dG.squeeze(0)
        g.requires_grad = True

        # calculate energies using NN
        e_nn = self.net(g)
        dE = torch.autograd.grad(torch.sum(e_nn), g, retain_graph = True)[0]
        with torch.no_grad():
            f_nn = torch.mul(torch.sum(dG, dim = 1), torch.sum(dE, dim = 1).unsqueeze(1)).squeeze(0)
        return e_nn, f_nn
            
    def save_model(self, path: str):
        """Saves trained model to file.
        It makes possible to load pre-trained net for calculations
        """
        torch.save(self.net.state_dict(), path)