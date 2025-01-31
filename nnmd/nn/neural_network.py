#TODO:
# 1. total refactoring & clean-up
# 2. all methods except __init__ must be covered by tests
# 3. parameters of all methods must be checked
# 4. all methods must be documented

import torch
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

        # target device
        self.device = torch.device('cuda') if self.use_cuda else torch.device('cpu')
    
    def _describe_training(self, train_dataset_size, val_dataset_size, test_dataset_size, batch_size, epochs):
        from importlib.metadata import version

        module_name = "nnmd"

        self.net_log.info(f"NNMD v{version(module_name)}\n")
        self.net_log.info(f"---Neural network parameters---")
        self.net_log.info(f"device:                             {self.device}")
        self.net_log.info(f"training epochs:                    {epochs}")
        self.net_log.info(f"batch size:                         {batch_size}")
        self.net_log.info(f"optimizer:                          {self.optim.__class__.__name__}")
        self.net_log.info(f"loss function:                      {self.criterion.__class__.__name__}")
        self.net_log.info(f"energies loss function coefficient: {self.e_loss_coeff}")
        self.net_log.info(f"forces loss function coefficient:   {self.f_loss_coeff}")
        self.net_log.info(f"learning rate:                      {self.learning_rate}")
        self.net_log.info(f"scheduler:                          {self.sched.__class__.__name__}")
        self.net_log.info(f"L2 regularization coefficient:      {self.l2_regularization}\n\n")
        self.net_log.info(f"----Atomic NNs parameters----")
        self.net_log.info(f"input sizes (number of descriptors): {self.input_sizes}")
        self.net_log.info(f"hidden sizes:                        {self.hidden_sizes}")
        self.net_log.info(f"output sizes (number of atoms):      {self.output_sizes}\n\n")
        self.net_log.info(f"---Dataset parameters---")
        self.net_log.info(f"Training sample size:   {train_dataset_size}")
        self.net_log.info(f"Validation sample size: {val_dataset_size}")
        self.net_log.info(f"Test sample size:       {test_dataset_size}\n\n")

    def config(self, neural_net_data: dict, input_sizes: list[int], output_sizes: list[int], path: str = None):
        """Configures HDNN instance.

        Args:
            neural_net_data (dict): data about high-dimentional neural network and its atomic subnets
            input_size (int): size of input data, number of descriptors
            output_size (int): size of output data, number of atoms of each species
            path (str): path to pre-trained models. If not None and parameter in neural_network_data ("load_models") is True,
            models will be loaded
        """
        self.neural_net_data = neural_net_data

        # amount of atom species in dataset
        self.species = neural_net_data['atom_species']

        # params related to atomic nn and its config
        self.input_sizes = input_sizes
        self.hidden_sizes = neural_net_data['hidden_sizes']
        self.output_sizes = output_sizes

        # params related to training
        self.learning_rate = neural_net_data['learning_rate']
        self.l2_regularization = neural_net_data['l2_regularization']

        # coefficients of energies/forces importance in loss function
        self.e_loss_coeff = neural_net_data['e_loss_coeff']
        self.f_loss_coeff = neural_net_data['f_loss_coeff']
        
        # create Atomic NNs for each atom species
        self.atomic_nets = []
        for i in range(len(self.species)):
            net = AtomicNN(input_size = self.input_sizes[i],
                                             hidden_sizes = self.hidden_sizes[i],
                                             output_size = self.output_sizes[i])
            net = net.to(self.device)

            if neural_net_data['load_models']:
                path2model = "".join([path, f"atomic_nn_{neural_net_data['atom_species'][i]}.pt"])
                net.load_state_dict(torch.load(path2model))
            else:
                # initialize weights of Atomic NNs
                for m in net.model:
                    # Xavier weights initialization works better on linear layers
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight)
                        nn.init.constant_(m.bias, 0)

            self.atomic_nets.append(net)

        # loss function
        self.criterion = torch.nn.MSELoss()

        params = []
        for net in self.atomic_nets:
            params += list(net.parameters())

        # Atomic nets optimizer
        self.optim = torch.optim.Adam(params = params,
                                      lr = self.learning_rate,
                                      weight_decay = self.l2_regularization)
    
    def _loss(self, e_nn: torch.Tensor, energies: torch.Tensor,
                    f_nn: torch.Tensor, forces: torch.Tensor, n_struct: int, n_atoms: int) -> torch.Tensor:
        """Gets loss of calculations.

        Args:
            e_nn (torch.Tensor): calculated energy
            energy (torch.Tensor): target energy
            f_nn (torch.Tensor): calculated forces
            forces (torch.Tensor): target forces
        """
        E_loss = self.e_loss_coeff * self.criterion(e_nn, energies)
        F_loss = self.f_loss_coeff / 3 * self.criterion(f_nn, forces)
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
        self.sched = torch.optim.lr_scheduler.ExponentialLR(self.optim, gamma = 0.99)
        #self.sched = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim, mode = 'min', factor = 0.75, patience = 5)
        
        # output log file
        self.net_log = Logger.get_logger("Net training info", "net.log")

        # save package/environment/nn info to log file
        self._describe_training(len(train_dataset), len(val_dataset), len(test_dataset), batch_size, epochs)

        # current learning rate
        curr_lr = self.optim.param_groups[0]['lr']
        
        # format of loss info message
        loss_info = "RMSE E = {e_loss:e}, RMSE F = {f_loss:e}, RMSE total = {loss:e} eV"

        # run training
        for epoch in range(epochs):
            e_loss, f_loss, loss = self._train_loop(epoch, train_loader)
            
            self.net_log.info(f"Epoch {epoch + 1}: training: " + loss_info.format(e_loss = torch.sqrt(e_loss).item(),
                                                                                  f_loss = torch.sqrt(f_loss).item(), 
                                                                                  loss = torch.sqrt(loss).item()))

            e_loss, f_loss, loss = self._validate(epoch, val_loader)
            
            self.net_log.info(f"Epoch {epoch + 1}: validation: " + loss_info.format(e_loss = torch.sqrt(e_loss).item(),
                                                                                    f_loss = torch.sqrt(f_loss).item(), 
                                                                                    loss = torch.sqrt(loss).item()))

            # if scheduler is used
            # update learning rate by rule
            if self.sched:
                if isinstance(self.sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.sched.step(loss)
                else:
                    self.sched.step()

            # print the learning rate at the end of the epoch
            # if it was changed
            if self.optim.param_groups[0]['lr'] != curr_lr:
                curr_lr = self.optim.param_groups[0]['lr']
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
            self.optim.zero_grad(set_to_none = True)

            # input data in batch:
            # g values (nn inputs),
            # dG values (for forces calculation),
            # energies and forces (nn targets)
            g, dG, energy, forces = data

            # normalize energies and forces by energy range 
            # calculate energies and forces using NN
            e_nn = []
            f_nn = []
            for i, atom_type in enumerate(self.species):
                e_nn_atom_type = self.atomic_nets[i](g[atom_type])
                e_nn.append(e_nn_atom_type)

                # calculate gradients of E by G
                dE = torch.autograd.grad(e_nn_atom_type.sum(), g[atom_type],
                                    create_graph = True)[0]

                # calculate forces using gradients of E values
                # and G values gradients by Einstein summation rule
                f_nn_atom_type = -torch.einsum('ijk,ijkl->ijl', dE, dG[atom_type])
                f_nn.append(f_nn_atom_type)

            # concatenate energies and forces for all atom species
            e_nn = torch.cat(e_nn, dim = 0)
            f_nn = torch.cat(f_nn, dim = 0)
            batch_loss, batch_e_loss, batch_f_loss = self._loss(e_nn.sum(dim = 1), energy, 
                                                                f_nn, forces, 
                                                                forces.size(0), forces.size(1))
            batch_loss.backward(retain_graph = True)
            self.optim.step()

            loss += batch_loss
            e_loss += batch_e_loss
            f_loss += batch_f_loss

        # normalize total epoch losses
        e_loss /= len(train_loader)
        f_loss /= len(train_loader)
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
            # g values (nn inputs),
            # dg values (necessary in forces calculation),
            # energies and forces (nn targets)
            g, dG, energy, forces = data

            e_nn = []
            f_nn = []
            for i, atom_type in enumerate(self.species):
                # send data to device
                dG_batch = dG[atom_type]
                g_batch = g[atom_type]

                e_nn_atom_type = self.atomic_nets[i](g_batch)
                e_nn.append(e_nn_atom_type)

                # calculate gradients of E by G
                dE = torch.autograd.grad(e_nn_atom_type.sum(), g_batch,
                                    create_graph = True)[0]
                
                # calculate forces using gradients of E values
                # and G values gradients by Einstein summation rule
                f_nn_atom_type = -torch.einsum('ijk,ijkl->ijl', dE, dG_batch)
                f_nn.append(f_nn_atom_type)

            # concatenate energies and forces for all atom species
            e_nn = torch.cat(e_nn, dim = 0)
            f_nn = torch.cat(f_nn, dim = 0)

            energy = energy.to(device = self.device)
            forces = forces.to(device = self.device)
            
            batch_loss, batch_e_loss, batch_f_loss = self._loss(e_nn.sum(dim = 1), 
                                                                energy, f_nn, forces, 
                                                                forces.size(0), forces.size(1))

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
            # g values (nn inputs),
            # dg values (necessary in forces calculation),
            # energies and forces (nn targets)
            g, dG, energy, forces = data

            # calculate energies  and forces using NN
            e_nn = []
            f_nn = []
            for i, atom_type in enumerate(self.species):
                dG_batch = dG[atom_type]
                g_batch = g[atom_type]

                e_nn_atom_type = self.atomic_nets[i](g_batch)
                e_nn.append(e_nn_atom_type)

                # calculate gradients of E by G
                dE = torch.autograd.grad(e_nn_atom_type.sum(), g_batch,
                                    create_graph = True)[0]

                # calculate forces using gradients of E values
                # and G values gradients by Einstein summation rule
                f_nn_atom_type = -torch.einsum('ijk,ijkl->ijl', dE, dG_batch)
                f_nn.append(f_nn_atom_type)

            # concatenate energies and forces for all atom species
            e_nn = torch.cat(e_nn, dim = 0)
            f_nn = torch.cat(f_nn, dim = 0)

            energy = energy.to(device = self.device)
            forces = forces.to(device = self.device)

            batch_loss, batch_e_loss, batch_f_loss = self._loss(e_nn.sum(dim = 1), energy, 
                                                                f_nn, forces, 
                                                                forces.size(0), forces.size(1))

            loss += batch_loss
            e_loss += batch_e_loss
            f_loss += batch_f_loss

        # normalize total epoch losses
        f_loss /= len(test_loader)
        e_loss /= len(test_loader)
        loss /= len(test_loader)    

        return e_loss, f_loss, loss

    def predict(self, cartesians: torch.Tensor, cell: torch.Tensor, symm_func_params: dict[str, float]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculates energy and forces for structs of atoms

        Args:
            cartesians (torch.Tensor): positions of atoms
            symm_func_params (dict[str, float]): parameters of symmetric functions
        """
        self.net.eval()
        cartesians = cartesians.unsqueeze(0).to(device = self.device)
        
        g, dg = calculate_sf(cartesians, cell, symm_func_params)
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
            torch.save(net.state_dict(), path + f"/atomic_nn_{self.neural_net_data['atom_species'][i]}.pth")