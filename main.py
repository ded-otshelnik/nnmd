import os
import shutil
import torch
import time 

from nnmd.nn.neural_network import Neural_Network
from nnmd.util.params_parser import parser

print("Getting info from GPAW simulation")
# get info from GPAW simulation
file = './samples/Cu111.txt'
n_atoms, cartesians, f_dft, e_dft = parser(file)
n_structs = len(cartesians)

print("Create an instance of NN")
# Global parameters of HDNN 
# Atomic NN, nodes amount in hidden layers
hidden_nodes = [40, 30]
epochs = 1000
# turn off/on GPU usage
use_cuda = False 

net = Neural_Network(hidden_nodes=hidden_nodes,
                     epochs = epochs,
                     use_cuda = use_cuda)

device = torch.device('cuda') if torch.cuda.is_available() and use_cuda else torch.device('cpu')
print(f"Move NN to {'GPU' if device.type == 'cuda' else 'CPU'}")
# move NN to right device 
net.to(device=device)

# 80% - train, 20% - test
sep = int(0.8 * n_structs)
print(f"Separate data to train and test datasets")

train_data, test_data = cartesians[:sep], cartesians[sep:]
train_e_dft, test_e_dft = e_dft[:sep], e_dft[sep:]
train_f_dft, test_f_dft = f_dft[:sep], f_dft[sep:]

# params that defines what we will do with NN
# train model
train = True
# predict energy and forces
predict = True
# save models params as files
save = False
path = 'models'

print("Config subnets and prepare data", end='\n\n')
# prepare data for nets and its subnets
eta, rs, k, _lambda, xi = 0.01, 0.5, 1, -1, 3
rc = 12.0
net.compile(train_data, len(train_data), len(train_data[0]), rc, eta, rs, k, _lambda, xi)

if train:
    print("Start training")
    start = time.time()
    net.fit(train_e_dft, train_f_dft)
    end = time.time()
    print("End training")
    print(f"Training time ({'GPU' if device.type == 'cuda' else 'CPU'}): {(end - start):.3f} s", end='\n\n')

if predict:
    print("Start testing")

    test_e_dft, test_f_dft = torch.tensor(test_e_dft, device=device), torch.tensor(test_f_dft, device=device)
    test_e_nn, test_f_nn = net.predict(test_data)
    test_loss = net.loss(test_e_nn, test_e_dft, test_f_nn, test_f_dft)

    print("End testing")

    test_loss_info = f"Test loss: {test_loss.cpu().detach().numpy():.3f} eV"

    print(test_loss_info, file=net.log)

if save:
    print("Save model")

    if os.path.exists('models'):
        shutil.rmtree('models', ignore_errors=True)
    os.mkdir('models')
    net.save_model()