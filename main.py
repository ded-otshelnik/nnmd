import os
import shutil
import torch
import time 

from mdnn.nn.neural_network import Neural_Network
from mdnn.util.params_parser import parser

file = './samples/Cu111.txt'
n_atoms, cartesians, f_dft, e_dft = parser(file)

hidden_nodes = [40, 30]

cuda_ = True 
device = torch.device('cuda') if torch.cuda.is_available() and cuda_ else torch.device('cpu')

net = Neural_Network(len(cartesians), n_atoms[0], hidden_nodes, epochs=100)
net.to(device=device)

# prepare data and subnets for nets 
eta, rs, k, _lambda, xi = 0.01, 0.5, 1, -1, 3
rc = 12.0
net.compile(cartesians, rc, eta, rs, k, _lambda, xi, load_models = False, path = 'models')

# train models
train = True
# save models as files
save = True
# try to predict energy and forces
predict = False

if train:
    net.fit(e_dft, f_dft)

if predict:
    start = time.time()
    e_nn, f_nn = net.predict(cartesians)
    end = time.time()
    print("CUDA train (Total):" if cuda_ else "CPU train (Total):", (end - start), end="\n")

if save:
    if os.path.exists('models'):
        shutil.rmtree('models', ignore_errors=True)
    os.mkdir('models')
    net.save_model()