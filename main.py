import os
import shutil
import torch

from mdnn.nn.neural_network import Neural_Network
from mdnn.util.params_parser import parser

file = './samples/Cu111.txt'
n_atoms, cartesians, f_dft, e_dft = parser(file)

hidden_nodes = [n_atoms[0], 15, 10]
rc = 12.0

eta, rs, k, _lambda, xi = 0.01, 0.5, 1, -1, 3
device = torch.device('cuda') if not torch.cuda.is_available() else torch.device('cpu')
net = Neural_Network(len(cartesians), n_atoms[0], rc, hidden_nodes, epochs=300)
net.to(device=device)

net.compile(cartesians, eta, rs, k, _lambda, xi, load_models = False, path = 'models')

# train models
train = True
# save models as files
save = True
# try to predict energy and forces
predict = False

if train:
    net.fit(e_dft, f_dft)

if predict:
    e_nn, f_nn = net.predict(cartesians[-1])
    net.loss(e_nn, torch.tensor(e_dft[-1]), f_nn, torch.tensor(f_dft[-1]))

if save:
    if os.path.exists('models'):
        shutil.rmtree('models', ignore_errors=True)
    os.mkdir('models')
    net.save_model()