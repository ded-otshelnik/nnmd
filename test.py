import torch
from nnmd.util.params_parser import parser

import time
import mdnn_cuda
import mdnn_cpp

file = './samples/Cu111.txt'
# file = './samples/ag.txt'
n_atoms, cartesians, f_dft, e_dft = parser(file)

rc = 12.0

eta, rs, k, _lambda, xi = 0.01, 0.5, 1, -1, 3

cartesians_cpp = torch.tensor(cartesians, dtype=torch.double)

start = time.time()
g = mdnn_cpp.calculate_sf(cartesians_cpp[0], rc, eta, rs, k, _lambda, xi)
end = time.time()
print("C++: ", (end - start))

device = torch.device('cuda')
cartesians_cuda = torch.tensor(cartesians, device=device, dtype=torch.double)

start = time.time()
g = mdnn_cuda.calculate_sf(cartesians_cuda[0], rc, eta, rs, k, _lambda, xi)
end = time.time()
print("CUDA:", (end - start))