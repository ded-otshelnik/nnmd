# from mdnn.nn.atomic_nn import AtomicNN

from mdnn.util.params_parser import parser
from mdnn.symmetry_functions.sf import calculate_sf, G_TYPE
from mdnn.symmetry_functions.pair_g import PairG

from itertools import product

from tqdm import tqdm

import numpy as np

def expit_mod(x, A, b, c):
    """Modified sigmoid function

    Args:
        x: array of values 
        A: normalization constant
        b, c: constants what determine the function shape
    """
    return A * x / (1 + np.exp(- b * (x - c)))


file = 'Cu111.txt'
n_atoms, cartesians, forces, energies = parser(file)

rc = 12.0
eta = 0.1
rs = 0
k = 0.5
lambda_ = 1
xi = 1.0    

# ch = product([0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.1, 0.3, 0.5],
#             [0, 0.5, 1, 1.5, 2],
#             [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
#             [-1, 1],
#             [0.5,1,2,3,4])
# file = open('log.txt', 'w+')
# print(energies[0],file=file)

# for eta, rs, k, lambda_, xi in ch:

#     energy = []
#     for cart in cartesians[0]:
#         temp = []
#         for i in [1, 2, 3, 4, 5]:
#             temp.append(calculate_sf(cart, cartesians[0], G_TYPE(i),
#                                     r_cutoff=rc, eta=eta, rs=rs, k=k, lambda_=lambda_, xi=xi).g)
#         energy.append(temp) 


#     min_ = [min(energy[:][i]) for i in range(5)]
#     max_ = [max(energy[:][i]) for i in range(5)]

#     scaled_energy = sum([sum([(2 * g[i] - min_[i]) / (max_[i] - min_[i]) - 1 for i in range(5)]) for g in energy])

#     print(scaled_energy, file=file)
#     print(f'rc = {rc}, eta = {eta}, rs = {rs}, k = {k}, lambda = {lambda_}, xi  = {xi}',
#           file=file, end='\n\n')
for struct, energy_ in zip(cartesians, energies):
    energy = []
    for cart in struct:
        temp = []
        for i in [1, 2, 3, 4, 5]:
            temp.append(calculate_sf(cart, struct, G_TYPE(i),
                                    r_cutoff=rc, eta=eta, rs=rs, k=k, lambda_=lambda_, xi=xi).g)
        energy.append(temp) 


    min_ = [min(energy[:][i]) for i in range(5)]
    max_ = [max(energy[:][i]) for i in range(5)]

    temp = [[(2 * g[i] - min_[i]) / (max_[i] - min_[i]) - 1 for i in range(5)] for g in energy]
    temp = [sum(i) for i in zip(*temp)]
    print("Grouped values of G(1-5): ", temp)
    scaled_energy = sum([sum([(2 * g[i] - min_[i]) / (max_[i] - min_[i]) - 1 for i in range(5)]) for g in energy])
    print("Computed scaled energy: ", scaled_energy)
    print("Computed after sigmoid: ", expit_mod(scaled_energy, 100, 1, 12.544))
    print("Must be:", energy_)
    print(f'rc = {rc}, eta = {eta}, rs = {rs}, k = {k}, lambda = {lambda_}, xi  = {xi}',
            end='\n\n')