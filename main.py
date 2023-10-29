from mdnn.nn.atomic_nn import AtomicNN
from mdnn.nn.neural_network import NeuralNetwork

from mdnn.util.params_parser import parser
from mdnn.symmetry_functions.sf import calculate_sf, G_TYPE
from mdnn.symmetry_functions.pair_g import PairG

from tqdm import 

file = 'example.txt'
cartesians, forces = parser(file)

rc = 3.0
eta = 0.1
rs = 0
k = 1
lambda_ = 1
xi = 1

print("G:")

pairs_g = []
for struct in cartesians:
    pairs_struct = []
    for ri in struct:
        pair = calculate_sf(ri, struct, G_TYPE.G1, r_cutoff=rc)
        pair += calculate_sf(ri, struct, G_TYPE.G2, r_cutoff=rc,
                                                    eta=eta, rs=rs)
        pair += calculate_sf(ri, struct, G_TYPE.G3, r_cutoff=rc, k=k)
        pair += calculate_sf(ri, struct, G_TYPE.G4, r_cutoff=rc, eta=eta, rs=rs, \
                                                    lambda_=lambda_, xi=xi)
        pair += calculate_sf(ri, struct, G_TYPE.G5, rc, eta=eta, rs=rs,
                                                    lambda_=lambda_, xi=xi)
        pairs_struct.append(pair)
        print(pair)
    pairs_g.append(pairs_struct)
print(pairs_g)