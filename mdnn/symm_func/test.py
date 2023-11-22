import numpy as np
from ..util.decorators import timer

import symm_func
from .symm import sf

rc = 12.0
eta = 0.3
rs = 1
k = 0.1
_lambda = 1
xi  = 2

def calculate_g(cartesians):
        """Calculates g and dg(by atoms and by parameters) values for each atom in structure.

        Args:
            cartesians: coordinates of atoms
        """
        def preprocess_g1(cartesians):
            """Computes G1 values on set of atoms.

            Args:
                cartesians: coordinates of atoms
            """
            g1 = []
            for atom in cartesians:
                g1.append(sf.calculate_sf(atom, cartesians, sf.G_TYPE(1), r_cutoff=rc))
            return g1
        
        g1 = preprocess_g1(cartesians)
        # descriptors array
        g = []
        # derivatives by radius array
        derivatives_g_r = []
        
        # loop by atom
        for i, atom in enumerate(cartesians):
            # atom descriptors and its derivatives 
            g_atom = [g1[i].g]
            dg_atom = g1[i].dg
            # calculate descriptors for atom with parameters
            for g_type in [2, 3, 4, 5]:
                gi = sf.calculate_sf(atom, cartesians, sf.G_TYPE(g_type),
                                r_cutoff=rc, eta=eta, rs=rs, k=k, lambda_=_lambda, xi=xi)
                g_atom.append(gi.g)
                for i in range(3):
                    dg_atom[i] += gi.dg[i]

            g.append(g_atom)
            derivatives_g_r.append(dg_atom)
        
        g_min = [min([g_item[i] for g_item in g]) for i in range(5)]
        g_max = [max([g_item[i] for g_item in g]) for i in range(5)]

        g = [[(g_item[i] - g_min[i]) / (g_max[i] - g_min[i]) for i in range(5)] for g_item in g]

        dg_min = [min([dg_item[i] for dg_item in derivatives_g_r]) for i in range(3)]
        dg_max = [max([dg_item[i] for dg_item in derivatives_g_r]) for i in range(3)]
        derivatives_g_r = [[(dg_item[i] - dg_min[i]) / (dg_max[i] - dg_min[i]) for i in range(3)] for dg_item in derivatives_g_r]
        return g, derivatives_g_r
        
@timer
def cpp_symm(cartesians):
    for i in range(len(cartesians)):
        _ = symm_func.calculate_g(cartesians[i], rc, eta, rs, k, _lambda, xi)

@timer
def py_symm(cartesians):
    for i in range(len(cartesians)):
        _ = calculate_g(cartesians[i])

from params_parser import parser

n_atoms, cartesians, forces, energies = parser('Cu111.txt')

py_symm(cartesians)
cpp_symm(cartesians)