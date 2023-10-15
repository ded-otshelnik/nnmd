import numpy as np

from .symm import g2

def calculate_sf(natoms: int, cartesians, g_type: int, r_cutoff: float, eta: float):
    """Calculates symmetry functions for each atom 

    Args:
        natoms (int): number of atoms
        cartesians: array of atoms cartesian coordinates
        g_type (int): type of symmetric function
        r_cutoff (float): cutoff radius
        eta (float): parameter of stymmetic function

    Raises:
        ValueError: if g_type are incorrect (not 2)
    """
    # check for type of symmetric func
    if g_type != 2:
        raise ValueError("Only G2 func has been implemented")
    g = g2
    g_arr = []
    for ri in cartesians:
        gi = 0
        for rj in cartesians:
            rij = ri - rj
            gi += g(eta, rij, r_cutoff)
        g_arr.append(gi)