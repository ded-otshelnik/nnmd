import numpy as np
import math

from .symm import g2, g4, g5

def calculate_sf(ri, cartesians, g_type: int, r_cutoff: float, eta: float, lambda_: float = None, xi: float = None):
    """Calculates symmetry function for atom 

    Args:
        ri: atom for which function is calculated
        cartesians: array of atoms cartesian coordinates
        g_type (int): type of symmetric function
        r_cutoff (float): cutoff radius
        eta (float): parameter of stymmetic function

    Raises:
        ValueError: if g_type are incorrect (not 2, 4 or 5)
    """

    # match by type of symmetric func
    match g_type:
        case 2:
            g, dg = 0, 0
            for rj in cartesians:
                # distance between centers of atoms
                rij = math.sqrt(sum((i - j) ** 2 for i, j in zip(ri, rj)))

                g, dg += g2(eta, rij, r_cutoff)
        case 4:
            g, dg = 0, [0, 0, 0]
            for rj in cartesians:
                for rk in cartesians:
                    # if rj and rk are equal (i.e. k = j)
                    if np.equal(rj, rk).all():
                        # do not use in calculations
                        continue 
                    # distances between centers of atoms
                    rij = math.sqrt(sum((i - j) ** 2 for i, j in zip(ri, rj)))
                    rik = math.sqrt(sum((i - k) ** 2 for i, k in zip(ri, rk)))
                    rjk = math.sqrt(sum((j - k) ** 2 for j, k in zip(rj, rk)))
                    # cosine of an angle between radius vectors and its derivatives
                    cos_v = (rij * rij + rik * rik - rjk * rjk) / (2 * rij * rik)
                    dcos_v = [0, 0, 0]
                    dcos_v[0] = 0.5 * (1 / rik + 1 / rij / rij * (rjk * rjk / rik - rik))
                    dcos_v[1] = 0.5 * (1 / rij + 1 / rik / rik * (rjk * rjk / rij - rij))
                    dcos_v[2] = -rjk / rij / rik
                    # g and its derivatives
                    output = g4(eta, xi, lambda_, rij, rik, rjk, cos_v, r_cutoff)
                    g += output[0]
                    for i in range(3):
                        dg[i] += output[1, i]
        case 5:
            g, dg = 0, [0, 0, 0]
            for rj in cartesians:
                for rk in cartesians:
                    # if rj and rk are equal (i.e. k = j)
                    if np.equal(rj, rk).all():
                        # do not use in calculations
                        continue 
                    # distances between centers of atoms
                    rij = math.sqrt(sum((i - j) ** 2 for i, j in zip(ri, rj)))
                    rik = math.sqrt(sum((i - k) ** 2 for i, k in zip(ri, rk)))
                    rjk = math.sqrt(sum((j - k) ** 2 for j, k in zip(rj, rk)))
                    # cosine of an angle between radius vectors and its derivatives
                    cos_v = (rij * rij + rik * rik - rjk * rjk) / (2 * rij * rik)
                    dcos_v = [0, 0, 0]
                    dcos_v[0] = 0.5 * (1 / rik + 1 / rij / rij * (rjk * rjk / rik - rik))
                    dcos_v[1] = 0.5 * (1 / rij + 1 / rik / rik * (rjk * rjk / rij - rij))
                    dcos_v[2] = -rjk / rij / rik
                    # g and its derivatives
                    output = g5(eta, xi, lambda_, rij, rik, cos_v, r_cutoff)
                    g += output[0]
                    for i in range(3):
                        dg[i] += output[1, i]
    # return values
    return g, dg