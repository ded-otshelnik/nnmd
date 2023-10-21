import numpy as np
import math

from .symm import g1, g2, g3, g4, g5

def calculate_sf(ri, cartesians, g_type: int, r_cutoff: float,
                eta: float = None, k: float = None, lambda_: float = None, xi: float = None, rs: float = None):
    

    # match by type of symmetric func
    match g_type:
        # calculate g1
        case 1:
            g, dg = 0, 0
            for rj in cartesians:
                # if i = j
                if np.equal(ri, rj).all():
                    # do not use in calculations
                    continue
                # distance between centers of atoms
                rij = math.sqrt(sum((i - j) ** 2 for i, j in zip(ri, rj)))
                # g1 and its derivatives
                g, dg += g1(rij, r_cutoff)
        # calculate g2
        case 2:
            g, dg = 0, 0
            for rj in cartesians:
                # if i = j
                if np.equal(ri, rj).all():
                    # do not use in calculations
                    continue
                # distance between centers of atoms
                rij = math.sqrt(sum((i - j) ** 2 for i, j in zip(ri, rj)))
                # g2 and its derivatives
                g, dg += g2(eta, rij, r_cutoff)
        # calculate g3
        case 3:
            g, dg = 0, 0
            for rj in cartesians:
                # if i = j
                if np.equal(ri, rj).all():
                    # do not use in calculations
                    continue
                # distance between centers of atoms
                rij = math.sqrt(sum((i - j) ** 2 for i, j in zip(ri, rj)))
                # g3 and its derivatives
                g, dg += g3(k, rij, r_cutoff)
        # calculate g4
        case 4:
            g, dg = 0, [0, 0, 0]
            for rj in cartesians:
                for rk in cartesians:
                    # if j = k, i = j or i = k
                    if np.equal(rj, rk).all() or np.equal(ri, rj).all() or np.equal(ri, rk).all():
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
                    # g4 and its derivatives
                    output = g4(eta, xi, lambda_, rij, rik, rjk, cos_v, r_cutoff)
                    g += output[0]
                    for i in range(3):
                        dg[i] += output[1, i]
        # calculate g5
        case 5:
            g, dg = 0, [0, 0, 0]
            for rj in cartesians:
                for rk in cartesians:
                    # if j = k, i = j or i = k
                    if np.equal(rj, rk).all() or np.equal(ri, rj).all() or np.equal(ri, rk).all():
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
                    # g5 and its derivatives
                    output = g5(eta, xi, lambda_, rij, rik, cos_v, r_cutoff)
                    g += output[0]
                    for i in range(3):
                        dg[i] += output[1, i]
        case _ :
            raise ValueError("Incorrect type of symmetric func")
    # return values
    return g, dg