import numpy as np
import math

from enum import Enum

from ..symmetry_functions.pair_g import PairG
from .symm import *

class G_TYPE(Enum):
    """Class implements an enumeration of types of g functions
    """
    G1 = 1
    G2 = 2
    G3 = 3
    G4 = 4
    G5 = 5

def calculate_sf(ri, cartesians, g_type: int, r_cutoff: float,
                eta: float = None, rs: float = None, k: float = None,
                lambda_: float = None, xi: float = None):
    """Calculates symmetric function for atom \
    with an environment of other atoms

    Args:
        ri (_type_): _description_
        cartesians (_type_): _description_
        g_type (int): _description_
        r_cutoff (float): _description_
        eta (float, optional): _description_. Defaults to None.
        k (float, optional): _description_. Defaults to None.
        lambda_ (float, optional): _description_. Defaults to None.
        xi (float, optional): _description_. Defaults to None.
        rs (float, optional): _description_. Defaults to None.

    Raises:
        ValueError: _description_

    Returns:
        Pair_G: _description_
    """
    # match by type of symmetric func
    match g_type:
        # calculate g1
        case G_TYPE.G1:
            g, dg = 0, [0, 0, 0]
            for rj in cartesians:
                # if i = j
                if np.equal(ri, rj).all():
                    # do not use in calculations
                    continue
                # distance between centers of atoms
                rij = math.sqrt(sum((i - j) ** 2 for i, j in zip(ri, rj)))
                # g1 and its derivatives
                output_g, output_dg = g1(rij, r_cutoff)
                g += output_g
                for i in range(3):
                    dg[i] += output_dg[i]
        # calculate g2
        case G_TYPE.G2:
            g, dg = 0, [0, 0, 0]
            for rj in cartesians:
                # if i = j
                if np.isclose(ri, rj).all():
                    # do not use in calculations
                    continue
                # distance between centers of atoms
                rij = math.sqrt(sum((i - j) ** 2 for i, j in zip(ri, rj)))
                # g2 and its derivatives
                output_g, output_dg  = g2(eta, rij, rs, r_cutoff)
                g += output_g
                for i in range(3):
                    dg[i] += output_dg[i]
        # calculate g3
        case G_TYPE.G3:
            g, dg = 0, [0, 0, 0]
            for rj in cartesians:
                # if i = j
                if np.isclose(ri, rj).all():
                    # do not use in calculations
                    continue
                # distance between centers of atoms
                rij = math.sqrt(sum((i - j) ** 2 for i, j in zip(ri, rj)))
                # g3 and its derivatives
                output_g, output_dg = g3(k, rij, r_cutoff)
                g += output_g
                for i in range(3):
                    dg[i] += output_dg[i]
        # calculate g4
        case G_TYPE.G4:
            g, dg = 0, [0, 0, 0]
            for rj in cartesians:
                for rk in cartesians:
                    # if j = k, i = j or i = k
                    if np.isclose(rj, rk).all() or np.isclose(ri, rj).all() or np.isclose(ri, rk).all():
                        # do not use in calculations
                        continue 
                  
                    # distances between centers of atoms and their derivatives
                    # i and j
                    rij = math.sqrt(sum((i - j) ** 2 for i, j in zip(ri, rj)))
                    # i and k
                    rik = math.sqrt(sum((i - k) ** 2 for i, k in zip(ri, rk)))
                    # j and k
                    rjk = math.sqrt(sum((j - k) ** 2 for j, k in zip(rj, rk)))

                    # cosine of an angle between radius vectors and its derivatives
                    cos_v = (rij * rij + rik * rik - rjk * rjk) / (2 * rij * rik)
                    dcos_v = [0, 0, 0]
                    dcos_v[0] = 0.5 * (1 / rik + 1 / rij / rij * (rjk * rjk / rik - rik))
                    dcos_v[1] = 0.5 * (1 / rij + 1 / rik / rik * (rjk * rjk / rij - rij))
                    dcos_v[2] = -rjk / rij / rik
                    # g4 and its derivatives
                    output_g, output_dg = g4(eta, xi, lambda_, rij, rik, rjk, cos_v, dcos_v, r_cutoff)
                    g += output_g
                    for i in range(3):
                        dg[i] += output_dg[i]
        # calculate g5
        case G_TYPE.G5:
            g, dg = 0, [0, 0, 0]
            for rj in cartesians:
                for rk in cartesians:
                    # if j = k, i = j or i = k
                    if np.isclose(rj, rk).all() or np.isclose(ri, rj).all() or np.isclose(ri, rk).all():
                        # do not use in calculations
                        continue 

                    # distances between centers of atoms and their derivatives
                    # i and j
                    rij = math.sqrt(sum((i - j) ** 2 for i, j in zip(ri, rj)))
                    # i and k
                    rik = math.sqrt(sum((i - k) ** 2 for i, k in zip(ri, rk)))
                    # j and k
                    rjk = math.sqrt(sum((j - k) ** 2 for j, k in zip(rj, rk)))

                    # cosine of an angle between radius vectors and its derivatives
                    cos_v = (rij * rij + rik * rik - rjk * rjk) / (2 * rij * rik)
                    dcos_v = [0, 0, 0]
                    dcos_v[0] = 0.5 * (1 / rik + 1 / rij / rij * (rjk * rjk / rik - rik))
                    dcos_v[1] = 0.5 * (1 / rij + 1 / rik / rik * (rjk * rjk / rij - rij))
                    dcos_v[2] = -rjk / rij / rik
                    # g5 and its derivatives
                    output_g, output_dg = g5(eta, xi, lambda_, rij, rik, cos_v, dcos_v, r_cutoff)
                    g += output_g
                    for i in range(3):
                        dg[i] += output_dg[i]
        case _ :
            raise ValueError("Incorrect type of symmetric function")
    # return values
    result = PairG(g_type, g, dg)

    return result

def calculate_dg_by_params(g, ri, cartesians, g_type: int, r_cutoff: float,
                eta: float = None, rs: float = None, k: float = None,
                lambda_: float = None, xi: float = None):
        match g_type:
            # calculate g2
            case G_TYPE.G2:
                dg_params = [0, 0]
                for rj in cartesians:
                    # if i = j
                    if np.isclose(ri, rj).all():
                        # do not use in calculations
                        continue
                    # distance between centers of atoms
                    rij = math.sqrt(sum((i - j) ** 2 for i, j in zip(ri, rj)))
                    # 0 - partial derivative by eta,
                    # 1 - partial derivative by rs
                    dg_params[0] += dg2_eta(rij, rs, g)
                    dg_params[1] += dg2_rs(eta, rij, rs, r_cutoff, g)
            # calculate g3
            case G_TYPE.G3:
                dg_params =  0
                for rj in cartesians:
                    # if i = j
                    if np.isclose(ri, rj).all():
                        # do not use in calculations
                        continue
                    # distance between centers of atoms
                    rij = math.sqrt(sum((i - j) ** 2 for i, j in zip(ri, rj)))
                    # g3 has only 1 partial derivative by param     
                    dg_params += dg3_k(k, rij, r_cutoff)
            # calculate g4
            case G_TYPE.G4:
                dg_params = [0, 0, 0]
                for rj in cartesians:
                    for rk in cartesians:
                        # if j = k, i = j or i = k
                        if np.isclose(rj, rk).all() or np.isclose(ri, rj).all() or np.isclose(ri, rk).all():
                            # do not use in calculations
                            continue 
                    
                        # distances between centers of atoms and their derivatives
                        # i and j
                        rij = math.sqrt(sum((i - j) ** 2 for i, j in zip(ri, rj)))
                        rij_vec = [(i - j) for i, j in zip(ri, rj)]
                        # i and k
                        rik = math.sqrt(sum((i - k) ** 2 for i, k in zip(ri, rk)))
                        rik_vec = [(i - j) for i, j in zip(ri, rj)]
                        # j and k
                        rjk = math.sqrt(sum((j - k) ** 2 for j, k in zip(rj, rk)))

                        # cosine of an angle between radius vectors
                        cos_v = sum([rij_ * rik_ for rij_, rik_ in zip(rij_vec, rik_vec)]) / (rij * rik)
                        # cos_v = (rij * rij + rik * rik - rjk * rjk) / (2 * rij * rik)
                        
                        dg_params[0] += dg4_eta(g, rij, rik, rjk)
                        dg_params[1] += dg4_lambda(g, xi, cos_v, lambda_)
                        dg_params[2] += dg4_xi(g, xi, cos_v, lambda_)
            # calculate g5
            case G_TYPE.G5:
                dg_params = [0, 0, 0]
                for rj in cartesians:
                    for rk in cartesians:
                        # if j = k, i = j or i = k
                        if np.isclose(rj, rk).all() or np.isclose(ri, rj).all() or np.isclose(ri, rk).all():
                            # do not use in calculations
                            continue 

                        # distances between centers of atoms and their derivatives
                        # i and j
                        rij = math.sqrt(sum((i - j) ** 2 for i, j in zip(ri, rj)))
                        rij_vec = [(i - j) for i, j in zip(ri, rj)]
                        # i and k
                        rik = math.sqrt(sum((i - k) ** 2 for i, k in zip(ri, rk)))
                        rik_vec = [(i - j) for i, j in zip(ri, rj)]
                        # j and k
                        rjk = math.sqrt(sum((j - k) ** 2 for j, k in zip(rj, rk)))

                        # cosine of an angle between radius vectors
                        cos_v = sum([rij_ * rik_ for rij_, rik_ in zip(rij_vec, rik_vec)]) / (rij * rik)
                        # cos_v = (rij * rij + rik * rik - rjk * rjk) / (2 * rij * rik)
                        
                        dg_params[0] += dg5_eta(g, rij, rik)
                        dg_params[1] += dg5_lambda(g, xi, cos_v, lambda_)
                        dg_params[2] += dg5_xi(g, xi, cos_v, lambda_)
            case _ :
                raise ValueError("Incorrect type of symmetric function")
        return dg_params