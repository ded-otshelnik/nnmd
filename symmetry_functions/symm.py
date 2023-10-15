import math

from .cutoff import cutf, dcutf

def g2(eta: float, rij: float, rc:float):
    """Second radial symmetric function

    Args:
        eta: width of Gaussian function
        rij: distance between atoms i and j
        rc: the center of Gaussian functions
    """
    # radial component
    g = math.exp(-eta * (rij - rc) ** 2)
    # derivative of g
    dg = g * (-2 * eta * (rij - rc) * cutf(rij, rc) + dcutf(rij, rc))
    # use the cutoff to g
    g *= cutf(rij, rc)

    return g, dg

def g4(eta, xi, lambda_, rij, rc):
    # TODO: implement(?)
    pass

def g5(eta, xi, lambda_, rij, rc):
    # TODO: implement(?)
    pass