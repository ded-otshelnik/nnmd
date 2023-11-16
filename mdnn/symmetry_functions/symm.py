import math

from .cutoff import cutf, dcutf

def g1(rij: float, rc: float):
    """Radial symmetric function. Returns g and its derivative value

    Args:
        rij: distance between atoms i and j
        rc: cutoff radius
    """
    return cutf(rij, rc), [dcutf(rij, rc), 0, 0]

def g2(eta: float, rij: float, rs: float, rc: float):
    """Radial symmetric function. Returns g and its derivative value

    Args:
        eta: width of Gaussian function
        rij: distance between atoms i and j
        rs: shift of Gaussian function
        rc: cutoff radius
    """
    # radial component
    g = math.exp(-eta * (rij - rs) ** 2)
    # derivative of g
    dg = [0, 0, 0]
    dg[0] = g * (-2 * eta * (rij - rs) * cutf(rij, rc) + dcutf(rij, rc))
    # use the cutoff to g
    g *= cutf(rij, rc)

    return g, dg

def g3(k: float, rij: float, rc: float):
    """Radial symmetric function. Returns g and its derivative value

    Args:
        k: adjustment of cosine function
        rij: distance between atoms i and j
        rc: cutoff radius
    """
    # radial component
    g = math.cos(k * rij)
    # derivatives of g
    dg = [0, 0, 0]
    dg[0] =  - k * math.sin(k * rij) * cutf(rij, rc) + g * dcutf(rij, rc)
    # use the cutoff to g
    g *= cutf(rij, rc)

    return g, dg

def g4(eta: float, xi: float, lambda_: float, rij: float, rik: float, rjk: float, cos_v: float, dcos_v: list, rc: float):
    """Angular symmetric function. Returns g and its derivatives values

    Args:
        eta: width of Gaussian function  
        xi: 1st parameter of symmetric function
        lambda: 2nd parameter of symmetric function
        rij: distance between atoms i and j
        rik: distance between atoms i and k
        rjk: distance between atoms j and k
        cos_v: cosine of angle between rij and rik
        dcos_v: derivatives of cosine func
        rc: cutoff radius
    """
    # exponent part of g formula
    expv = math.exp(-eta * (rij ** 2 + rik ** 2 + rjk ** 2))
    # cosine part of formula
    cosv = 1 + lambda_ * cos_v
    powcos = 0 if abs(cosv) < 10e-4 else math.pow(cosv, xi) 

    # angular g component
    g = math.pow(2, 1 - xi) * powcos * expv * \
        cutf(rij, rc) * cutf(rik, rc) * cutf(rjk, rc)
    
    # derivatives of g
    dg = [0, 0, 0]
    # derivative by rij
    dg[0] = expv * powcos * cutf(rik, rc) * cutf(rjk, rc) * \
            (( -2 * eta * rij * cutf(rij, rc) + dcutf(rij, rc)) * cosv) + \
            xi * lambda_ * cutf(rij, rc) * \
            dcos_v[0]
    # derivative by rik
    dg[1] = expv * powcos * cutf(rij, rc) * cutf(rjk, rc) * \
            (( -2 * eta * rik * cutf(rik, rc)) * cosv) + \
            xi * lambda_ * cutf(rik, rc) * \
            dcos_v[1]
    # derivative by rjk
    dg[2] = expv * powcos * cutf(rij, rc) * cutf(rik, rc) * \
            (( -2 * eta * rjk * cutf(rjk, rc)) * cosv) + \
            xi * lambda_ * cutf(rjk, rc) * \
            dcos_v[2]
    
    return g, dg



def g5(eta: float, xi: float, lambda_: float, rij: float, rik: float, cos_v: float, dcos_v: list, rc: float):
    """Second angular symmetric function. Returns g and its derivatives values

    Args:
        eta: width of Gaussian function  
        xi: 1st parameter of symmetric function
        lambda: 2nd parameter of symmetric function
        rij: distance between atoms i and j
        rik: distance between atoms i and k
        cos_v: cosine of angle between rij and rik
        dcos_v: derivatives of cosine func
        rc: cutoff radius
    """
    # exponent part of g formula
    expv = math.exp(-eta * (rij ** 2 + rik ** 2))
    # cosine part of g formula
    cosv = 1 + lambda_ * cos_v
    powcos = 0 if abs(cosv) < 10e-4 else math.pow(cosv, xi)  

    # angular component
    g = math.pow(2, 1 - xi) * powcos * expv * \
        cutf(rij, rc) * cutf(rik, rc)
    
    # derivatives of g
    dg = [0, 0, 0]
    # derivative by rij
    dg[0] = expv * powcos * cutf(rik, rc) * \
            (( -2 * eta * rij * cutf(rij, rc) + dcutf(rij, rc)) * cosv) + \
            xi * lambda_ * cutf(rij, rc) * \
            dcos_v[0]
    # derivative by rik
    dg[1] = expv * powcos * cutf(rij, rc) * \
            (( -2 * eta * rik * cutf(rik, rc)) * cosv) + \
            xi * lambda_ * cutf(rik, rc) * \
            dcos_v[1]
    # derivative by rjk
    dg[2] = expv * powcos * cutf(rij, rc) * cutf(rik, rc) * \
            -2 * eta * cosv + \
            xi * lambda_ * dcos_v[2]
    
    return g, dg

# partial derivatives of descriptors by parameters

def dg2_eta(rij: float, rs: float, g: float):
    """Partial derivative of G2 by eta parameter

    Args:
        rij: distance between atoms i and j
        rs: shift of Gaussian function
        g: scaled G2 value
    """
    return -(rij - rs) ** 2 * g

def dg2_rs(eta: float, rij: float, rs: float, rc: float, g: float):
    """Partial derivative of G2 by rs parameter

    Args:
        eta: width of Gaussian function
        rij: distance between atoms i and j
        rs: shift of Gaussian function
        rc: cutoff radius
        g: scaled G2 value
    """
    return (-eta * 2 * (rij - rs)) * g


def dg3_k(k: float, rij: float, rc: float):
    """Partial derivative of G3 by k parameter

    Args:
        k: adjustment of cosine function
        rij: distance between atoms i and j
        rc: cutoff radius
    """
    return - math.sin(k * rij) * rij * cutf(rij, rc)

def dg4_eta(g: float, rij: float, rik: float, rjk: float):
    """Partial derivative of G4 by eta parameter

    Args:
        g: actual G4 value of atom
        rij: distance between atoms i and j
        rik:  distance between atoms i and k
        rjk:  distance between atoms j and k
    """
    return - (rij ** 2 + rik ** 2 + rjk ** 2) * g

def dg4_lambda(g: float, xi: float, cos_v: float, lambda_: float):
    """Partial derivative of G4 by lambda parameter

    Args:
        g: actual G4 value
        xi: 1st parameter of symmetric function G4
        cos_v: cosine of angle between rij and rik
        lambda_: 1st parameter of symmetric function G4
    """
    return xi * g / ((1 / cos_v) + lambda_) if cos_v > 10e-2 else 0

def dg4_xi(g: float, xi: float, cos_v: float, lambda_: float):
    """Partial derivative of G4 by xi parameter

    Args:
        g: actual G4 value
        xi: 1st parameter of symmetric function G4
        cos_v: cosine of angle between rij and rik
        lambda_: 1st parameter of symmetric function G4
    """
    return - g * math.log(2) + g * math.log(1 + lambda_ * cos_v)

def dg5_eta(g: float, rij: float, rik: float):
    """Partial derivative of G5 by eta parameter

    Args:
        g: actual G5 value of atom
        rij: distance between atoms i and j
        rik:  distance between atoms i and k
    """
    return - (rij ** 2 + rik ** 2) * g
def dg5_lambda(g: float, xi: float, cos_v: float, lambda_: float):
    """Partial derivative of G4 by lambda parameter

    Args:
        g: actual G5 value
        xi: 1st parameter of symmetric function G5
        cos_v: cosine of angle between rij and rik
        lambda_: 1st parameter of symmetric function G5
    """
    return xi * g / ((1 / cos_v) + lambda_) if cos_v > 10e-2 else 0

def dg5_xi(g: float, xi: float, cos_v: float, lambda_: float):
    """Partial derivative of G5 by xi parameter

    Args:
        g: actual G5 value
        xi: 1st parameter of symmetric function G5
        cos_v: cosine of angle between rij and rik
        lambda_: 1st parameter of symmetric function G5
    """
    return - g * math.log(2) + g * math.log(1 + lambda_ * cos_v)