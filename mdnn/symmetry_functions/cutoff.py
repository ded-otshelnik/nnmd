import math

def cutf(rij: float, rc:float):
    """Cutoff function

    Args:
        rij: distance between atoms i and j
        rc: the cutoff radius
    """
    return 0.5 * (math.cos(math.pi * rij / rc) + 1) if rij < rc else 0.

def dcutf(rij: float, rc:float):
    """Derivative of cutoff function

    Args:
        rij: distance between atoms i and j
        rc: the cutoff radius
    """
    return 0.5 * (-math.pi * math.sin(math.pi * rij / rc) / rc) if rij < rc else 0.