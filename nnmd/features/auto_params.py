import numpy as np

def params_for_G2(n_radial: int, r_cutoff: float):
    """
    Generate parameters for G2 symmetry functions.

    Args:
        n_radial: int - number of radial symmetry functions
        r_cutoff: float - cutoff radius

    Returns:
        list of tuples (r_cutoff, rs, eta)
    """
    params = []
    rs = r_cutoff / (n_radial - 1)
    eta = 5 * np.log(10) / (4 * (rs) ** 2)
    for i in range(n_radial):
        params.append((r_cutoff, rs * i, eta))
    return params

def params_for_G4(n_angular: int, r_cutoff: float):
    """
    Generate parameters for G4 symmetry functions.

    Args:
        n_angular: int - number of angular symmetry functions
        r_cutoff: float - cutoff radius

    Returns:
        list of tuples (r_cutoff, eta, lambd, xi)
    """
    params = []

    ind = 1
    eta = 2 * np.log(10) / ((r_cutoff) ** 2)
    for i in range(n_angular):
        xi = 1 + i * 30 / (n_angular - 2)
        for lambd in [1, -1]:
            params.append((r_cutoff, eta, lambd, xi))
            if ind >= n_angular:
                break
            ind += 1
    return params