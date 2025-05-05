import numpy as np


def calculate_params(r_cutoff, N_g1, N_g2, N_g3, N_g4, N_g5):
    """
    Calculate parameters for symmetry functions.

    ----
    Args:
        r_cutoff: float - cutoff radius
        N_g1: int - number of G1 functions
        N_g2: int - number of G2 functions
        N_g3: int - number of G3 functions
        N_g4: int - number of G4 functions
        N_g5: int - number of G5 functions

    ----
    Returns:
        list of parameters for symmetry functions
    """
    r_s = np.round(np.linspace(0, r_cutoff, N_g2), 6)
    a = r_cutoff / (N_g2 - 1)
    eta_rad = np.round(5 * np.log(10) / (4 * a**2), 6)

    kappa = 1

    eta_ang = np.round(2 * np.log(10) / (r_cutoff**2), 6)
    lambd = 1

    zeta_4 = np.round(
        [
            1 + i * 30 / (N_g4 - 2) if i > 0 else 1
            for i in range(int(np.ceil(N_g4 / 2 - 1)) + 1)
        ],
        6,
    )
    zeta_5 = np.round(
        [
            1 + i * 30 / (N_g5 - 2) if i > 0 else 1
            for i in range(int(np.ceil(N_g5 / 2 - 1)) + 1)
        ],
        6,
    )

    params = []
    for i in range(N_g1):
        params.append([r_cutoff])

    for i in range(N_g2):
        params.append([r_cutoff, r_s[i], eta_rad])

    for i in range(N_g3):
        params.append([r_cutoff, kappa])

    for i in range(N_g4):
        params.append([r_cutoff, eta_ang, zeta_4[i // 2], (-1) ** i * lambd])

    for i in range(N_g5):
        params.append([r_cutoff, eta_ang, zeta_5[i // 2], (-1) ** i * lambd])
    return params
