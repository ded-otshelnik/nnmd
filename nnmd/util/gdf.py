import math

def gdf(n_tot, g_arr, g_iter, sigma, N_s = 3):
    """Gaussian density function. Returns density values \
    for all atoms in sample

    Args:
        n_tot: number of atoms
        g_arr: training 
        g_iter: 2-D array, g values on all iterations
        N_s: dimention of G
        sigma: standard deviation
    """
    density = []
    for g in g_arr:
        g_density = 0
        for gi in g_iter:
            for gij in gi:
                g_density += math.exp(- abs(g - gij) ** 2 / (2 * sigma ** 2 * N_s) )
        density.append(g_density / n_tot)
    return density

def gdf_inv(n_tot, g_arr, g_iter, sigma, lambda_):
    # TODO: implement gdf_inv
    """Inverse gaussian density function. Returns density values \
    for all atoms in sample

    Args:
        n_tot: number of atoms
        g_arr: training 
        g_iter: 2-D array, g values on all iterations
        N_s: dimention of G
        sigma: standard deviation
        lam
    """
    density = []
    for g in g_arr:
        g_density = 0
        for gi in g_iter:
            for gij in gi:
                g_density += math.sqrt(lambda_ / (2 * math.pi * gij ** 3)) * math.exp(- lambda_ * (gij - sigma) ** 2 / (2 * sigma ** 2 * gij)  )
        density.append(g_density / n_tot)
    return density