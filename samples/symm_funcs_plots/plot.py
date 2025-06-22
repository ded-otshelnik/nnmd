import torch
import matplotlib.pyplot as plt

import nnmd
from nnmd.features import f_cutoff


def g4_function(
    distances: torch.Tensor,
    cos_theta: torch.Tensor,
    cutoff: float,
    eta: float,
    zeta: float,
    lambd: int,
) -> torch.Tensor:
    """
    Calculate G4 symmetry functions for a triplet of atoms.

    Args:
        distances: torch.Tensor of shape (batch_size, n_atoms, n_atoms)
        cutoff: float - cutoff radius
        eta: float - width of the Gaussian
        zeta: float - exponent of the cosine term
        lambd: int - sign of the cosine term

    Returns:
        torch.Tensor of shape (batch_size, n_atoms)
    """
    # Smooth cutoff function for pairwise distances
    fc = f_cutoff(distances, cutoff)  # Shape: (B, N, N)

    # Expand distances and displacements to compute triplets
    rij = distances.unsqueeze(3)  # Shape: (B, N, N, 1)
    rik = distances.unsqueeze(2)  # Shape: (B, N, 1, N)
    rjk = distances.unsqueeze(1)  # Shape: (B, 1, N, N)

    # Valid triplets: Apply cutoff mask
    cutoff_mask = (
        (rij < cutoff) & (rik < cutoff) & (rjk < cutoff)
    )  # Shape: (B, N, N, N)

    # Smooth cutoff product
    fc_triplet = fc.unsqueeze(3) * fc.unsqueeze(2)  # Shape: (B, N, N, N)

    exponent = 2 ** (1 - zeta) * torch.exp(
        -eta * (rij**2 + rik**2 + rjk**2)
    )  # Shape: (B, N, N, N)

    # Angular symmetry function G^4
    G4 = torch.sum(
        fc_triplet * (1 + lambd * cos_theta) ** zeta * exponent * cutoff_mask,
        dim=(-2, -1),
    )  # Shape: (B, N)
    return G4


params = {"font.size": 16}
plt.rcParams.update(params)

a, b = 0, 14
n = 1000
rij = torch.linspace(a, b, n).unsqueeze(1).unsqueeze(2)
angles = torch.linspace(0, 360, 1000) * torch.pi / torch.full((1000,), 180)


def test_g2_eta(rij, fname):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(a, b)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(r"$R_{ij} (Bohr)$")
    ax.set_ylabel(r"$G^2$")
    rc = 11.3

    for eta in [5, 1, 0.4, 0.2, 0.1, 0.06, 0.03, 0.01]:
        g = nnmd.features.g2_function(rij, rc, eta, 0)
        ax.plot(rij.squeeze(1), g, label=r"$\eta$ = {}".format(eta))

    ax.legend()
    plt.savefig(fname)


def test_g2_rs(rij, fname):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(a, b)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(r"$R_{ij} (Bohr)$")
    ax.set_ylabel(r"$G^2$")
    rc = 11.3
    eta = 3.0

    for rs in range(2, 10):
        g = nnmd.features.g2_function(rij, rc, eta, rs)
        ax.plot(rij.squeeze(1), g, label=r"$R_s$ = {}".format(rs))

    ax.legend()
    ax.grid()
    fig.tight_layout()
    plt.savefig(fname)


def test_g4(angles, lambda_, fname):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(angles.min(), angles.max())
    ax.set_xlabel(r"$\theta$, radians")
    ax.set_ylabel(r"$G_4, \lambda$ = {}".format(lambda_))
    rc = 11.3
    eta = 1

    coords = torch.tensor([[[0.75, 0.0, 0.0], [0.0, 0.75, 0.0], [0.0, 0.0, 0.75]]])
    distances = torch.cdist(coords, coords)
    for zeta in [1, 2, 4, 16, 64]:
        g = []
        for cos_v in torch.cos(angles):
            g_ = g4_function(distances, cos_v, rc, eta, zeta, lambda_).squeeze()
            g.append(g_[0])

        ax.plot(angles, g, label=r"$\zeta$ = {}".format(zeta))

    ax.legend()
    ax.grid()
    fig.tight_layout()
    plt.savefig(fname)


test_g2_eta(rij, "g2_eta.jpg")

test_g2_rs(rij, "g2_rs.jpg")

# with positive lambda parameter
test_g4(angles, +1, "g4_pos.jpg")

# with negative lambda parameter
test_g4(angles, -1, "g4_neg.jpg")
