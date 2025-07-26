import torch

from enum import Enum


class SymmetryFunction(Enum):
    """
    Enumeration of implemented symmetry functions.
    Each symmetry function corresponds to a specific type of symmetry function.
    """

    G1 = 1
    G2 = 2
    G4 = 4
    G5 = 5


def _pairwise_filter(distances, r_cutoff):
    """
    Filter pairs based on cutoff distance.

    Args:
        distances: torch.Tensor of shape (batch_size, n_atoms, n_atoms)
        r_cutoff: float - cutoff radius

    Returns:
        torch.Tensor of shape (batch_size, n_atoms, n_atoms)
    """
    mask = (distances < r_cutoff) & (distances > 0)

    return mask


def _triplets_filter(distances: torch.Tensor, r_cutoff: float) -> torch.Tensor:
    """
    Filter triplets based on cutoff distance.

    Args:
        distances: torch.Tensor of shape (batch_size, n_atoms, n_atoms)
        r_cutoff: float - cutoff radius

    Returns:
        torch.Tensor of shape (batch_size, n_atoms, n_atoms)
    """
    mask = (distances < r_cutoff) & (distances > 0)

    # Ensure the triplet is valid
    mask = mask.unsqueeze(1) & mask.unsqueeze(2) & mask.unsqueeze(3)

    return mask


def calculate_distances(positions: torch.Tensor, cell: torch.Tensor):
    """
    Calculate pairwise distances and displacements with PBC.

    Args:
        positions: torch.Tensor of shape (n_moleculs, n_atoms, 3)
        cell: torch.Tensor of shape (3, 3)
        r_cutoff: float - cutoff radius

    Returns:
        torch.Tensor of shape (n_atoms, n_atoms)
    """
    # Create pairwise displacement matrix (num_atoms x num_atoms x 3)
    disp = positions.unsqueeze(2) - positions.unsqueeze(1)

    # Apply periodic boundary conditions if cell is provided
    # (i.e., if it has a non-zero volume)
    if cell.sum() > 0:
        # Convert displacements to fractional coordinates
        inv_cell = torch.linalg.inv(cell.T)
        frac_disp = torch.matmul(disp, inv_cell.T)

        # Apply the minimum image convention
        frac_disp -= torch.round(frac_disp)

        # Convert back to Cartesian coordinates
        cart_disp = torch.matmul(frac_disp, cell)

        # Compute distances
        distances = torch.linalg.norm(cart_disp, dim=-1)  # (B, N, N)
    else:
        # Compute distances in Cartesian coordinates
        distances = torch.linalg.norm(disp, dim=-1)

    return distances


def f_cutoff(r: torch.Tensor, cutoff: float) -> torch.Tensor:
    """
    Calculate the cutoff function for the pair of atoms.
    Function is vectorized and can be applied to a batch of pairs.

    Args:
        r: torch.Tensor of shape (n_atoms, n_atoms)
        cutoff: float - cutoff radius

    Returns:
        torch.Tensor of shape (n_atoms, n_atoms)
    """
    fc = 0.5 * (torch.cos(torch.pi * r / cutoff) + 1.0)
    fc = torch.where(r < cutoff, fc, torch.zeros_like(fc))
    return fc


def g1_function(r: torch.Tensor, cutoff: float) -> torch.Tensor:
    """
    Calculate G1 symmetry functions for a pair of atoms.
    Function is vectorized and can be applied to a batch of pairs.

    Args:
        r: torch.Tensor of shape (batch_size, n_atoms, n_atoms)
        cutoff: float - cutoff radius

    Returns:
        torch.Tensor of shape (batch_size, n_atoms)
    """
    # exclude self-interaction
    mask = _pairwise_filter(r, cutoff)

    G1 = f_cutoff(r, cutoff)
    G1 = torch.where(mask, G1, torch.zeros_like(G1))
    G1 = G1.sum(dim=-1)
    return G1


def g2_function(r: torch.Tensor, cutoff: float, eta: float, rs: float) -> torch.Tensor:
    """
    Calculate G2 symmetry functions for a pair of atoms.
    Function is vectorized and can be applied to a batch of pairs.

    Args:
        r: torch.Tensor of shape (batch_size, n_atoms, n_atoms)
        cutoff: float - cutoff radius
        eta: float - width of the Gaussian
        rs: float - center of the Gaussian

    Returns:
        torch.Tensor of shape (batch_size, n_atoms)
    """
    # exclude self-interaction
    mask = _pairwise_filter(r, cutoff)

    G2 = torch.exp(-eta * (r - rs) ** 2) * f_cutoff(r, cutoff)
    G2 = torch.where(mask, G2, torch.zeros_like(G2))
    G2 = G2.sum(dim=-1)
    return G2


def g4_function(
    distances: torch.Tensor, cutoff: float, eta: float, zeta: int, lambd: int
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
    fc = f_cutoff(distances, cutoff)  # Shape: (B, N, N)
    rij = distances.unsqueeze(3)  # Shape: (B, N, N, 1)
    rik = distances.unsqueeze(2)  # Shape: (B, N, 1, N)
    rjk = distances.unsqueeze(1)  # Shape: (B, 1, N, N)

    # Valid triplets distances are cutoffs and more than 0
    mask = _triplets_filter(distances, cutoff)  # Shape: (B, N, N, N)
    fc_triplet = (
        fc.unsqueeze(3) * fc.unsqueeze(2) * fc.unsqueeze(1)
    )  # Shape: (B, N, N, N)

    # Cosine of the angle θ_ijk (refactor, can causes nans)
    cos_theta = torch.where(
        (torch.abs(2 * rij * rik) >= 10e-4),
        (rij**2 + rik**2 - rjk**2) / (2 * rij * rik).clamp(min=1e-8),
        torch.zeros_like(mask),
    )

    exponent = 2 ** (1 - zeta) * torch.exp(
        -eta * (rij**2 + rik**2 + rjk**2)
    )  # Shape: (B, N, N, N)

    # For pow with float power data is converted to complex
    # to avoid nan in pow
    cosine_part = torch.where(
        torch.abs(cos_theta) > 10e-4,
        torch.pow(
            (1 + lambd * torch.abs(cos_theta)).to(dtype=torch.complex64), zeta
        ).real,
        torch.zeros_like(cos_theta),
    )

    # Angular symmetry function G^4
    G4 = cosine_part * exponent * fc_triplet
    G4 = torch.where(mask, G4, torch.zeros_like(G4))
    G4 = torch.sum(
        G4,
        dim=(-2, -1),
    )  # Shape: (B, N)

    return G4


def g5_function(
    distances: torch.Tensor, cutoff: float, eta: float, zeta: int, lambd: int
) -> torch.Tensor:
    """
    Calculate G5 symmetry functions for a triplet of atoms.

    Args:
        distances: torch.Tensor of shape (batch_size, n_atoms, n_atoms)
        cutoff: float - cutoff radius
        eta: float - width of the Gaussian
        zeta: float - exponent of the cosine term
        lambd: int - sign of the cosine term

    Returns:
        torch.Tensor of shape (batch_size, n_atoms)
    """
    fc = f_cutoff(distances, cutoff)  # Shape: (B, N, N)
    rij = distances.unsqueeze(3)  # Shape: (B, N, N, 1)
    rik = distances.unsqueeze(2)  # Shape: (B, N, 1, N)
    rjk = distances.unsqueeze(1)  # Shape: (B, 1, N, N)

    # Valid triplets distances are cutoffs and more than 0
    mask = _triplets_filter(distances, cutoff)  # Shape: (B, N, N, N)
    fc_triplet = (
        fc.unsqueeze(3) * fc.unsqueeze(2) * fc.unsqueeze(1)
    )  # Shape: (B, N, N, N)

    # Cosine of the angle θ_ijk (refactor, can causes nans)
    cos_theta = torch.where(
        (torch.abs(2 * rij * rik) >= 10e-4),
        (rij**2 + rik**2 - rjk**2) / (2 * rij * rik).clamp(min=1e-8),
        torch.zeros_like(mask),
    )

    exponent = 2 ** (1 - zeta) * torch.exp(
        -eta * (rij**2 + rik**2)
    )  # Shape: (B, N, N, N)

    # For pow with float power data is converted to complex
    # to avoid nan in pow
    cosine_part = torch.where(
        torch.abs(cos_theta) > 10e-4,
        torch.pow(
            (1 + lambd * torch.abs(cos_theta)).to(dtype=torch.complex64), zeta
        ).real,
        torch.zeros_like(cos_theta),
    )

    # Angular symmetry function G^5
    G5 = cosine_part * exponent * fc_triplet
    G5 = torch.where(mask, G5, torch.zeros_like(G5))
    G5 = torch.sum(G5, dim=(-2, -1))  # Shape: (B, N)

    return G5
