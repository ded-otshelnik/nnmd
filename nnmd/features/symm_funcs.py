import functools
import torch

from enum import Enum

class SymmetryFunction(Enum):
    G2 = 2
    G4 = 4
    G5 = 5

def calculate_distances(positions: torch.Tensor, cell: torch.Tensor):
    """
    Calculate pairwise distances and displacements with PBC.

    Args:
        positions: torch.Tensor of shape (n_moleculs, n_atoms, 3)
        cell: torch.Tensor of shape (3, 3)
        r_cutoff: float - cutoff radius

    Returns:
        torch.Tensor of shape (n_atoms, n_atoms), torch.Tensor of shape (n_atoms, n_atoms, 3)
    """
    # Create pairwise displacement matrix (num_atoms x num_atoms x 3)
    disp = positions.unsqueeze(2) - positions.unsqueeze(1)  # (B, N, N, 3)

    # Convert displacements to fractional coordinates
    inv_cell = torch.linalg.inv(cell.T)
    frac_disp = torch.matmul(disp, inv_cell.T)

    # Apply the minimum image convention
    frac_disp -= torch.round(frac_disp)

    # Convert back to Cartesian coordinates
    cart_disp = torch.matmul(frac_disp, cell)

    # Compute distances
    distances = torch.linalg.norm(cart_disp, dim = -1)  # (B, N, N)
    return distances, cart_disp

def f_cutoff(r, cutoff):
    """
    Calculate the cutoff function for the pair of atoms.
    Function is vectorized and can be applied to a batch of pairs.

    Args:
        r: torch.Tensor of shape (n_atoms, n_atoms)
        cutoff: float - cutoff radius
    
    Returns:
        torch.Tensor of shape (n_atoms, n_atoms)
    """
    mask = r < cutoff
    fc = 0.5 * (torch.cos(torch.pi * r / cutoff) + 1.0)
    fc = torch.where(mask, fc, torch.zeros_like(fc))
    return fc

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
    fc = f_cutoff(r, cutoff)
    return (torch.exp(-eta * (r - rs) ** 2) * fc).sum(dim = 2)

def g4_function(distances, triplets, cutoff, eta, zeta, lambd) -> torch.Tensor:
    """
    Calculate G4 symmetry functions for a triplet of atoms.

    Args:
        distances: torch.Tensor of shape (batch_size, n_atoms, n_atoms)
        triplets: torch.Tensor of shape (batch_size, n_atoms, n_atoms, 3)
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
    rjk = torch.linalg.norm(triplets.unsqueeze(2) - triplets.unsqueeze(3), dim = -1)  # Shape: (B, N, N, N)

    # Valid triplets: Apply cutoff mask
    cutoff_mask = (rij < cutoff) & (rik < cutoff) & (rjk < cutoff)  # Shape: (B, N, N, N)

    # Cosine of the angle θ_ijk
    cos_theta = (rij ** 2 + rik ** 2 - rjk ** 2) / (2 * rij * rik).clamp(min = 1e-8)  # Shape: (B, N, N, N)

    # Smooth cutoff product
    fc_triplet = fc.unsqueeze(3) * fc.unsqueeze(2)  # Shape: (B, N, N, N)

    exponent = 2 ** (1 - zeta)  * torch.exp(-eta * (rij ** 2 + rik ** 2 + rjk ** 2))  # Shape: (B, N, N, N)

    # Angular symmetry function G^4
    G4 = torch.sum(fc_triplet * (1 + lambd * cos_theta) ** zeta * exponent * cutoff_mask, dim = (-2, -1))  # Shape: (B, N)
    return G4

def g5_function(distances, triplets, cutoff, eta, zeta, lambd) -> torch.Tensor:
    """
    Calculate G5 symmetry functions for a triplet of atoms.

    Args:
        distances: torch.Tensor of shape (batch_size, n_atoms, n_atoms)
        triplets: torch.Tensor of shape (batch_size, n_atoms, n_atoms, 3)
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
    rjk = torch.linalg.norm(triplets.unsqueeze(2) - triplets.unsqueeze(3), dim = -1)  # Shape: (B, N, N, N)

    # Valid triplets: Apply cutoff mask
    cutoff_mask = (rij < cutoff) & (rik < cutoff)  # Shape: (B, N, N, N)

    # Cosine of the angle θ_ijk
    cos_theta = (rij ** 2 + rik ** 2 - rjk ** 2) / (2 * rij * rik).clamp(min = 1e-8) 

    # Smooth cutoff product
    fc_triplet = (fc.unsqueeze(3) * fc.unsqueeze(2))  # Shape: (B, N, N, N)

    exponent = 2 ** (1 - zeta) * torch.exp(-eta * (rij ** 2 + rik ** 2))  # Shape: (B, N, N, N)

    # Angular symmetry function G^5
    G5 = torch.sum(fc_triplet * (1 + lambd * cos_theta) ** zeta * exponent * cutoff_mask, dim = (-2, -1))  # Shape: (B, N)
    return G5