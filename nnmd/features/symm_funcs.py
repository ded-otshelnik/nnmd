import torch

from enum import Enum

class SymmetryFunction(Enum):
    G2 = 2
    G4 = 4
    G5 = 5

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
    return torch.where(torch.abs(r - cutoff) >= 1e-8, 0.5 * (torch.cos(r * torch.pi / cutoff) + 1), torch.zeros_like(r))

def g2_function(r: torch.Tensor, cutoff: float, eta: float, rs: float) -> torch.Tensor:
        """
        Calculate G2 symmetry functions for a pair of atoms.
        Function is vectorized and can be applied to a batch of pairs.
        
        Args:
            r: torch.Tensor of shape (n_atoms, n_atoms)
            cutoff: float - cutoff radius
            eta: float - width of the Gaussian
            rs: float - center of the Gaussian

        Returns:
            torch.Tensor of shape (n_atoms, n_atoms)
        """
        fc = f_cutoff(r, cutoff)
        return (torch.exp(-eta * (r - rs) ** 2) * fc).sum(dim = 1)

def g4_function(rij: torch.Tensor, rjk: torch.Tensor, rik: torch.Tensor,
                cos_theta: torch.Tensor, cutoff: float, eta: float, zeta: float, lambd: float,
                mask: torch.Tensor) -> torch.Tensor:
        """
        Calculate G4 symmetry functions for a triplet of atoms.
        Function is vectorized and can be applied to a batch of triplets.
        Function does not filter out the diagonal elements of the input tensors.

        Args:
            rij: torch.Tensor of shape (n_atoms, n_atoms)
            rik: torch.Tensor of shape (n_atoms, n_atoms)
            rjk: torch.Tensor of shape (n_atoms, n_atoms)
            cos_theta: torch.Tensor of shape (n_atoms, n_atoms)
            cutoff: float - cutoff radius
            eta: float - width of the Gaussian
            zeta: float - exponent of the cosine term
            lambd: int - sign of the cosine term
            mask: torch.Tensor of shape (n_atoms, n_atoms) - mask for the diagonal elements

        Returns:
            torch.Tensor of shape (n_atoms, n_atoms)
        """
        fc_ij = f_cutoff(rij, cutoff)
        fc_ik = f_cutoff(rik, cutoff)
        fc_jk = f_cutoff(rjk, cutoff)
        fc = fc_ij * fc_ik * fc_jk

        term1 = (1 + lambd * cos_theta) ** zeta
        term2 = torch.exp(-eta * (rij  ** 2 + rik ** 2 + rjk ** 2))

        g = 2 ** (1 - zeta) * term1 * term2 * fc

        # apply the mask to the symmetry functions
        g[None, None, :] = g[None, None, :] * mask.float()

        # replace NaNs with zeros
        # for the case when the denominator in the cosine term is zero
        g[torch.isnan(g)] = 0.0

        # sum the symmetry functions part over the atoms
        g = g.sum(dim = -1).sum(dim = -1).squeeze()
        return g

def g5_function(rij: torch.Tensor, rik: torch.Tensor,
                cos_theta: torch.Tensor, cutoff: float, eta: float, zeta: float, lambd: float,
                mask: torch.Tensor) -> torch.Tensor:
        """
        Calculate G5 symmetry functions for a triplet of atoms.
        Function is vectorized and can be applied to a batch of triplets.
        Function does not filter out the diagonal elements of the input tensors.

        Args:
            rij: torch.Tensor of shape (n_atoms, n_atoms)
            rik: torch.Tensor of shape (n_atoms, n_atoms)
            cos_theta: torch.Tensor of shape (n_atoms, n_atoms)
            cutoff: float - cutoff radius
            eta: float - width of the Gaussian
            zeta: float - exponent of the cosine term
            lambd: int - sign of the cosine term
            mask: torch.Tensor of shape (n_atoms, n_atoms) - mask for the diagonal elements

        Returns:
            torch.Tensor of shape (n_atoms, n_atoms)
        """
        fc_ij = f_cutoff(rij, cutoff)
        fc_ik = f_cutoff(rik, cutoff)
        fc = fc_ij * fc_ik

        term1 = (1 + lambd * cos_theta) ** zeta
        term2 = torch.exp(-eta * (rij  ** 2 + rik ** 2))

        g = 2 ** (1 - zeta) * term1 * term2 * fc

        # apply the mask to the symmetry functions
        g[None, None, :] = g[None, None, :] * mask.float()

        # replace NaNs with zeros
        g[torch.isnan(g)] = 0.0

        # sum the symmetry functions part over the atoms
        g = g.sum(dim = -1).sum(dim = -1).squeeze()
        return g

def calculate_distances(cartesians: torch.Tensor) -> torch.Tensor:
    """
    Calculate pairwise distances between atoms in a molecule
    Args:
        cartesians: torch.Tensor of shape (n_atoms, 3)
    Returns:
        torch.Tensor of shape (n_atoms, n_atoms)
    """
    diff = cartesians.unsqueeze(1) - cartesians.unsqueeze(0)
    # add a small number to avoid nan in gradients
    distances = torch.sqrt(torch.sum(diff ** 2, dim = -1) + 10e-10)
    return distances

def calculate_cosines(rij, rik, rjk):
    """
    Calculate cosines of the angles between atom triplets in a molecule
    Args:
        rij: torch.Tensor of shape (n_atoms, n_atoms)
        rik: torch.Tensor of shape (n_atoms, n_atoms)
        rjk: torch.Tensor of shape (n_atoms, n_atoms)
    Returns:
        torch.Tensor of shape (n_atoms, n_atoms)
    """
    cosines = torch.where(torch.abs(2 * rij * rik) <= 10e-4,
                           (rij ** 2 + rik ** 2 - rjk ** 2) / (2 * rij * rik),
                            torch.zeros_like(rij))

    return cosines

def calculate_mask(cartesians):
    """
    Calculate a mask for the diagonal elements of the symmetry functions tensor
    Args:
        cartesians: torch.Tensor of shape (n_atoms, 3)
    Returns:
        torch.Tensor of shape (n_atoms, n_atoms)
    """
    n_atoms = cartesians.shape[0]
    mask = (torch.arange(n_atoms, device = cartesians.device).unsqueeze(1) != torch.arange(n_atoms, device = cartesians.device)).unsqueeze(2)
    mask = mask & mask.transpose(0, 1) & mask.transpose(1, 2)
    return mask