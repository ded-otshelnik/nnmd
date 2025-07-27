import functools

import torch
from tqdm import tqdm

from .symm_funcs import (
    calculate_distances,
    g1_function,
    g2_function,
    g4_function,
    g5_function,
    SymmetryFunction,
)


def _set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_sf(
    cartesians: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    symm_funcs_data: dict,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate symmetry functions for a batch of atomic systems.

    Args:
        cartesians: torch.Tensor of shape (n_batch, n_atoms, 3)
        cell: torch.Tensor of shape (3, 3) - the three vectors defining unit cell
        pbc: torch.Tensor of shape (3,) - periodic boundary conditions
        symm_funcs_data: dict - dictionary with symmetry functions data

    Returns:
        torch.Tensor of shape (n_molecules, n_atoms, n_symm_funcs)
    """
    torch.autograd.set_detect_anomaly(True)

    def closure(
        cart: torch.Tensor, cell: torch.Tensor, pbc: torch.Tensor, symm_funcs_data: dict
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate symmetry functions for a single molecule.


        Args:
            cart: torch.Tensor of shape (n_atoms, 3) - cartesian coordinates of atoms
            cell: torch.Tensor of shape (3, 3) - the three vectors defining unit cell
            pbc: torch.Tensor of shape (3,) - periodic boundary conditions
            symm_funcs_data: dict - dictionary with symmetry functions data

        Returns:
            torch.Tensor of shape (n_atoms, n_symm_funcs)
        """
        distances = calculate_distances(cart, cell, pbc)

        g_struct = []
        dg_struct = []
        for g_func, g_params in zip(
            symm_funcs_data["features"], symm_funcs_data["params"]
        ):
            match SymmetryFunction(g_func):
                case SymmetryFunction.G1:
                    # Parameters: r_cutoff
                    g_values = g1_function(distances, *g_params)
                case SymmetryFunction.G2:
                    # Parameters: r_cutoff, eta, rs
                    g_values = g2_function(distances, *g_params)
                case SymmetryFunction.G4:
                    # Parameters: r_cutoff, eta, zeta, lambda
                    g_values = g4_function(distances, *g_params)
                case SymmetryFunction.G5:
                    # Parameters: r_cutoff, eta, zeta, lambda
                    g_values = g5_function(distances, *g_params)
                case _:
                    raise ValueError(f"Unknown symmetry function number: {g_func}")

            # Gradients of the symmetry functions
            # will be used for forces calculation
            dg_values = torch.autograd.grad(
                g_values,
                cart,
                grad_outputs=torch.ones_like(g_values),
                create_graph=True,
            )[0]

            g_struct.append(g_values.detach())
            dg_struct.append(dg_values.detach())

            torch.cuda.empty_cache()

        g_struct = torch.stack(g_struct, dim=-1)
        dg_struct = torch.stack(dg_struct, dim=-1)

        # Normalize the symmetry functions
        g_struct = g_struct / torch.norm(g_struct, p=2, dim=-1, keepdim=True)

        # Permute to the shape (n_batch, n_atoms, n_symm_funcs, 3)
        dg_struct = dg_struct.permute(0, 1, 3, 2)

        if torch.isnan(g_struct).sum().item() != 0:
            raise ValueError("NaN values in the symmetry functions")
        if torch.isnan(dg_struct).sum().item() != 0:
            raise ValueError("NaN values in the gradients of the symmetry functions")

        return g_struct, dg_struct

    op = functools.partial(closure, cell=cell, pbc=pbc, symm_funcs_data=symm_funcs_data)

    _set_seed(42)

    cartesians.requires_grad = True
    cartesians_chunks = torch.chunk(cartesians, len(cartesians) // 2 + 1, dim=0)

    r = [
        op(cart)
        for cart in tqdm(
            cartesians_chunks,
            desc="Calculating symmetry functions",
            total=len(cartesians),
            disable=kwargs.get("disable_tqdm", False),
        )
    ]
    g, dg = zip(*r)

    g = torch.cat(g, dim=0)
    dg = torch.cat(dg, dim=0)

    cartesians.requires_grad = False

    return g, dg
