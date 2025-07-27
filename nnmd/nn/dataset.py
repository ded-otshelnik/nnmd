import torch
import numpy as np
from torch.utils.data import Dataset

from ..features import calculate_sf


class TrainAtomicDataset(Dataset):
    def __init__(
        self,
        sf_data: dict[str, torch.Tensor],
        energies: torch.Tensor,
        forces: torch.Tensor,
        symm_func_params: dict,
    ) -> None:

        self.energies: torch.Tensor = energies
        self.forces: torch.Tensor = forces

        self.sf_data: dict[str, torch.Tensor] = sf_data
        self.symm_func_params: dict = symm_func_params

        self.len = len(self.energies)

    def __getitem__(self, index):
        g = {spec: self.sf_data[spec][0][index] for spec in self.sf_data.keys()}
        dG = {spec: self.sf_data[spec][1][index] for spec in self.sf_data.keys()}
        return g, dG, self.energies[index], self.forces[index]

    def __len__(self):
        return self.len

    @classmethod
    def make_atomic_dataset(
        cls,
        dataset: dict,
        **kwargs,
    ) -> "TrainAtomicDataset":
        """Create atomic dataset with symmetric functions.

        Args:
            dataset (dict): dictionary with positions by species, \
            unit cell, forces and velocities.

        Returns:
            TrainAtomicDataset: dataset with symmetric functions.
        """
        device = torch.device("cuda")

        atoms = dataset["reference_data"]

        symm_func_params = dataset["symmetry_functions_set"]

        # convert data to torch tensors
        cell = torch.tensor(dataset["unit_cell"], dtype=torch.float32, device=device)
        pbc = torch.tensor(dataset["pbc"], dtype=torch.float32, device=device)

        # get cartesian coordinates for each species
        cartesians = {
            spec: torch.tensor(
                np.array([data[spec]["positions"] for data in atoms]),
                dtype=torch.float32,
                device=device,
            )
            for spec in atoms[0].keys()
            if spec not in ["forces", "energy", "velocities"]
        }

        energies = torch.tensor(
            np.array([data["energy"] for data in atoms]),
            dtype=torch.float32,
            device=device,
        )

        if energies.ndim == 1:
            energies = energies.unsqueeze(1)

        forces = torch.tensor(
            np.array([data["forces"] for data in atoms]),
            dtype=torch.float32,
            device=device,
        )

        # scale energies and forces for better error convergence
        emin, emax = energies.min(), energies.max()
        energies = (energies - emin) / (emax - emin)
        forces /= emax - emin

        sf_data = {}
        for spec in cartesians.keys():
            if "saved" in kwargs and kwargs["saved"]:
                g_spec = torch.load(f"g_{spec}.pt", map_location=device)
                dg_spec = torch.load(f"dg_{spec}.pt", map_location=device)
            else:
                g_spec, dg_spec = calculate_sf(
                    cartesians[spec],
                    cell,
                    pbc,
                    symm_func_params[spec],
                    disable_tqdm=kwargs.get("disable_tqdm", False),
                )
                g_spec = g_spec.to(device)
                dg_spec = dg_spec.to(device)

                torch.save(g_spec, f"g_{spec}.pt")
                torch.save(dg_spec, f"dg_{spec}.pt")

            if not g_spec.requires_grad:
                g_spec.requires_grad = True

            sf_data[spec] = (g_spec, dg_spec)

        print(
            "Atomic dataset created with the following species:",
            ", ".join(sf_data.keys()),
        )
        print(f"Total species in dataset: {len(sf_data)}")
        print(f"Shapes of g and dg tensors: {g_spec.shape}, {dg_spec.shape}")
        print(
            f"Shapes of energies and forces tensors: {energies.shape}, {forces.shape}"
        )

        return TrainAtomicDataset(sf_data, energies, forces, symm_func_params)
