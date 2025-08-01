"""ASE calcualtor for interactions with NNMD models"""

from typing import List
import ase
from ase.calculators.calculator import Calculator

import torch


class NNMD_calc(Calculator):
    """NNMD interface with ASE as a calculator"""

    def __init__(
        self,
        model: torch.nn.Module = None,
        atoms: ase.Atoms = None,
        properties: List[str] = ["energy", "forces"],
        symm_funcs_data: dict = None,
    ):
        """
        Initialize the NNMD calculator.

        Args:
            model (nnmd.nn.Model): NNMD pre-trained model
            atoms (ase.Atoms): optional, ase Atoms object
            properties (list[str]): properties to calculate. \
            Defaults to ["energy", "forces"].
            symm_funcs_data (dict): symmetry functions data
        """
        Calculator.__init__(self)
        self.implemented_properties = properties
        self.model: torch.nn.Module = model
        self.atoms: ase.Atoms = atoms
        self.species = set(atoms.get_chemical_symbols()) if atoms else set()
        self.pbc = atoms.pbc
        self.symm_funcs_data = symm_funcs_data

    def _get_prediction(self):
        """Get forward prediction from the model."""
        positions = {}
        for spec in self.species:
            mask = self._atoms_to_calc.get_atomic_numbers() == spec
            positions[spec] = torch.tensor(
                self._atoms_to_calc.positions[mask],
                dtype=self.model.dtype,
                device=self.model.device,
            )

        cell = torch.tensor(
            self._atoms_to_calc.cell.array,
            dtype=self.model.dtype,
            device=self.model.device,
        )
        return self.model.predict(positions, cell, self.symm_funcs_data)

    def calculate(self, atoms=None, properties=None, system_changes=None):
        """Run a calculation.

        Args:
            atoms: ase Atoms object
            properties: properties to calculate (inherited from ASE calculator class)
            system_changes: changes in the system (inherited from ASE calculator class)
        """
        if atoms is not None:
            self.atoms = atoms.copy()
        self._atoms_to_calc: ase.Atoms = self.atoms

        results = self._get_prediction()

        # the below conversion works for energy and forces,
        # it is assumed that the distance unit is angstrom
        results = {
            "energy": results[0].sum().item(),
            "forces": results[1].cpu().detach().numpy(),
        }

        self.results = results
