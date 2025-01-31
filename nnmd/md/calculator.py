# -*- coding: utf-8 -*-
"""ASE calcualtor for interactions with NNMD models"""

import ase
import numpy as np
import torch
from ase.calculators.calculator import Calculator

class NNMD_calc(Calculator):
    def __init__(self, model: torch.nn.Module = None,
                 atoms: ase.Atoms = None, to_eV = 1.0,
                 properties = ['energy', 'forces'],
                 symm_funcs_data = None):
        """NNMD interface with ASE as a calculator

        Args:
            model: torch.nn.Module object
            atoms: optional, ase Atoms object
            properties: properties to calculate.
                the properties to calculate is fixed for each calculator,
                to avoid resetting the predictor during get_* calls.
        """
        Calculator.__init__(self)
        self.implemented_properties = properties
        self.model: torch.nn.Module = model
        self.pbc = False
        self.atoms = atoms
        self.to_eV = to_eV
        self.symm_funcs_data = symm_funcs_data

    def _generator(self):
        while True:
            if self._atoms_to_calc.pbc.any():
                data = {
                    'cell': self._atoms_to_calc.cell[np.newaxis, :, :],
                    'coord': self._atoms_to_calc.positions,
                    'ind_1': np.zeros([len(self._atoms_to_calc), 1]),
                    'elems': self._atoms_to_calc.numbers}
            else:
                data = {
                    'coord': self._atoms_to_calc.positions,
                    'ind_1': np.zeros([len(self._atoms_to_calc), 1]),
                    'elems': self._atoms_to_calc.numbers}
            yield data

    def _get_prediction(self, dtype = torch.float32):
        positions = torch.tensor(self._atoms_to_calc.positions, dtype = dtype)
        cell = torch.tensor(self._atoms_to_calc.cell.array, dtype = dtype)
        return self.model.predict(positions, cell, self.symm_funcs_data)

    def calculate(self, atoms = None, properties = None, system_changes = None):
        """Run a calculation. 

        The properties and system_changes are ignored here since we do
        not want to reset the predictor frequently. Whenever
        calculator is executed, the predictor is run. The calculate
        method will not be executed if atoms are not changed since
        last run (this should be handled by
        ase.calculator.Calculator).
        """
        if atoms is not None:
            self.atoms = atoms.copy()
        self._atoms_to_calc: ase.Atoms = self.atoms

        results = self._get_prediction()

        # the below conversion works for energy, forces, and stress,
        # it is assumed that the distance unit is angstrom
        results = {'energy': results[0].sum().item() * self.to_eV, 
                   'forces': results[1].cpu().detach().numpy() * self.to_eV}
        
        if 'stress' in results and self._atoms_to_calc.pbc.all():
            results['stress'] = results['stress'].flat[[0, 4, 8, 5, 2, 1]]
        self.results = results