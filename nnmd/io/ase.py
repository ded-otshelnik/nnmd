from ase.io import Trajectory


def traj_parser(traj_file: str) -> tuple:
    """Extracts positions, energies, forces and velocities from ASE trajectory file
    with respect to each species.

    Args:
        traj_file (str): ASE trajectory file.

    Returns:
        tuple: number of atoms in each species, list of dictionaries with
        positions, forces, velocities and energies for each atomic structure,
        unit cell and pbc.
    """
    traj = Trajectory(traj_file)

    # get species in dataset
    species = set(traj[0].symbols)
    # get unit cell
    cell = traj[0].get_cell().array
    pbc = traj[0].pbc
    if isinstance(pbc, bool):
        pbc = [pbc, pbc, pbc]

    # get number of atoms in each species
    n_atoms = list(len(traj[0][traj[0].symbols == spec]) for spec in species)

    result = []

    # collect necessary data from each atomic structure and species
    # in a form of list of dictionaries
    for atoms in traj[1:]:
        data = {}
        for spec in species:
            mask = atoms.symbols == spec
            data[spec] = {"positions": atoms.positions[mask]}
        data["forces"] = atoms.get_forces()
        data["velocities"] = atoms.get_velocities()
        data["energy"] = atoms.get_potential_energy()
        result.append(data)

    return n_atoms, result, cell, pbc
