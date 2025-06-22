import numpy as np

from ase import Atoms
from ase.calculators.lj import LennardJones
from ase.io import Trajectory
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
import ase.units as units


# Create a simple system of atoms
traj_file = "input/lennard_jones_dimer.traj"
positions = [[0, 0, 0], [0.97, 0, 0]]
n_atoms = len(positions)
atoms = Atoms(f"H{n_atoms}", positions=positions)

# Set the initial velocities
MaxwellBoltzmannDistribution(atoms, temperature_K=300)

# Set the Lennard-Jones potential calculator
lj_calculator = LennardJones()
atoms.calc = lj_calculator

# timestep of the simulation (in fs)
timestep = 10e-5 * units.fs
dyn = VelocityVerlet(atoms, timestep)

with Trajectory(traj_file, "w", atoms) as traj:
    dyn.attach(traj.write, interval=1)
    dyn.run(35000)

plot = True
if plot:
    import matplotlib.pyplot as plt

    # Load the trajectory
    traj = Trajectory(traj_file)
    plot = [(frame.get_distance(0, 1), frame.get_potential_energy()) for frame in traj]

    # Separate distance and potential energy for plotting
    distances, energies = zip(*plot)

    # Plot the potential energy as a function of distance
    plt.plot(distances, energies, label="Potential Energy")
    plt.xlabel("Distance (Å)")
    plt.ylabel("Potential Energy (eV)")
    plt.title("Potential Energy vs Distance for Lennard-Jones Trimer")
    plt.grid()
    plt.legend()
    plt.show()

    # Load the trajectory
    traj = Trajectory(traj_file)
    plot = [(i, frame.get_distance(0, 1)) for i, frame in enumerate(traj)]

    # Separate distance and potential energy for plotting
    distances, energies = zip(*plot)

    # Plot the potential energy as a function of distance
    plt.plot(distances, energies)
    plt.xlabel("Distance (Å)")
    plt.ylabel("Time Step")
    plt.title("Distance 0 to 1, Lennard-Jones Trimer")
    plt.grid()
    plt.show()
