import copy
import numpy as np

def lennard_jones_gen():
    """Generate Lennard-Jones potentials and forces
    """

    def lennard_jones_component(points, dims):
        interatomic_vector = np.subtract(points[1], points[0])

        f_distance = lambda point: (sum([coord ** 2 for coord in point])) ** 0.5

        def calculate_energy(eps, teta, interatomic_vector):
            r = f_distance(interatomic_vector)
            energy = 4 * eps * (((teta / r) ** 12) - ((teta / r) ** 6))

            return energy

        def calculate_forces(eps, teta, interatomic_vector, h, points, dims):
            """Force = - derivative of U by r
            """

            force_deriv = lambda r: 4 * eps * ((12 * (teta / r) ** 13) - ( 6 * (teta / r) ** 7))
            forces = []
            for i in range(len(points)):
                distance_actual = f_distance(points[i])
                point_moved = np.copy(points[i])
                force_atom = []
                for j in range(dims):
                    point_moved[j] += h 
                    dr = np.vdot(points[i], point_moved) / distance_actual
                    temp =  - force_deriv(distance_actual) * dr
                    force_atom.append(- force_deriv(distance_actual) * dr)

                    point_moved[i] -= h 
                forces.append(force_atom)
            return forces

        # parameters related to LJ potential
        teta = 1.0
        eps = 1.0
        h = 0.1

        E = calculate_energy(eps, teta, interatomic_vector)
        F = calculate_forces(eps, teta, interatomic_vector, h, points, dims)

        return E, F, f_distance(interatomic_vector)

    points = [[1.6, 1.6], [2.4, 2.1]]
    cartesians = [copy.deepcopy(points)]
    e_new, f_new, vect_distance = lennard_jones_component(points, np.ndim(points))
    e_dft = [e_new]
    f_dft = [copy.deepcopy(f_new)]
    distances = [vect_distance]

    h = 0.001
    n_steps = 2 ** 6 - 1
    for _ in range(n_steps):
        # move 2nd atom by const distances
        points[0][0] -= h
        points[0][1] -= h
        points[1][0] += h
        points[1][1] += h
        cartesians.append(copy.deepcopy(points))

        e_new, f_new, vect_distance = lennard_jones_component(points, np.ndim(points))
        e_dft.append(e_new)
        f_dft.append(copy.deepcopy(f_new))

        distances.append(vect_distance)

    return cartesians, e_dft, f_dft, distances