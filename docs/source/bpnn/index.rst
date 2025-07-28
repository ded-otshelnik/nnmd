Behler-Parrinello Neural Network
################################

**Behler-Parrinello Neural Network (BPNN)** is a type of neural network specifically designed
to approximate potential energy surfaces in molecular dynamics simulations :cite:`bpnn, behler2021`.

It uses special architecture called **High-Dimensional Neural Network (HDNN)** that contains 2 main parts:

1. **Symmetry Functions** - to encode the local environment of atoms in a system. Must be unique descriptor of the local environment and invariant to translations, rotations, and permutations of atoms.
These functions are used as input to the network.
2. Subnets - **Atomic Networks (AtomicNN)** - for each sort of atoms (chemical species) in the system. This nets are
multilayer perceptrons (MLP) that take symmetry functions as input and output the energy contribution of each atom.

.. figure:: ../../images/bpnn.png
   :width: 100%
   :align: center
   :alt: Plot of schematic Behler-Parrinello-Neural-Network architecture.

Behler-Parrinello Neural Network can be used to calculate various atomic characteristics, such as:

- Atomic energy (sum of atomic contributions),
- Atomic forces (gradients of atomic energy with respect to atomic positions by chain rule),
- Atomic stress (gradients of atomic energy with respect to atomic positions and box vectors by chain rule),

.. note::
   The BPNN architecture is a generalization of the Behler-Parrinello approach to neural networks,
   which allows for the representation of complex potential energy surfaces in high-dimensional spaces.
   Approach also makes possible to use different architectures, e.g,  graph convolutional neural networks (GCNN).

Atomic-Centered Symmetry Functions
**********************************

For the BPNN to work, it is necessary to encode the local environment of atoms in a system.
This is done using **Atomic-Centered Symmetry Functions (ACSF)** :cite:`acsf` that is used as an input to the network.

ACSFs are designed to be invariant to translations, rotations, and permutations of atoms in the system
so it encodes uniquely the local environment of each atom. There are two types of ACSFs:

1. **Radial Functions** - depend only on the distance between atoms.

.. math:: G_1^i = \sum_j^{N_{\text{atoms}}} f_c (R_{ij}),
.. math:: G_2^i = \sum_j^{N_{\text{atoms}}} e^{-\eta (R_{ij} - R_s)^2} f_c (R_{ij}),
.. math:: G_3^i = \sum_j^{N_{\text{atoms}}} \kappa \cos(R_{ij}) f_c (R_{ij}),

where :math:`f_c` is a cutoff function that can be defined as:

.. math:: f_c (R_{ij}) = \begin{cases}
    \frac{1}{2} \left( 1 + \cos\left(\frac{\pi R_{ij}}{R_c}\right) \right), & \text{for} R_{ij} < R_c \\
    0, & \text{for} R_{ij} \geq R_c \\
    \end{cases}

2. **Angular Functions** - depend on the angles between bonds.

.. math::  G_4^i = 2^{(1-\zeta)} \sum_{j,k \ne j} \left( 1 + \lambda \cos(\theta_{ijk}) \right)^{\zeta} \\
           e^{-\eta (R_{ij}^2 + R_{ik}^2  + R_{jk}^2)} f_c (R_{ij}) f_c (R_{ik}) f_c (R_{jk}),
.. math::  G_5^i = 2^{(1-\zeta)} \sum_{j,k \ne j} \left( 1 + \lambda \cos(\theta_{ijk}) \right)^{\zeta} \\
           e^{-\eta (R_{ij}^2 + R_{ik}^2)} f_c (R_{ij}) f_c (R_{ik}),

where :math:`\theta_{ijk}` is the angle between the bonds :math:`ij` and :math:`ik` and can be defined as:

.. math:: \theta_{ijk} = \arccos\left(\frac{R_{ij}^2 + R_{ik}^2 - R_{jk}^2}{2 R_{ij} R_{ik}}\right)

Set of symmetry functions defines the input of BPNN. Usually about 50 symmetry functions for each species are used.

Choice of ACSF parameters 
*************************

The choice of ACSF parameters is crucial for the performance of the BPNN. The parameters include:

- **Cutoff radius** :math:`R_c` - the distance beyond which the symmetry functions are zero.
- :math:`\eta` - controls the width of the Gaussian functions in the symmetry functions.
- :math:`R_s` - the distance at which the Gaussian functions are centered in :math:`G_2`.
- :math:`\kappa` - controls the amplitude of the cosine functions in :math:`G_3`.
- :math:`\zeta` - controls the smoothness of the angular functions.
- :math:`\lambda` - controls the sign of the angle to the symmetry functions. Usually set to 1 or -1.
