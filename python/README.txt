Use this file to control the microcanonical ensemble parameters,
time parameters, and measurement parameters for a simulation of
the Lennard-Jones, classical simple liquid. The numbers mean the
following:

Microcanonical ensemble parameters:
1. Number of particles N.
2. Density rho, related to volume V.
3. Total energy per particle, related to total energy E.

Time parameters:
4. Number of blocks.
5. Number of time steps in block.
6. Time step length.

Measurement parameters:
7. Number of values g(r) is measured for.
8. Number of values C_vv(t_d) is measured for.
9. Number of momenta used to calculate C_rhorho(t_d).

Flags:
10. If flag set to 0, code does not store instantaneous positions and velocities.
    If flag set to 1, code stores instantaneous positions and velocities. 
