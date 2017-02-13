# Data-Driven Control

DINSAC: A Simulation Ensemble Manager to Couple Data-Driven Control with In-Situ Analysis

DINSAC stands for Data-Drive IN-Situ Analysis and Control. It leverages a data feature lattice to link an analytical objective with a global exploration strategy by controlling task scheduling. It dispatches many simulation jobs, each of which executes with locally coupled, in-situ analysis operations, and aggregates output into a unified scientific exploration effort. Running in-transit within the HPC environment, a controller schedules subsequent tasks by selecting ideal input parameters derived from the lattice to meet a user-defined, global exploration objective.

Implementation is in python and designed for integration in the MARCC HPC computing center. For additional information, please contact me, Ben Ring, ring@cs.jhu.edu.
