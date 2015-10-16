#!/bin/bash

#SBATCH --output=out/sim-M-init.out

srun python3 src/piEstimator.py sim -m
