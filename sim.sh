#!/bin/bash

#SBATCH --output=out/sim-M-init.out

srun python3 piEstimator.py sim -m
