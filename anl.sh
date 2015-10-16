#!/bin/bash

#SBATCH --output=out/anl-M-init.out

srun python3 src/piEstimator.py anl -m
