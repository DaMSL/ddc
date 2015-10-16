#!/bin/bash

#SBATCH --output=out/anl-M-init.out

srun python3 piEstimator.py anl -m
