#!/bin/bash

#SBATCH --output=out/ctl-M-init.out

srun python3 src/piEstimator.py ctl -m
