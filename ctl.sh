#!/bin/bash

#SBATCH --output=out/ctl-M-init.out

srun python3 piEstimator.py ctl -m
