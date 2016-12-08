#!/bin/bash -l

#SBATCH
#SBATCH --job-name=dlat
#SBATCH --time=24:0:0
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24

python dolattice.py


