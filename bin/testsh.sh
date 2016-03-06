#!/bin/bash -l

#SBATCH
#SBATCH --job-name=simmd
#SBATCH --time=0:10:0
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --output=0004.out
#SBATCH --workdir=~/ddc
module load namd
module load redis

python3 src/simmd.py simmd -w simmd:0001

