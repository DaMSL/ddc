#!/bin/bash

#SBATCH
#SBATCH --job-name=osvc-red
#SBATCH --time=6:00:0
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --output=/home-1/bring4@jhu.edu/ddc/overlay-%j.out
#SBATCH --workdir=/home-1/bring4@jhu.edu/ddc

module load namd
module load redis

src/overlay.py --name=${1} redis start

