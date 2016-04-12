#!/bin/bash

#SBATCH
#SBATCH --job-name=osvc-red
#SBATCH --time=0:5:0
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/home-1/bring4@jhu.edu/ddc/osvc-redis-%j.out
#SBATCH --workdir=/home-1/bring4@jhu.edu/ddc
#SBATCH --partition=debug,shared

module load namd
module load redis

src/overlay.py --name=${1} redis start

