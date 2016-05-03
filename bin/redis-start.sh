#!/bin/bash

#SBATCH
#SBATCH --job-name=osvc-red
#SBATCH --time=6:0:0
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --output=/home-1/bring4@jhu.edu/ddc/osvc-redis-%j.out
#SBATCH --workdir=/home-1/bring4@jhu.edu/ddc
#SBATCH --partition=shared,parallel

module load namd
module load redis

src/overlay.py --name=${1} redis start

