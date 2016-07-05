#!/bin/bash

#SBATCH
#SBATCH --job-name=osvc-red
#SBATCH --time=12:0:0
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --output=/home-1/bring4@jhu.edu/ddc/log/osvc-redis-%j.out
#SBATCH --workdir=/home-1/bring4@jhu.edu/ddc
#SBATCH --partition=parallel,shared

module load namd
module load redis

src/overlay.py --name=${1} redis start

