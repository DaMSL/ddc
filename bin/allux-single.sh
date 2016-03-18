#!/bin/bash

#SBATCH
#SBATCH --job-name=testalx
#SBATCH --time=0:30:0
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --output=/home-1/bring4@jhu.edu/ddc/overlay-%j.out
#SBATCH --workdir=/home-1/bring4@jhu.edu/ddc
#SBATCH --partition=lrgmem

src/overlay.py --name=debug alluxio start
