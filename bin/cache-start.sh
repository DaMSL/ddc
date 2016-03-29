#!/bin/bash

#SBATCH
#SBATCH --job-name=osvc-cah
#SBATCH --time=4-0:00:0
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --output=/home-1/bring4@jhu.edu/ddc/ovs-cache.log
#SBATCH --workdir=/home-1/bring4@jhu.edu/ddc
#SBATCH --partition=lrgmem,shared,parallel,gpu,debug

module load namd
module load redis

src/overlay.py --name=${1} cache start

