#!/bin/bash

#SBATCH
#SBATCH --job-name=red
#SBATCH --time=48:00:0
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --output=/home-1/bring4@jhu.edu/ddc/redis.out
#SBATCH --workdir=/home-1/bring4@jhu.edu/ddc
#SBATCH --partition=lrgmem

module load namd
module load redis

redis-server ${1}_db.conf

