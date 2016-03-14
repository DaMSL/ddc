#!/bin/bash

#SBATCH
#SBATCH --job-name=osvc-alx
#SBATCH --time=4:00:0
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --output=/home-1/bring4@jhu.edu/ddc/overlay-%j.out
#SBATCH --workdir=/home-1/bring4@jhu.edu/ddc
#SBATCH --partition=lrgmem

#src/overlay.py --name=debug alluxio start
export ALLUXIO_HOME=$HOME/pkg/alluxio-1.0.0
export ALLUXIO_MASTER_ADDRESS=localhost
export ALLUXIO_UNDERFS_ADDRESS=$HOME/work/alluxio/debug
export ALLUXIO_RAM_FOLDER=/tmp/alluxio
export ALLUXIO_WORKER_MEMORY_SIZE=40GB
export DEFAULT_LIBEXEC_DIR=${ALLUXIO_HOME}/libexec

alluxio format
alluxio formatWorker
alluxio-start.sh master
master=$!
alluxio-start.sh worker Mount
#wait master

