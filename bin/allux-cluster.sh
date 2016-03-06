#!/bin/bash

#SBATCH
#SBATCH --job-name=overlay
#SBATCH --time=0:05:0
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --output=/home-1/bring4@jhu.edu/ddc/overlay-%j.out
#SBATCH --workdir=/home-1/bring4@jhu.edu/ddc

NR_PROCS=$(($SLURM_NTASKS))

srun -N1 -n1 src/overlay.py --name=debug --role=MASTER alluxio start &
pids[0]=$!
srun -N1 -n1 src/overlay.py --name=debug --role=SLAVE alluxio start &
pids[1]=$!

for pid in ${pids[*]};
do
    wait ${pid} #Wait on all PIDs, this returns 0 if ANY process fails
done

