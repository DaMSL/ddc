#!/bin/bash


if [[ "$#" -ne 1 ]]; then
    echo "Usage:  go.sh <epoch> --  ensure you have a file name <epoch>.conf"
    exit 1
fi

module load redis
module load namd

srun -N 1 -c 1 --share -o log/${1}_sim.out python src/simmd.py -c ${1}.conf &
sleep 5
srun -N 1 -c 1 --share -o log/${1}_anl.out python src/anlmd.py -c ${1}.conf &
sleep 5
srun -N 1 -c 1 --share -o log/${1}_ctl.out python src/ctlmd.py -c ${1}.conf &
