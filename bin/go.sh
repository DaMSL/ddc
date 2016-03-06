#!/bin/bash

PART=debug
STARTNUM=0

if [[ "$#" -ne 1 ]]; then
    echo "Usage:  go.sh <epoch> --  ensure you have a file name <epoch>.json"
    exit 1
fi

module load redis
module load namd

srun -N 1 -n 1 -c 1 -p $PART -J sm-0000.00 --share -o ${1}.sim.log python src/simmd.py -c ${1}.json &
sleep 5

srun -N 1 -n 1 -c 1 -p $PART -J am-0000.00 --share -o ${1}.anl.log python src/anl.py -c ${1}.json &
sleep 5

srun -N 1 -n 1 -c 1 -p $PART -J cm-0000.00 --share -o ${1}.ctl.log python src/ctl.py -c ${1}.json &