#!/bin/bash

PART=shared
STARTNUM=0

if [[ "$#" -ne 1 ]]; then
    echo "Usage:  go.sh <epoch> --  ensure you have a file name <epoch>.json"
    exit 1
fi

module load redis
module load namd

srun -N 1 -n 1 -c 1 -p $PART -J sm-0000.00 -o ${1}-s.log python src/simanl.py -c ${1} &
