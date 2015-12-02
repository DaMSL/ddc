#!/bin/bash

srun -N 1 -c 1 --share -o debug_sim.out python src/simmd.py  &
sleep 5
srun -N 1 -c 1 --share -o debug_anl.out python src/anlmd.py  &
sleep 5
srun -N 1 -c 1 --share -o debug_ctl.out python src/ctlmd.py  &
