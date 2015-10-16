#!/bin/bash

echo 'Initiating Simulation Macrothread.....'
sbatch sim.sh

sleep 10
echo
echo 'Initiating Analysis Macrothread....'
sbatch anl.sh

sleep 10
echo
echo 'Initiating Control Macrothread....'
sbatch ctl.sh
