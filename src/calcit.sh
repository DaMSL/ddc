#!/bin/bash -l

#SBATCH
#SBATCH --job-name=parse
#SBATCH --time=12:0:0
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --output=parallel2_parse.out
#SBATCH --partition=lrgmem


python ldpara2.py


