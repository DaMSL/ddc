#!/bin/bash -l

#SBATCH
#SBATCH --job-name=calcop
#SBATCH --time=12:0:0
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --output=lattintrin_%j.out
#SBATCH --partition=shared,parallel


# python calcvar.py ${1} ${2} --seqnum 8

for cl in 1000 500 200 100 50 30 25 20 15 12 10 8 6 5
do
  python benchlatt.py ${1} ${cl} --seqnum 8
done


