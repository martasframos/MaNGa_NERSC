#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=00:05:00
#SBATCH --constraint=haswell
#SBATCH -N 1
#SBATCH -J Galaxies_fit
#SBATCH -o ofiles/Fit_%j.o
#SBATCH -L SCRATCH,project

now=$(date +"%T")
echo "start time ... $now"

i0=0
i1=29
niter=2500
n_cpu=32

export OMP_NUM_THREADS=1

python -W ignore /global/cscratch1/sd/marta_r/2021-10-10/_Multiprocessing/Fiting_programs/Working_code_test.py \
    $i0 $i1 $niter $n_cpu 

now=$(date +"%T")
echo "end time ... $now"