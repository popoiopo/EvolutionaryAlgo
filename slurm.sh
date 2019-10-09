#!/bin/bash
#Sbatch -J GenAlgo
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -o outputfile
#SBATCH -e errfile
#SBATCH -t 48:00:00
srun python Generalist_EA_par_main.py
