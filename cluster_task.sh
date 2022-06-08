#!/bin/bash
# 
# name of the job for better recognizing it in the queue overview
#SBATCH --job-name=partiqlegan
# 
# define how many nodes we need
#SBATCH --nodes=1
#
# we only need on 1 cpu at a time
#SBATCH --ntasks=1
#
# expected duration of the job
#              hh:mm:ss
#SBATCH --time=24:00:00
# 
# partition the job will run on
#SBATCH --partition single
# 
# expected memory requirements
#SBATCH --mem=64000MB
#
# output path
#SBATCH --output="slurm_log/%j.out"

./venv/bin/python -m kedro run --pipeline $1

# Done
exit 0


