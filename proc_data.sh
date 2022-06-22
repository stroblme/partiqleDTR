#!/bin/bash
# 
# name of the job for better recognizing it in the queue overview
#SBATCH --job-name=partiqlegan-datagen
# 
# define how many nodes we need
#SBATCH --nodes=1
#
# we only need on 1 cpu at a time
#SBATCH --ntasks=1
#
# expected duration of the job
#              hh:mm:ss
#SBATCH --time=00:30:00
# 
# partition the job will run on
#SBATCH --partition single
# 
# expected memory requirements
#SBATCH --mem=16000MB
#
# output path
#SBATCH --output="logs/slurm/slurm-%j.out"

./venv/bin/python -m kedro run --pipeline data_processing_artificial_pipeline

# Done
exit 0


