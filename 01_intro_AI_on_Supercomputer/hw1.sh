#!/bin/bash -l

#PBS -N hw1
#PBS -A ALCFAITP
#PBS -l select=1
#PBS -l walltime=0:30:00
#PBS -q preemptable
#PBS -l filesystems=home:eagle
#PBS -o hw1.out 
#PBS -e hw1.err 

module load conda
conda activate groqflow
python /home/prayaagkatta/ai-science-training-series/01_intro_AI_on_Supercomputer/01_linear_regression_sgd.py