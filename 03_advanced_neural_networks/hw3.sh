#!/bin/bash -l

#PBS -N hw3
#PBS -A ALCFAITP
#PBS -l select=1
#PBS -l walltime=2:00:00
#PBS -q preemptable
#PBS -l filesystems=home:eagle
#PBS -o /home/prayaagkatta/ai-science-training-series/03_advanced_neural_networks/hw3.out 
#PBS -e /home/prayaagkatta/ai-science-training-series/03_advanced_neural_networks/hw3.err 

module load conda
conda activate groqflow
cd /home/prayaagkatta/ai-science-training-series/03_advanced_neural_networks
python /home/prayaagkatta/ai-science-training-series/03_advanced_neural_networks/01_conv_networks.py