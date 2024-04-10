#!/bin/bash -l

#PBS -N hw5
#PBS -A ALCFAITP
#PBS -l select=1:ncpus=0:ngpus=1
#PBS -l walltime=1:00:00
#PBS -q debug
#PBS -l filesystems=home:eagle
#PBS -o /home/prayaagkatta/ai-science-training-series/05_llm_part2/hw5.out 
#PBS -e /home/prayaagkatta/ai-science-training-series/05_llm_part2/hw5.err 

module load conda
conda activate groqflow
cd /home/prayaagkatta/ai-science-training-series/05_llm_part2
python /home/prayaagkatta/ai-science-training-series/05_llm_part2/LLM_part02.py