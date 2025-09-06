#!/bin/bash -l

# Set SCC project
#$ -P ivc-ml
#$ -t 1-1  # Array job: 10 tasks
#$ -pe omp 4
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -l h_rt=16:00:00
#$ -N train
#$ -j y
#$ -o /projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/qsub_runs

module load miniconda
conda activate video_evalss
python train.py