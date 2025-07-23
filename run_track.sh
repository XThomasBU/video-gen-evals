#!/bin/bash -l

# Set SCC project
#$ -P ivc-ml
#$ -t 1-1  # Array job specification
#$ -pe omp 4 # Request 4 CPU cores
#$ -l gpus=1 # Request 1 GPU
#$ -l gpu_c=8.0
#$ -l h_rt=42:00:00
#$ -N ucf101_track_wan21
#$ -j y # Merge standard output and error
#$ -o /projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/qsub_runs

module load miniconda
conda activate video_evals

module load ffmpeg

python -m src.human_mesh.track_generator