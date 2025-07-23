#!/bin/bash -l

# Set SCC project
#$ -P ivc-ml
#$ -t 1-1  # Array job specification
#$ -pe omp 4 # Request 4 CPU cores
#$ -l gpus=1 # Request 1 GPU
#$ -l gpu_c=8.0
#$ -l h_rt=21:00:00
#$ -N ucf101_all
#$ -j y # Merge standard output and error
#$ -o /projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/qsub_runs

module load miniconda
conda activate video_evals

module load ffmpeg

python get_frames.py --video_dir /projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/saved_data/ucf101 --output_dir /projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/saved_data/ucf101_all_classes