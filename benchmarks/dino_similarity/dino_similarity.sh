#!/bin/bash -l

# Set SCC project
#$ -P ivc-ml
#$ -t 1-1  # Array job specification
#$ -pe omp 4 # Request 4 CPU cores
#$ -l gpus=1 # Request 1 GPU
#$ -l gpu_c=8.0
#$ -l h_rt=21:00:00
#$ -N youtube_dino_sim
#$ -j y # Merge standard output and error
#$ -o /projectnb/ivc-ml/audreyzh/GEN_VIDEO_EVAL/video-gen-evals/qsub_runs

module load miniconda
conda activate video_eval

python /projectnb/ivc-ml/audreyzh/GEN_VIDEO_EVAL/video-gen-evals/benchmarks/dino_similarity/dino_similarity.py --video_dir /projectnb/ivc-ml/xthomas/SHARED/video_evals/YOUTUBE_DATA --frames_dir /projectnb/ivc-ml/audreyzh/GEN_VIDEO_EVAL/video-gen-evals/saved_data/YOUTUBE_DATA --output_dir /projectnb/ivc-ml/audreyzh/GEN_VIDEO_EVAL/video-gen-evals/benchmarks/dino_similarity/youtube_dino_similarities.json