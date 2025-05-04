module load gcc/11.2.0
module load cuda/11.8
module load ffmpeg

python src/human_mesh/TokenHMR/tokenhmr/demo.py \
    --img_folder data/demo \
    --batch_size=1 \
    --full_frame \
    --checkpoint src/human_mesh/TokenHMR/data/checkpoints/tokenhmr_model_latest.ckpt \
    --model_config src/human_mesh/TokenHMR/data/checkpoints/model_config.yaml \
    --side_view \
    --save_mesh \
    --full_frame \
    --out_folder saved_data/demo_results


python src/video_models/models/cogvideox.py