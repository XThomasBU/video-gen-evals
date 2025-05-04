#  Just a base skeleton for now

## src/
- **video_models/**: Contains all the video generative models.
  - Models are located in: `src/video_models/models/`
- **human_mesh/**: Contains code for extracting 3D human pose and mesh information.
- **eval_metrics/**: Contains code for computing evaluation metrics.
- **motion_tracking/**: Contains code for motion quality tracking (physics-IQ, VAMP, TRAJAN (Direct Motion Models for Assessing Generated Videos), etc.).
- **semantic_tracking/**: Contains code for semantic tracking (e.g., semantic context, physical plausibility, action recognition, MLLM-based approaches, etc.).



## 📊 Data

**HACS (Human Action Clips and Segments)**  
- 2-second video clips of human actions  
- Filtered videos downloaded using:
  data/hacs/download_videos.py  
- Video list:
  data/hacs/files/filtered_hacs.json  
- **Usage**: First frame of each clip conditions the video generative models. (And possibly as reference for evaluation.)

---

## 🎥 Video Generative Models

- All open-source models are in:
  src/video_models/models/  
- Each model saves **generated videos and frames** to:
  saved_data/

---

## 🕺 3D Human Pose & Mesh

**TokenHMR**  
- Follow installation instructions in: `src/human_mesh/TokenHMR/README.md`
- Update paths in `src/human_mesh/TokenHMR/data/checkpoints/model_config.yaml`
- Extracts 3D human pose and mesh information  
- Code location:
  src/human_mesh/

---

## 📏 Evaluation Metrics

- Code for computing metrics:
  src/eval_metrics/

---


### Sanity Checks

To check if TokenHMR is working correctly, run the following command:
```bash
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
```

To check if video generative models are working correctly, run the following command:
```bash
python src/video_models/models/cogvideox.py
```