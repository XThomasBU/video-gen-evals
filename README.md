# Base Video Evaluation Framework

## 📁 src/

- **video_models/**  
  Contains all the video generative models.  
  Models are located in: `src/video_models/models/`

- **human_mesh/**  
  Contains code for extracting 3D human pose and mesh information.

- **eval_metrics/**  
  Contains code for computing evaluation metrics.

- **motion_tracking/**  
  Motion quality tracking (physics-IQ, VAMP, TRAJAN — *Direct Motion Models for Assessing Generated Videos*, etc.).

- **semantic_tracking/**  
  Semantic tracking (e.g., semantic context, physical plausibility, action recognition, MLLM-based approaches, etc.).

---

## Installation

```bash
conda create -n video_eval python=3.10
conda activate video_eval
pip install -r requirements.txt
```

---

## Download HACS Video Clips

List videos in `data/hacs/files/filtered_hacs.json`.

```bash
python data/hacs/download_videos.py
```

---

## Sanity Checks

**Video generation test**
```bash
python src/video_models/models/cogvideox.py
```

Add Runway API to a .env file as `RUNWAYML_API_SECRET=XXXXXXXX`.
```bash
python src/video_models/models/runway.py
```


**TokenHMR test**
```bash
python src/human_mesh/TokenHMR/tokenhmr/demo.py \
    --img_folder data/demo \
    --batch_size=1 \
    --full_frame \
    --checkpoint src/human_mesh/TokenHMR/data/checkpoints/tokenhmr_model_latest.ckpt \
    --model_config src/human_mesh/TokenHMR/data/checkpoints/model_config.yaml \
    --side_view \
    --save_mesh \
    --out_folder saved_data/demo_results
```

---

## Outputs

Generated videos and frames are saved to:
```
saved_data/
```

---

## Evalation metrics
- TODO: Add evaluation metrics for video generation.
- Should be as simple as evaluator.evaluate(generated_video, reference_video, config)