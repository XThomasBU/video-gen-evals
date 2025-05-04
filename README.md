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

**HACS (Human Action Clips and Segments)**

- 2-second video clips of human actions.
- Filtered videos downloaded using: `data/hacs/download_videos.py`.
- Update list: `data/hacs/files/filtered_hacs.json`.
- **Usage**: The first frame of each clip conditions the video generative models (and can also serve as reference for evaluation).

```bash
python data/hacs/download_videos.py
```

Downloaded videos are saved to `saved_data/hacs/{youtube_id}`.
- **Note**: The first frame of each clip is saved as `saved_data/hacs/{youtube_id}/selected_frames/00000.jpg`.
- **Note**: The first frame of each clip is used as the conditioning frame for video generation.
- **Note**: `saved_data/hacs/{youtube_id}/metadata.json` contains the metadata for each clip (fps, action --> which can be used for the prompt, etc.).

---

## Sanity Checks

**Video generation test**
```bash
python -m src.video_models.models.cogvideox
```

Add Runway API to a .env file as `RUNWAYML_API_SECRET=XXXXXXXX`.
```bash
python -m src.video_models.models.runway
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

## Generate HACS Video Generations

`python src/video_models/gen_hacs_videos.py`

- Output Location:  
  Generated videos are saved to `saved_data/hacs/{youtube_id}/generated_videos/`

- Text Conditioning Prompt:  
  If available, the prompt format is:  
  `A person doing [action]` → where `action` comes from the HACS label of the clip.

---

### Example Directory Structure (saved_data/hacs/x_gEdkM6kwE/): Inside the directory of each video clip:

`full_frames/` → Contains the full frames of the video clip (from the downloaded video).  
`selected_frames/` → Selected frames based on the HACS CSV.  
`selected_frames/frame_00001.png` → First selected frame.  
`metadata.json` → Contains `fps`, `action` (can be used for the prompt), etc.  
`x_gEdkM6kwE.mp4` → Full downloaded video clip.  
`x_gEdkM6kwE_selected.mp4` → Video made from selected frames.

`generated_videos_{model_name}/` → Generated video outputs for the given model.  
  `frames/` → Generated frames.  
  `video.mp4` → Generated video.  
  `metadata.json` → Metadata for the generated video.

Example:  
If `model_name = cogvideox`, the directory will be `generated_videos_cogvideox/`.

## Evalation metrics
- TODO: Add evaluation metrics for video generation.
- Should be as simple as `evaluator.evaluate(generated_video, reference_video, config)`