# Base Video Evaluation Framework


Code atm only works for single person videos, in generated videos if it all there are multiple people, only the first person information is used.

## Project Structure

The project is organized into the following directories:

### `src/`

* **`video_models/`**: Contains all the video generative models. Models are located in: `src/video_models/models/`
  * `gen_ucf101_videos.py`: Script for generating UCF101 generated videos.
* **`human_mesh/`**: Contains code for extracting 3D human pose and mesh information.
* **`eval_metrics/`**: Contains code for computing evaluation metrics.
* **`motion_tracking/`**: Motion quality tracking (physics-IQ, VAMP, TRAJAN — *Direct Motion Models for Assessing Generated Videos*, etc.).
* **`semantic_tracking/`**: Semantic tracking (e.g., semantic context, physical plausibility, action recognition, MLLM-based approaches, etc.).

---

## Installation

To set up the project, follow these steps:

```bash
conda create -n video_eval python=3.10
conda activate video_eval
pip install -r requirements.txt
```

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

## Generate UCF101 Video Generations

`python src/video_models/gen_ucf101_videos.py`

* Output Location:  
  Generated videos are saved to `saved_data/videos/model_name/ucf_id/video.mp4`

* Text Conditioning Prompt:  
  If available, the prompt format is:  
  `A person doing [action]` → where `action` comes from the UCF101 label of the clip.

---


## Evaluation Metrics

- TODO: Consolidate evaluation metrics for video generation.
- Should be as simple as `evaluator.evaluate(generated_video, reference_video, config)`