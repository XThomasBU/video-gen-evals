#  TBD

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
- Extracts 3D human pose and mesh information  
- Code location:
  src/human_mesh/

---

## 📏 Evaluation Metrics

- Code for computing metrics:
  src/eval_metrics/

---