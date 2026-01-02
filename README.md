# Generative Action Tell-Tales: Assessing Human Motion in Synthesized Videos

**Authors**: Xavier Thomas¹, Youngsun Lim¹, Ananya Srinivasan²*, Audrey Zheng³*, Deepti Ghadiyaram¹⁴

¹Boston University | ²Belmont High School | ³Canyon Crest Academy | ⁴Runway  
*Equal contribution

> **Note**: This code is under construction.

PyTorch implementation for evaluating human motion quality in synthesized videos.

## Quick Start

### 1. Setup Dependencies

```bash
# Clone TokenHMR
git clone https://github.com/saidwivedi/TokenHMR.git
cd TokenHMR
# Download TokenHMR models and requirements per their instructions

# Clone DWpose
git clone https://github.com/IDEA-Research/DWPose.git
# Download DWpose models and requirements per their instructions
```

**Required modifications** (apply after cloning):

1. **TokenHMR**: 
    - Add `modifications/mesh_generator.py` to `TokenHMR/tokenhmr/mesh_generator.py`
    - Update `TokenHMR/tokenhmr/lib/models/heads/token_head.py` - modify `SMPLTokenDecoderHead` class as shown in modifications/token_head.py`

2. **DWpose**: 
   - Update `DWPose/ControlNet-v1-1-nightly/annotator/process_video.py` - add required code as shown in `modifications/process_video.py`
   - Update `DWPose/ControlNet-v1-1-nightly/annotator/dwpose/__init__.py` - add required code as shown in `modifications/dwpose_init.py`

### 2. Extract Meshes

Use TokenHMR to extract meshes from videos:

Update in `extract_mesh.py`:
```python
DIR = "PATH_TO_VIDEOS"
```

```bash
python extract_mesh.py
```


### 3. Extract Keypoints

Use DWpose to extract keypoints from videos.

Update in `DWPose-onnx/ControlNet-v1-1-nightly/annotator/process_video.py`:
```python
VID_DIR = "PATH_TO_VIDEOS"
POSE_SAVE_PATH = "PATH_TO_SAVE_KEYPOINTS"
```

```bash
cd DWPose-onnx/ControlNet-v1-1-nightly/annotator
python process_video.py
```

### 4. Configure Paths

Edit `GLOBAL_CONFIG` in `train.py`:
```python
"paths": {
    "mesh_human_data_real": "/path/to/real_meshes",
    "mesh_human_data_generated": "/path/to/generated_meshes",
    "real_kp_dir": "/path/to/real_keypoints",
    "gen_kp_dir": "/path/to/generated_keypoints",
}
```

## Training

```bash
python train.py
```

## Citation
If you find this code useful, please cite our paper:

```
@article{thomas2025generative,
  title={Generative Action Tell-Tales: Assessing Human Motion in Synthesized Videos},
  author={Thomas, Xavier and Lim, Youngsun and Srinivasan, Ananya and Zheng, Audrey and Ghadiyaram, Deepti},
  journal={arXiv preprint arXiv:2512.01803},
  year={2025}
}
