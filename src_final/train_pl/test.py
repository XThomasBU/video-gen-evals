import os
import sys
import json
import math
import random
import typing as T
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# --------------------- local project imports ---------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models import TemporalTransformerV2Plus
from losses import *
from utils import *


cogvideo_ds = NpzVideoDataset("/home/coder/projects/video_evals/video-gen-evals/src_final/meshes_npz_cogvideox")
print(f"cogvideo_ds: {len(cogvideo_ds)} samples")

gen4_ds = NpzVideoDataset("/home/coder/projects/video_evals/video-gen-evals/src_final/meshes_npz_gen4")
print(f"gen4_ds: {len(gen4_ds)} samples")

real_ds = NpzVideoDataset("/home/coder/projects/video_evals/video-gen-evals/src_final/meshes_npz")
print(f"real_ds: {len(real_ds)} samples")

# creat a subset of real_ds with only actions that are in cogvideo_ds and gen4_ds
common_actions = set(cogvideo_ds.actions) & set(gen4_ds.actions)
common_actions = common_actions & set(real_ds.actions)
print(f"common_actions: {len(common_actions)} actions")
