import warnings
from dataclasses import dataclass
from typing import Optional

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
# import argparse

from phalp.configs.base import FullConfig
from phalp.models.hmar.hmr import HMR2018Predictor
from phalp.trackers.PHALP import PHALP
from phalp.utils import get_pylogger

import torch
import os
import numpy as np
import cv2
import joblib
from scipy.spatial.transform import Rotation as R
import cv2
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

log = get_pylogger(__name__)

class TokenHMRPredictor(HMR2018Predictor):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        # Setup our new model
        from src.human_mesh.TokenHMR.tokenhmr.lib.models import load_tokenhmr

        # Load checkpoints
        model, _ = load_tokenhmr(checkpoint_path=cfg.checkpoint, \
                                 model_cfg=cfg.model_config, \
                                 is_train_state=False, is_demo=True)

        self.model = model
        self.model.eval()

    def forward(self, x):
        hmar_out = self.hmar_old(x)
        batch = {
            'img': x[:,:3,:,:],
            'mask': (x[:,3,:,:]).clip(0,1),
        }
        model_out = self.model(batch)

        # Overriding the SMPL params with the TokenHMR params
        out = hmar_out | {
            'pose_smpl': model_out['pred_smpl_params'],
            'pred_cam': model_out['pred_cam'],
        }
        return out

class PHALP_Prime_TokenHMR(PHALP):
    def __init__(self, cfg):
        super().__init__(cfg)

    def setup_hmr(self):
        self.HMAR = TokenHMRPredictor(self.cfg)

@dataclass
class Human4DConfig(FullConfig):
    checkpoint: Optional[str] = None
    model_config: Optional[str] = None
    output_dir: Optional[str] = None

cs = ConfigStore.instance()
cs.store(name="config", node=Human4DConfig)

# Initialize Hydra
hydra.core.global_hydra.GlobalHydra.instance().clear()
hydra.initialize(config_path="../../src/human_mesh/TokenHMR/tokenhmr/lib/configs_hydra")

checkpoint = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/src/human_mesh/TokenHMR/data/checkpoints/tokenhmr_model_latest.ckpt"
model_config = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/src/human_mesh/TokenHMR/data/checkpoints/model_config.yaml"

cfg: DictConfig = hydra.compose(
    config_name="config"
)
cfg.checkpoint = checkpoint
cfg.model_config = model_config

HMAR = TokenHMRPredictor(cfg)
HMAR = HMAR.to("cuda")

def _pdist_l2(a, b):
    """Compute pair-wise squared l2 distances between points in `a` and `b`.""" 
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a2, b2 = np.square(a).sum(axis=1), np.square(b).sum(axis=1)
    r2 = -2. * np.dot(a, b.T) + a2[:, None] + b2[None, :]
    r2 = np.clip(r2, 0., float(np.inf))

    return r2


def get_uv_distance(t_uv, d_uv):
    t_uv         = torch.tensor(t_uv).cuda().float()
    d_uv         = torch.tensor(d_uv).cuda().float()

    d_mask       = d_uv[3:, :, :]>0.5
    t_mask       = t_uv[3:, :, :]>0.5
    
    mask_dt      = torch.logical_and(d_mask, t_mask)
    mask_dt      = mask_dt.repeat(4, 1, 1)
    mask_        = torch.logical_not(mask_dt)

    # print("Non-zero mask count:", mask_dt.sum().item())
    
    t_uv[mask_]  = 0.0
    d_uv[mask_]  = 0.0

    with torch.no_grad():
        t_emb    = HMAR.autoencoder_hmar(t_uv.unsqueeze(0), en=True)
        d_emb    = HMAR.autoencoder_hmar(d_uv.unsqueeze(0), en=True)
    t_emb        = t_emb.view(-1)
    d_emb        = d_emb.view(-1)

    # convert to np array
    t_emb = t_emb.cpu().numpy()
    d_emb = d_emb.cpu().numpy()

    return np.sum((t_emb-d_emb)**2)



def plot_uv_pose_residuals(data, alpha=1.0, scale=1e3, key_uv="uv", key_pose="pose", save_path=None, title=None):
    frame_list = sorted(data.keys())
    residuals = []
    frame_indices = []

    for i in range(1, len(frame_list)):
        f1, f2 = frame_list[i - 1], frame_list[i]
        if key_uv not in data[f1] or key_uv not in data[f2] or key_pose not in data[f1] or key_pose not in data[f2]:
            continue
        if not data[f1][key_uv] or not data[f2][key_uv] or not data[f1][key_pose] or not data[f2][key_pose]:
            continue

        uv1 = np.array(data[f1][key_uv])[0]
        uv2 = np.array(data[f2][key_uv])[0]
        pose1 = np.array(data[f1][key_pose])[0]
        pose2 = np.array(data[f2][key_pose])[0]

        d_uv = get_uv_distance(uv1, uv2) / scale
        d_pose = cosine(pose1, pose2)  # or use np.linalg.norm(pose1 - pose2)

        residual = d_uv - alpha * d_pose
        residuals.append(residual)
        frame_indices.append(i)

    plt.figure(figsize=(10, 4))
    plt.plot(frame_indices, residuals)
    plt.title(f"UV-Pose Residual Distortion Signal: {title}")
    plt.xlabel("Frame Index")
    plt.ylabel("Residual")
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight') 



def plot_three_uv_pose_residuals(datasets, titles, save_path, alpha=1.0, scale=1e3, key_uv="uv", key_pose="pose"):
    """
    datasets: list of tracking data dicts
    titles:   list of titles for each dataset
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 4), sharey=True)
    for ax, data, title in zip(axes, datasets, titles):
        frame_list = sorted(data.keys())
        residuals, frame_indices = [], []
        for i in range(1, len(frame_list)):
            f1, f2 = frame_list[i - 1], frame_list[i]
            if key_uv not in data[f1] or key_uv not in data[f2] or key_pose not in data[f1] or key_pose not in data[f2]:
                continue

            if len(data[f1][key_uv]) == 0 or len(data[f2][key_uv]) == 0 or len(data[f1][key_pose]) == 0 or len(data[f2][key_pose]) == 0:
                print("No data for frame", f1, f2)
                continue

            uv1 = np.array(data[f1][key_uv])[0]
            uv2 = np.array(data[f2][key_uv])[0]
            pose1 = np.array(data[f1][key_pose])[0]
            pose2 = np.array(data[f2][key_pose])[0]
            d_uv = get_uv_distance(uv1, uv2) / scale
            d_pose = cosine(pose1, pose2)
            residual = d_uv - alpha * d_pose
            residuals.append(residual)
            frame_indices.append(i)
        mean_res = np.mean(residuals)
        std_res  = np.std(residuals)
        ax.plot(frame_indices, residuals, linewidth=1)
        ax.set_title(f"{title}\nμ={mean_res:.4f}, σ={std_res:.4f}")
        ax.set_xlabel("Frame Index")
        ax.grid(True)
    axes[0].set_ylabel("UV–Pose Residual")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')

if __name__ == "__main__":

    # Paths to tracking results
    cog_pkl  = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/saved_data/ucf101/v_HulaHoop_g20_c07/cogvideox_tracking_output/results/demo_video.pkl"
    gen4_pkl = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/saved_data/ucf101/v_HulaHoop_g20_c07/runway_gen4_turbo_tracking_output/results/demo_video.pkl"
    real_pkl = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/saved_data/ucf101/v_HulaHoop_g20_c07/real_video_tracking_output/results/demo_v_HulaHoop_g20_c07_full.pkl"

    # Load tracked data
    data_cog  = joblib.load(cog_pkl)
    data_gen4 = joblib.load(gen4_pkl)
    data_real = joblib.load(real_pkl)


    plot_uv_pose_residuals(data_cog, title="CogVideo", key_uv="uv", key_pose="pose", save_path="uv_cog_pose_residuals.png")
    plot_uv_pose_residuals(data_gen4, title="Gen4", key_uv="uv", key_pose="pose", save_path="uv_gen4_pose_residuals.png")
    plot_uv_pose_residuals(data_real, title="Real", key_uv="uv", key_pose="pose", save_path="uv_real_pose_residuals.png")

    plot_three_uv_pose_residuals(datasets=[data_cog, data_gen4, data_real], titles=["CogVideo", "Gen4", "Real"], save_path="uv_pose_residuals_three.png")

    exit()
