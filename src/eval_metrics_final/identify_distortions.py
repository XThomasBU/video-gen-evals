import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
from dataclasses import dataclass
from typing import Optional

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
# import argparse
from pathlib import Path
from tabulate import tabulate
from phalp.configs.base import FullConfig
from phalp.models.hmar.hmr import HMR2018Predictor
from phalp.trackers.PHALP import PHALP
from phalp.utils import get_pylogger
import seaborn as sns
import torch
import os
import numpy as np
import cv2
import joblib
from scipy.spatial.transform import Rotation as R
import cv2
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import pycocotools.mask as mask_utils
from src.human_mesh.TokenHMR.tokenhmr.lib.datasets.utils import expand_to_aspect_ratio, generate_image_patch_cv2
from src.human_mesh.TokenHMR.tokenhmr.lib.datasets.vitdet_dataset import (
    ViTDetDataset,
)
import yaml
from yacs.config import CfgNode
from src.human_mesh.TokenHMR.tokenhmr.lib.datasets.utils import gen_trans_from_patch_cv
from skimage.morphology import skeletonize, thin

import cv2

warnings.filterwarnings('ignore')
codebook = np.load("/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/codebook.npy")

log = get_pylogger(__name__)

def compute_adjacent_framewise_tokenwise_distances(frame_embeddings):
    """
    frame_embeddings: (num_frames, num_tokens, embedding_dim)
    returns:
        distances: (num_frames - 1, num_tokens)
    """
    diffs = frame_embeddings[1:] - frame_embeddings[:-1]
    dists = np.linalg.norm(diffs, axis=-1)
    return dists

def squiggliness(signal):
    signal = np.asarray(signal)
    if len(signal) < 2:
        return 0.0

    diffs = np.abs(np.diff(signal))
    total_variation = np.sum(diffs)
    return total_variation / len(signal)

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

# ——— CONFIG ———
CODEBOOK_PATH = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/codebook.npy"
PCA_COMPONENTS = 128

REAL_ROOT = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/ucf101/mesh_real_videos"
GEN_VIDEO_PATH = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/ucf101/mesh_cogvideox_videos/HulaHoop/v_HulaHoop_g02_c01"

VELOCITY_THRESHOLD_Z = 2.5
POSE_THRESHOLD_Z = 2.5

# ——— LOAD CODEBOOK ———
print("Loading codebook...")
codebook = np.load(CODEBOOK_PATH).astype(np.float64)
print(f"Codebook shape: {codebook.shape}")

# ——— DATA LOADING ———
def load_video_logits(video_folder):
    frames = sorted(Path(video_folder).glob("tokenhmr_mesh/*.pkl"))
    all_out = []
    for p in tqdm(frames, desc=f"Loading frames in {video_folder}"):
        with open(p, "rb") as f:
            data = pickle.load(f)
        logits = np.array(data.get("cls_logits_softmax", []))
        if logits.ndim > 1:
            logits = logits[0]
        all_out.append(logits)
    return np.array(all_out)

def logits_to_embeddings(logits_seq, codebook):
    return np.einsum('ntc,cd->ntd', logits_seq, codebook)

def flatten_embeddings(embs):
    return embs.reshape(embs.shape[0], -1)

def compute_velocity(traj):
    return np.linalg.norm(np.diff(traj, axis=0), axis=1)

def mahalanobis_batch(X, mean, cov_inv):
    diff = X - mean
    return np.sqrt(np.sum(diff @ cov_inv * diff, axis=1))

def get_decoder_features(embs_seq, decoder):
    # decoder = decoder.to("cuda")
    # embs_seq_tensor = torch.from_numpy(embs_seq).to(torch.float32).to("cuda")
    # embs_seq_tensor = embs_seq_tensor.permute(0, 2, 1)
    # decoder_features = embs_seq_tensor
    # for idx, layer in enumerate(decoder.decoder):
    #     decoder_features = layer(decoder_features)
    #     if idx == 14:
    #         break
    # return decoder_features.detach().cpu().numpy()
    return embs_seq

# ——— MAIN ———
def main():
    gen_class = Path(GEN_VIDEO_PATH).parent.name
    print(f"\n=== ACTION CLASS: {gen_class} ===")

    # --- Load REAL videos
    real_class_dir = Path(REAL_ROOT) / gen_class
    real_embs_list = []

    print("\n=== Loading REAL videos ===")
    for vid in sorted(os.listdir(real_class_dir)):
        vid_path = real_class_dir / vid
        logits_seq = load_video_logits(vid_path)
        if logits_seq.size == 0 or logits_seq.shape[0] < 2:
            continue
        embs_seq = logits_to_embeddings(logits_seq, codebook)
        decoder_features = get_decoder_features(embs_seq, HMAR.model.smpl_head.decpose.tokenizer_proxy.tokenizer.decoder)
        # logits_seq = torch.from_numpy(logits_seq).to(torch.float32).to("cuda")
        # smpl_thetas6D, discrete_token =HMAR.model.smpl_head.decpose.tokenize(logits_seq)
        # print(smpl_thetas6D.shape)
        # print(discrete_token.shape)
        # exit()
        flat_embs = flatten_embeddings(decoder_features)
        real_embs_list.append(flat_embs)

    if len(real_embs_list) < 2:
        print("Not enough real videos!")
        return

    all_real_flat = np.vstack(real_embs_list)
    print(f"Collected real frames: {all_real_flat.shape}")

    # --- PCA fit
    scaler = StandardScaler()
    all_real_scaled = scaler.fit_transform(all_real_flat)
    pca = PCA(n_components=min(PCA_COMPONENTS, all_real_scaled.shape[1], all_real_scaled.shape[0]))
    all_real_pca = pca.fit_transform(all_real_scaled)

    # --- Compute mean and covariance for Mahalanobis
    mean_real = np.mean(all_real_pca, axis=0)
    cov_real = np.cov(all_real_pca.T) + 1e-6 * np.eye(all_real_pca.shape[1])
    cov_inv_real = np.linalg.inv(cov_real)

    # --- Compute real video velocity distribution
    all_real_velocities = []
    for traj in real_embs_list:
        traj_scaled = scaler.transform(traj)
        traj_pca = pca.transform(traj_scaled)
        vel = compute_velocity(traj_pca)
        all_real_velocities.extend(vel)

    all_real_velocities = np.array(all_real_velocities)
    real_vel_mean = np.mean(all_real_velocities)
    real_vel_std = np.std(all_real_velocities)
    print(f"Real velocity mean: {real_vel_mean:.4f}, std: {real_vel_std:.4f}")

    # --- Real Mahalanobis distances for z-score
    real_mahalanobis = mahalanobis_batch(all_real_pca, mean_real, cov_inv_real)
    real_maha_mean = np.mean(real_mahalanobis)
    real_maha_std = np.std(real_mahalanobis)
    print(f"Real Mahalanobis mean: {real_maha_mean:.4f}, std: {real_maha_std:.4f}")

    # --- Load Generated Video
    print("\n=== Loading Generated Video ===")
    gen_logits = load_video_logits(GEN_VIDEO_PATH)
    gen_embs = logits_to_embeddings(gen_logits, codebook)
    gen_embs = get_decoder_features(gen_embs, HMAR.model.smpl_head.decpose.tokenizer_proxy.tokenizer.decoder)
    gen_flat = flatten_embeddings(gen_embs)

    gen_scaled = scaler.transform(gen_flat)
    gen_pca_traj = pca.transform(gen_scaled)

    # --- Compute Generated Video Velocity
    gen_velocity = compute_velocity(gen_pca_traj)
    gen_velocity_z = (gen_velocity - real_vel_mean) / (real_vel_std + 1e-8)
    flagged_velocity_frames = np.where(gen_velocity_z > VELOCITY_THRESHOLD_Z)[0]

    # --- Compute Generated Video Mahalanobis distances
    gen_mahalanobis = mahalanobis_batch(gen_pca_traj, mean_real, cov_inv_real)
    gen_maha_z = (gen_mahalanobis - real_maha_mean) / (real_maha_std + 1e-8)
    flagged_pose_frames = np.where(gen_maha_z > POSE_THRESHOLD_Z)[0]

    # --- Report
    print("\n=== POSE CHANGE & OOD DIAGNOSTICS ===")
    print(f"Total frames: {len(gen_pca_traj)}")
    print(f"Flagged high-change transitions (velocity): {list(flagged_velocity_frames)}")
    print(f"Flagged out-of-distribution pose frames (Mahalanobis): {list(flagged_pose_frames)}")

    # --- 2D Visualization of PCA Space
    print("\n=== Generating 2D Visualization ===")
    pca_2d = PCA(n_components=2)
    all_combined = np.vstack([all_real_scaled, gen_scaled])
    pca_2d.fit(all_combined)

    real_2d = pca_2d.transform(all_real_scaled)
    gen_2d = pca_2d.transform(gen_scaled)

    plt.figure(figsize=(8, 6))
    plt.scatter(real_2d[:, 0], real_2d[:, 1], s=10, color='gray', alpha=0.4, label='Real Frames')
    plt.scatter(gen_2d[:, 0], gen_2d[:, 1], s=20, color='blue', alpha=0.6, label='Generated Frames')
    plt.scatter(gen_2d[flagged_pose_frames, 0], gen_2d[flagged_pose_frames, 1], 
                s=50, color='red', marker='x', label='Flagged OOD Frames')
    plt.title(f"2D Pose Space Projection ({gen_class})")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{gen_class}_2d_pose_scatter.png")
    print(f"Saved 2D scatterplot as {gen_class}_2d_pose_scatter.png")

    # --- Histograms
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(all_real_velocities, bins=50, alpha=0.5, density=True, label='Real Velocities')
    plt.hist(gen_velocity, bins=50, alpha=0.5, density=True, label='Generated Velocities')
    plt.title("Velocity Norm Distribution")
    plt.xlabel("Velocity Norm")
    plt.ylabel("Density")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.hist(real_mahalanobis, bins=50, alpha=0.5, density=True, label='Real Mahalanobis')
    plt.hist(gen_mahalanobis, bins=50, alpha=0.5, density=True, label='Generated Mahalanobis')
    plt.title("Mahalanobis Distance Distribution")
    plt.xlabel("Mahalanobis Distance")
    plt.ylabel("Density")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{gen_class}_diagnostics_histograms.png")
    print(f"Saved histogram as {gen_class}_diagnostics_histograms.png")

    # --- Velocity Plot with Threshold
    plt.figure(figsize=(10, 4))
    plt.plot(gen_velocity, label='Generated Velocity Norm')
    plt.axhline(real_vel_mean, color='green', linestyle='--', label='Real Mean')
    plt.axhline(real_vel_mean + VELOCITY_THRESHOLD_Z * real_vel_std, color='red', linestyle='--', label='Threshold')
    plt.title("Velocity Norm Across Generated Video Frames")
    plt.xlabel("Frame")
    plt.ylabel("Velocity Norm")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{gen_class}_velocity_diagnostics.png")
    print(f"Saved velocity plot as {gen_class}_velocity_diagnostics.png")

if __name__ == "__main__":
    main()