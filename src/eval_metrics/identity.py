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

def _pdist_l2(a, b):
    """Compute pair-wise squared l2 distances between points in `a` and `b`.""" 
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a2, b2 = np.square(a).sum(axis=1), np.square(b).sum(axis=1)
    r2 = -2. * np.dot(a, b.T) + a2[:, None] + b2[None, :]
    r2 = np.clip(r2, 0., float(np.inf))

    return r2

def get_uv_embeddings(uv):
    uv = torch.tensor(uv).cuda().float()
    mask = uv[3:, :, :]>0.5
    mask = mask.repeat(4, 1, 1)
    uv[mask] = 0.0
    with torch.no_grad():
        emb    = HMAR.autoencoder_hmar(uv.unsqueeze(0), en=True)
    emb        = emb.view(-1)
    emb        = emb.cpu().numpy()
    return emb

def get_decoder_features(contin_token):
    print(contin_token.shape)
    exit()

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

from skimage.color import rgb2gray
from skimage.feature import graycomatrix, graycoprops
from skimage.util import img_as_ubyte

def compute_glcm_features(uv_map):
    # 1. Convert to grayscale (ignore alpha or mask)
    if uv_map.shape[2] == 4:
        rgb = uv_map[:, :, :3]
    else:
        rgb = uv_map
    gray = rgb2gray(rgb)
    gray = img_as_ubyte(gray)  # ensure uint8 (0–255)

    # 2. Compute GLCM (0°, 45°, 90°, 135°)
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix(gray, distances=distances, angles=angles, symmetric=True, normed=True)

    # 3. Extract texture features
    props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
    feats = [graycoprops(glcm, prop).flatten() for prop in props]

    return np.concatenate(feats)
    
def get_adjacent_framewise_consistency(data_array):
    framewise_diff_velocity = []
    framewise_diff_acceleration = []

    for i in range(1, len(data_array)):
        velocity = np.linalg.norm(data_array[i] - data_array[i-1])
        framewise_diff_velocity.append(velocity)

    for i in range(1, len(framewise_diff_velocity)):
        acceleration = framewise_diff_velocity[i] - framewise_diff_velocity[i-1]
        framewise_diff_acceleration.append(acceleration)

    framewise_diff_velocity = np.array(framewise_diff_velocity)
    framewise_diff_acceleration = np.array(framewise_diff_acceleration)

    velocity_mean = np.mean(framewise_diff_velocity) + 1e-8
    velocity_consistency = np.exp(-np.std(framewise_diff_velocity) / velocity_mean)

    acceleration_mean = np.mean(np.abs(framewise_diff_acceleration)) + 1e-8
    acceleration_consistency = np.exp(-np.std(framewise_diff_acceleration) / acceleration_mean)

    return velocity_consistency, acceleration_consistency, velocity_mean, acceleration_mean

def get_beta_drift(betas):
    ref_beta = betas[0]
    diff_betas = []
    for beta in betas:
        diff_betas.append(np.linalg.norm(beta - ref_beta))
    return np.mean(diff_betas)

def get_camera_drift(camera):
    ref_camera = camera[0]
    diff_camera = []
    for cam in camera:
        diff_camera.append(np.linalg.norm(cam - ref_camera))
    return np.mean(diff_camera)

def get_metrics(data, model_name):
    centroids = []
    betas = []
    appe = []
    uv = []
    loca = []
    pose = []
    camera = []
    joints = []
    pred_pose = []
    pred_loca = []
    pred_uv = []
    pred_appe = []
    frame_tokens = []
    for frame_id in data:


        # try:
        # print(data[frame_id].keys())
        # exit()

        # get centroids
        top_left_x = data[frame_id]['bbox'][0][0]
        top_left_y = data[frame_id]['bbox'][0][1]
        bbox_width = data[frame_id]['bbox'][0][2]
        bbox_height = data[frame_id]['bbox'][0][3]
        c_x = top_left_x + bbox_width / 2
        c_y = top_left_y + bbox_height / 2
        centroids.append([c_x, c_y])

        # get clip image embedding of uv map?

        # joints vector
        joints.append(data[frame_id]['2d_joints'][0])

        # appe vector
        appe.append(data[frame_id]['appe'][0])

        camera.append(data[frame_id]['camera'][0])

        # beta vector
        betas.append(data[frame_id]['prediction_smpl_betas'][0])

        # uv vector
        uv.append(data[frame_id]['uv'][0])

        # loca vector
        loca.append(data[frame_id]['loca'][0])

        # pose vector
        pose.append(data[frame_id]['pose'][0])

        # pred_pose vector
        print(np.array(data[frame_id]['prediction_pose'][0]).shape)
        exit()
        pred_pose.append(data[frame_id]['prediction_pose'][0])

        # pred_loca vector
        pred_loca.append(data[frame_id]['prediction_loca'][0])

        # pred_uv vector
        pred_uv.append(data[frame_id]['prediction_uv'][0])

        # pred_appe vector
        pred_appe.append(get_uv_embeddings(data[frame_id]['prediction_uv'][0]))

        # frame_tokens
        frame_tokens.append(data[frame_id]['smpl'][0]['discrete_token'])
        # except:
        #     print(f"Error processing frame {frame_id}")
        #     continue

    centroids = np.array(centroids)
    appe = np.array(appe)
    betas = np.array(betas)
    uv = np.array(uv)
    loca = np.array(loca)
    pose = np.array(pose)
    camera = np.array(camera)
    pred_pose = np.array(pred_pose)
    pred_loca = np.array(pred_loca)
    pred_uv = np.array(pred_uv)
    pred_appe = np.array(pred_appe)
    frame_tokens = np.array(frame_tokens)
    print(frame_tokens.shape)
    frame_embeddings = codebook[frame_tokens]
    print(frame_embeddings.shape)
    dist = compute_adjacent_framewise_tokenwise_distances(frame_embeddings)
    print(np.mean(dist))
    # exit()
    
    # check deviation of pose and pred_pose
    pred_pose_deviation = np.linalg.norm(pred_pose - pose, axis=1)
    pred_pose_squiggliness = squiggliness(pred_pose_deviation)

    # check deviation of loca and pred_loca
    pred_loca_deviation = np.linalg.norm(pred_loca - loca, axis=1)
    pred_loca_squiggliness = squiggliness(pred_loca_deviation)

    # check deviation of uv and pred_uv
    pred_uv_deviation = np.linalg.norm(pred_uv - uv, axis=1)

    # check deviation of appe and pred_appe
    pred_appe_deviation = np.linalg.norm(pred_appe - appe, axis=1)
    pred_appe_squiggliness = squiggliness(pred_appe_deviation)

    velocity_consistency, acceleration_consistency, velocity_mean, acceleration_mean = get_adjacent_framewise_consistency(centroids)
    velocity_consistency_joints, acceleration_consistency_joints, velocity_mean_joints, acceleration_mean_joints = get_adjacent_framewise_consistency(joints)
    beta_velocity_consistency, beta_acceleration_consistency, beta_velocity_mean, beta_acceleration_mean = get_adjacent_framewise_consistency(betas)
    camera_velocity_consistency, camera_acceleration_consistency, camera_velocity_mean, camera_acceleration_mean = get_adjacent_framewise_consistency(camera)
    beta_drift = get_beta_drift(betas)
    camera_drift = get_camera_drift(camera)
    appe_drift = get_beta_drift(appe)
    apped_velocity_consistency, apped_acceleration_consistency, apped_velocity_mean, apped_acceleration_mean = get_adjacent_framewise_consistency(pred_appe)

    beta_squiggliness = squiggliness(betas)
    appe_squiggliness = squiggliness(appe)
    camera_squiggliness = squiggliness(camera)

    return {
        "velocity_consistency": velocity_consistency,
        "acceleration_consistency": acceleration_consistency,
        "velocity_mean": velocity_mean,
        "acceleration_mean": acceleration_mean,
        "beta_velocity_consistency": beta_velocity_consistency,
        "beta_acceleration_consistency": beta_acceleration_consistency,
        "beta_velocity_mean": beta_velocity_mean,
        "beta_acceleration_mean": beta_acceleration_mean,
        "beta_drift": np.mean(beta_drift),
        "camera_velocity_consistency": camera_velocity_consistency,
        "camera_acceleration_consistency": camera_acceleration_consistency,
        "camera_velocity_mean": camera_velocity_mean,
        "camera_acceleration_mean": camera_acceleration_mean,
        "camera_drift": camera_drift,
        "pred_pose_deviation": np.mean(pred_pose_deviation),
        "pred_pose_squiggliness": pred_pose_squiggliness,
        "pred_loca_squiggliness": pred_loca_squiggliness,
        "pred_loca_deviation": np.mean(pred_loca_deviation),
        "velocity_consistency_joints": velocity_consistency_joints,
        "acceleration_consistency_joints": acceleration_consistency_joints,
        "velocity_mean_joints": velocity_mean_joints,
        "acceleration_mean_joints": acceleration_mean_joints,
        "pred_appe_squiggliness": pred_appe_squiggliness,
        "pred_appe_deviation": np.mean(pred_appe_deviation),
        "appe_drift": np.mean(appe_drift),
        "appe_velocity_consistency": apped_velocity_consistency,
        "appe_acceleration_consistency": apped_acceleration_consistency,
        "appe_velocity_mean": apped_velocity_mean,
        "appe_acceleration_mean": apped_acceleration_mean,
        # "beta_deviation": np.mean(beta_deviation),
        "beta_squiggliness": beta_squiggliness,
        # "appe_deviation": np.mean(appe_deviation),
        "appe_squiggliness": appe_squiggliness,
        "camera_squiggliness": camera_squiggliness,
        "frame_token_dist": np.mean(dist),
    }

if __name__ == "__main__":

    action_folders = [
        # "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/ucf101/tracked_real_videos/SoccerJuggling",
        "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/ucf101/tracked_real_videos/SoccerJuggling",
        # "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/ucf101/tracked_real_videos/HandstandPushups",
        # "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/ucf101/tracked_real_videos/BabyCrawling",
        # "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/ucf101/tracked_real_videos/JumpingJack"

    ]

    

    for action in action_folders:
        real_pkl_paths = []
        gen4_pkl_paths = []
        cog_pkl_paths = []
        gen3_pkl_paths = []
        wan21_pkl_paths = []
        opensora_pkl_paths = []
        hunet_pkl_paths = []
        tables = []
        video_names = []  # Store video names
        all_metrics_data = []  # Store all metrics data for min/max analysis
        
        for video in sorted(os.listdir(action)):
            full_path = Path(action) / video
            print(f"Processing {full_path}")
            video_names.append(video)  # Store video name
            track_info_pkl_path = os.path.join(full_path, "results/demo_video.pkl")
            real_pkl_paths.append(track_info_pkl_path)

            track_path = str(full_path).replace("tracked_real_videos", "tracked_runway_gen4_videos")
            track_info_pkl_path = os.path.join(track_path, "results/demo_video.pkl")
            gen4_pkl_paths.append(track_info_pkl_path)

            track_path = str(full_path).replace("tracked_real_videos", "tracked_cogvideox_videos")
            track_info_pkl_path = os.path.join(track_path, "results/demo_video.pkl")
            cog_pkl_paths.append(track_info_pkl_path)

            track_path = str(full_path).replace("tracked_real_videos", "tracked_runway_gen3_alpha_videos")
            track_info_pkl_path = os.path.join(track_path, "results/demo_video.pkl")
            gen3_pkl_paths.append(track_info_pkl_path)

            track_path = str(full_path).replace("tracked_real_videos", "tracked_wan21_videos")
            track_info_pkl_path = os.path.join(track_path, "results/demo_video.pkl")
            wan21_pkl_paths.append(track_info_pkl_path)

            track_path = str(full_path).replace("tracked_real_videos", "tracked_opensora_256p_videos")
            track_info_pkl_path = os.path.join(track_path, "results/demo_video.pkl")
            opensora_pkl_paths.append(track_info_pkl_path)

            track_path = str(full_path).replace("tracked_real_videos", "tracked_hunyuan_360p_videos")
            track_info_pkl_path = os.path.join(track_path, "results/demo_video.pkl")
            hunet_pkl_paths.append(track_info_pkl_path)

        for video_idx, (real_pkl_path, gen4_pkl_path, cog_pkl_path, gen3_pkl_path, wan21_pkl_path, opensora_pkl_path, hunet_pkl_path) in enumerate(zip(real_pkl_paths, gen4_pkl_paths, cog_pkl_paths, gen3_pkl_paths, wan21_pkl_paths, opensora_pkl_paths, hunet_pkl_paths)):
            data_gen3 = joblib.load(gen3_pkl_path)

            # print(real_pkl_path, gen4_pkl_path, cog_pkl_path)
            # exit()
            # Load tracked data
            data_cog  = joblib.load(cog_pkl_path)
            metrics_cog = get_metrics(data_cog, "CogVideox")
            data_gen4 = joblib.load(gen4_pkl_path)
            data_real = joblib.load(real_pkl_path)
            data_wan21 = joblib.load(wan21_pkl_path)
            data_opensora = joblib.load(opensora_pkl_path)
            data_hunet = joblib.load(hunet_pkl_path)
            metrics_gen4 = get_metrics(data_gen4, "Runway Gen4")
            metrics_real = get_metrics(data_real, "Real")
            metrics_gen3 = get_metrics(data_gen3, "Runway Gen3")
            metrics_wan21 = get_metrics(data_wan21, "Wan21")
            metrics_opensora = get_metrics(data_opensora, "OpenSora")
            metrics_hunet = get_metrics(data_hunet, "Hunyuan")
            # metrics_to_print = [
            #     "velocity_consistency",
            #     "acceleration_consistency",
            #     "velocity_mean",
            #     "acceleration_mean",
            #     "beta_drift",
            #     "camera_drift",
            #     "pred_pose_squiggliness",
            #     "pred_pose_deviation",
            #     "pred_loca_squiggliness",
            #     "pred_loca_deviation",
            #     "pred_appe_deviation",
            #     "velocity_consistency_joints",
            #     "acceleration_consistency_joints",
            #     "velocity_mean_joints",
            #     "acceleration_mean_joints",
            #     "pred_appe_squiggliness",
            #     "pred_appe_deviation",
            #     "appe_drift"
            # ]
            metrics_to_print = [
                # "velocity_consistency",
                "velocity_mean",
                # "acceleration_consistency",
                "acceleration_mean",
                "pred_pose_squiggliness",
                # "pred_pose_deviation",
                "pred_loca_squiggliness",
                # "pred_loca_deviation",
                # "pred_appe_deviation",
                "pred_appe_squiggliness",
                # "velocity_consistency_joints",
                # "acceleration_consistency_joints",
                "velocity_mean_joints",
                "acceleration_mean_joints",
                "beta_drift",
                # "beta_velocity_consistency",
                # "beta_acceleration_consistency",
                # "beta_velocity_mean",
                # "beta_acceleration_mean",
                "camera_drift",
                # "camera_velocity_consistency",
                # "camera_acceleration_consistency",
                # "camera_velocity_mean",
                # "camera_acceleration_mean",
                "appe_drift",
                # "appe_velocity_consistency",
                # "appe_acceleration_consistency",
                # "appe_velocity_mean",
                # "appe_acceleration_mean",
                # "beta_deviation",
                "beta_squiggliness",
                # "appe_deviation",
                "appe_squiggliness",
                "camera_squiggliness",
                "frame_token_dist",
            ]

            # Store metrics data for min/max analysis
            video_metrics_data = {
                'video_name': video_names[video_idx],
                'Real': metrics_real,
                'CogVideox': metrics_cog,
                'Runway Gen4': metrics_gen4,
                'Runway Gen3': metrics_gen3,
                'Wan21': metrics_wan21,
                'OpenSora': metrics_opensora,
                'Hunyuan': metrics_hunet
            }
            all_metrics_data.append(video_metrics_data)

            table = [
                [metric, 
                    f"{metrics_real[metric]:.4f}",
                    f"{metrics_cog[metric]:.4f}", 
                    f"{metrics_gen4[metric]:.4f}", 
                    f"{metrics_gen3[metric]:.4f}", 
                    f"{metrics_wan21[metric]:.4f}",
                    f"{metrics_opensora[metric]:.4f}",
                    f"{metrics_hunet[metric]:.4f}"] 
                for metric in metrics_to_print
            ]
            tables.append(table)

            headers = ["Metric", "Real", "CogVideox", "Runway Gen4", "Runway Gen3", "Wan21", "OpenSora", "Hunyuan"]
            # print(tabulate(table, headers=headers, tablefmt="grid"))
            # print()
            # exit()

        # === Aggregate metrics over all videos ===
        avg_table = []
        std_dev_table = []
        num_videos = len(tables)

        for metric_idx in range(len(metrics_to_print)):
            metric_name = metrics_to_print[metric_idx]
            values = np.array([
                [float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6]), float(row[7])]
                for table in tables for row in [table[metric_idx]]
            ])
            values = values.reshape(num_videos, 7)  # [num_videos, num_models]
            mean_vals = np.mean(values, axis=0)
            std_dev_vals = np.std(values, axis=0)
            avg_table.append([metric_name] + [f"{v:.4f}" for v in mean_vals])
            std_dev_table.append([metric_name] + [f"{v:.4f}" for v in std_dev_vals])

        print(f"===> Average Metrics Across All Videos for Action: {action}")
        print(tabulate(avg_table, headers=headers, tablefmt="grid"))
        print(tabulate(std_dev_table, headers=headers, tablefmt="grid"))

        # === Find min/max/median values for each metric across all models ===
        print("\n=== Min/Max/Median Values for Each Metric (All Models) ===")
        model_names = ["Real", "CogVideox", "Runway Gen4", "Runway Gen3", "Wan21", "OpenSora", "Hunyuan"]
        for metric in metrics_to_print:
            print(f"\n--- {metric} ---")
            # Collect all values for this metric
            all_values = []
            for video_data in all_metrics_data:
                video_name = video_data['video_name']
                for model_name in model_names[1:]:
                    value = video_data[model_name][metric]
                    all_values.append((value, video_name, model_name))
            # Find min, max and median
            min_val, min_video, min_model = min(all_values, key=lambda x: x[0])
            max_val, max_video, max_model = max(all_values, key=lambda x: x[0])
            median_val, median_video, median_model = sorted(all_values, key=lambda x: x[0])[len(all_values) // 2]
            print(f"MIN: {min_val:.4f} (Video: {min_video}, Model: {min_model})")
            print(f"MAX: {max_val:.4f} (Video: {max_video}, Model: {max_model})")
            print(f"MEDIAN: {median_val:.4f} (Video: {median_video}, Model: {median_model})")

        # === Find min/max/median values for each metric for each model separately ===
        print("\n=== Min/Max/Median Values for Each Metric for Each Model ===")
        for model_name in model_names[1:]:
            print(f"\n--- {model_name} ---")
            for metric in metrics_to_print:
                # Collect all values for this metric and model
                values = []
                for video_data in all_metrics_data:
                    value = video_data[model_name][metric]
                    video_name = video_data['video_name']
                    values.append((value, video_name))
                # Find min, max, and median
                min_val, min_video = min(values, key=lambda x: x[0])
                max_val, max_video = max(values, key=lambda x: x[0])
                median_val, median_video = sorted(values, key=lambda x: x[0])[len(values) // 2]
                print(f"{metric}: MIN: {min_val:.4f} (Video: {min_video}) | MAX: {max_val:.4f} (Video: {max_video}) | MEDIAN: {median_val:.4f} (Video: {median_video})")

        # === Best Model Ranking by Proximity to Real ===
        # === Best & Worst Model Ranking by Proximity to Real ===
        model_scores = {name: 0 for name in headers[1:]}  # Includes 'Real', but we skip it in logic
        best_per_metric = []

        for metric in avg_table:
            metric_name = metric[0]
            real_val = float(metric[1])
            competitor_names = headers[2:]  # Skip 'Metric' and 'Real'
            competitor_vals = np.array([float(v) for v in metric[2:]])
            diffs = np.abs(competitor_vals - real_val)

            # Best model (min diff)
            best_idx = np.argmin(diffs)
            best_model = competitor_names[best_idx]
            best_val = competitor_vals[best_idx]
            model_scores[best_model] += 1

            # Worst model (max diff)
            worst_idx = np.argmax(diffs)
            worst_model = competitor_names[worst_idx]
            worst_val = competitor_vals[worst_idx]

            best_per_metric.append((metric_name, best_model, best_val, worst_model, worst_val, real_val))

        # Ranking summary (exclude Real)
        ranking = sorted([(model, score) for model, score in model_scores.items() if model != "Real"],
                        key=lambda x: -x[1])
        ranking_table = [[model, score] for model, score in ranking]

        print("\n=== Model Ranking by Proximity to Real")
        print(tabulate(ranking_table, headers=["Model", "Score (closest to Real)"], tablefmt="grid"))

        print("\n=== Closest & Farthest Model to Real for Each Metric ===")
        print(tabulate(best_per_metric,
                    headers=["Metric", "Best Model", "Best Value", "Worst Model", "Worst Value", "Real Value"],
                    tablefmt="grid", floatfmt=".4f"))
        
        # === Prepare data matrix for plotting ===
        metric_names = [row[0] for row in avg_table]
        model_names = headers[1:]  # ['Real', 'CogVideox', ..., 'Hunyuan']
        values = np.array([[float(v) for v in row[1:]] for row in avg_table])  # [num_metrics, num_models]
        num_metrics = len(metric_names)
        num_models = len(model_names)

    # Prepare data
    metric_names = [row[0] for row in avg_table]
    model_names = headers[1:]  # Exclude "Metric"
    values = np.array([[float(v) for v in row[1:]] for row in avg_table])

    num_metrics = len(metric_names)
    num_models = len(model_names)

    # Grid config
    plots_per_row = 3
    nrows = (num_metrics + plots_per_row - 1) // plots_per_row
    fig_height = nrows * 4
    fig_width = plots_per_row * 6

    fig, axes = plt.subplots(nrows=nrows, ncols=plots_per_row, figsize=(fig_width, fig_height))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i >= num_metrics:
            ax.axis('off')
            continue

        vals = values[i]
        real_val = vals[0]
        other_vals = vals[1:]
        other_models = model_names[1:]

        # Sort others by value
        sorted_indices = np.argsort(other_vals)
        sorted_vals = [real_val] + [other_vals[j] for j in sorted_indices]
        sorted_labels = ["Real"] + [other_models[j] for j in sorted_indices]

        # Color map (higher = darker)
        cmap = plt.cm.Blues
        norm = plt.Normalize(vmin=np.min(sorted_vals), vmax=np.max(sorted_vals))
        colors = [cmap(norm(v)) for v in sorted_vals]

        bars = ax.barh(range(len(sorted_vals)), sorted_vals, color=colors)
        ax.set_yticks(range(len(sorted_labels)))
        ax.set_yticklabels(sorted_labels, fontsize=14)
        ax.invert_yaxis()
        ax.set_title(metric_names[i], fontsize=16)

        # Add value text and extend x-axis limit to avoid overflow
        max_val = max(sorted_vals)
        xlim_extension = max_val * 0.25  # Extend x-axis by 25%
        ax.set_xlim(left=0, right=max_val + xlim_extension)

        for idx, v in enumerate(sorted_vals):
            ax.text(v + max_val * 0.02, idx, f"{v:.3f}", va='center', ha='left', fontsize=13)

    plt.tight_layout(h_pad=3.0, w_pad=3.0)
    plt.savefig("Pushups_readable_adjusted.png", dpi=300)