import os
import json
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from collections import defaultdict
import torch.nn.functional as F

from train import  extract_windows
from train2 import (
    TemporalTransformer,
    # load_video_sequence,
    # extract_windows,
    collate_fn,
    ALL_CLASSES,
    WINDOW_SIZE,
    STRIDE,
    LATENT_DIM,
    DEVICE,
)
from tabulate import tabulate


INPUT_DIM= 1370
STRIDE = 0


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
POSE_DIR = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/DWPose/KEYPOINTS/DWPOSE_BODIES_WAN21"
def load_video_sequence(video_folder):
    frames = sorted(Path(video_folder).glob("tokenhmr_mesh/*.pkl"))
    frame_vecs = []

    twod_points_dir = str(video_folder).replace("/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/ucf101/mesh_wan21_videos", POSE_DIR)
    twod_points_paths = sorted(Path(twod_points_dir).glob("*.npy"))

    for idx, p in enumerate(frames):
        # try:
        with open(p, "rb") as f:
            data = pickle.load(f)
        params = data["pred_smpl_params"]

        if isinstance(params, list):
            if len(params) < 1:
                continue
            params = params[0]
        if not isinstance(params, dict):
            continue

        # print(np.array(params["token_out"]).shape, np.array(params["global_orient"]).shape, np.array(params["body_pose"]).shape, np.array(params["betas"]).shape)

        vit_feature   = np.array(params["token_out"])[0].flatten()
        global_orient = np.array(params["global_orient"])[0].flatten()
        body_pose     = np.array(params["body_pose"])[0].flatten()
        betas         = np.array(params["betas"])[0].flatten()

        twod_point_path = twod_points_paths[idx]
        twod_kp = np.load(twod_point_path).flatten()
        twod_kp = twod_kp[:120]  # Ensure 36 keypoints

        # # Normalize each part
        vit_feature   /= np.linalg.norm(vit_feature) + 1e-8
        global_orient /= np.linalg.norm(global_orient) + 1e-8
        body_pose     /= np.linalg.norm(body_pose) + 1e-8
        betas         /= np.linalg.norm(betas) + 1e-8
        twod_kp       /= np.linalg.norm(twod_kp) + 1e-8
        # twod_kp = twod_kp[:36]  # Ensure 120 keypoints

        # if INPUT_DIM == 1250:
        #     vec = np.concatenate([vit_feature, global_orient, body_pose, betas], axis=0)
        #     vec = vec / np.linalg.norm(vec) + 1e-8
        #     if vec.shape[0] != 1250:
        #         continue
        # elif INPUT_DIM == 226:
        #     vec = np.concatenate([global_orient, body_pose, betas], axis=0)
        #     vec = vec / np.linalg.norm(vec) + 1e-8
        #     if vec.shape[0] != 226:
        #         continue
        # elif INPUT_DIM == 1250 + 36:
        vec = np.concatenate([vit_feature, global_orient, body_pose, betas, twod_kp], axis=0)
        # vec = vec / np.linalg.norm(vec) + 1e-8
        # if vec.shape[0] != 1250 + 36:
        #     continue

        frame_vecs.append(torch.tensor(vec, dtype=torch.float32))
        # except:
        #     continue

    if len(frame_vecs) < 2:
        return None

    # [T, 1250]
    frame_tensor = torch.stack(frame_vecs, dim=0)

    # Compute motion vectors (frame-to-frame deltas)
    motion_vecs = frame_tensor[1:] - frame_tensor[:-1]  # [T-1, 1250])
    motion_vecs = torch.cat([torch.zeros(1, INPUT_DIM), motion_vecs], dim=0)  # [T, 1250]


    # Concatenate original + motion
    enriched_tensor = torch.cat([frame_tensor, motion_vecs], dim=1)  # [T, 2500] # 2500 = 1250 * 2
    # print(enriched_tensor.shape)
    # exit()

    # # return enriched_tensor
    # if INPUT_DIM == (1250 + 36) * 2:
    #     return enriched_tensor
    # return frame_tensor
    return enriched_tensor


VID_PATH = {
    'wan21': '/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/wan21_videos_5',
    'runway_gen3_alpha': '/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/runway_gen3_alpha_videos_5',
    'runway_gen4': '/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/runway_gen4_videos_5',
    'hunyuan_360p': '/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/hunyuan_videos_360p_formatted',
    'opensora_256p': '/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/opensora_256p_videos_formatted',
}

def second_order_steady_loss(frame_embs):
    """
    frame_embs: [B, T, D] (no CLS token)
    Returns scalar steady loss for the batch.
    """
    # Compute pairwise differences
    diff1 = frame_embs[:, :-2, :] - frame_embs[:, 1:-1, :]   # [B, T-2, D]
    diff2 = frame_embs[:, 1:-1, :] - frame_embs[:, 2:, :]    # [B, T-2, D]
    # Difference of differences (second order derivative)
    steady = diff1 - diff2                                   # [B, T-2, D]
    loss = (steady ** 2).mean()                              # MSE
    return loss

for MODEL in ["wan21"]:
    GEN_ROOT = f"/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/ucf101/mesh_{MODEL}_videos"
    if "cogvideox" in GEN_ROOT:
        WINDOW_SIZE = 32
        STRIDE = 8
    elif "opensora" in GEN_ROOT:
        WINDOW_SIZE = 32
        STRIDE = 8
    elif "hunyuan" in GEN_ROOT:
        WINDOW_SIZE = 32
        STRIDE = 8
    elif "runway" in GEN_ROOT:
        WINDOW_SIZE = 32
        STRIDE = 8
    elif "wan21" in GEN_ROOT:
        WINDOW_SIZE = 32
        STRIDE = 8
    else:
        WINDOW_SIZE = 32
        STRIDE = 8
    BATCH_SIZE = 64

    # ————— load train embeddings & centroids —————
    all_train_embeds = torch.load(f"SAVE_NEW2/all_train_embeds_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.pt")
    all_train_labels = torch.load(f"SAVE_NEW2/all_train_labels_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.pt")
    all_train_vid_ids = torch.load(f"SAVE_NEW2/all_train_vid_ids_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.pt")

    # with open(f"SAVE_NEW2/centroids_window_{WINDOW_SIZE}_stride_{STRIDE}_valid_window.pkl", "rb") as f:
    #     centroids = pickle.load(f)
    # for k in centroids:
    #     centroids[k] = centroids[k].to(DEVICE)
    # Compute class centroids
    centroids = {}
    for cls in torch.unique(all_train_labels):
        mask_cls = (all_train_labels == cls)
        embeds_cls = all_train_embeds[mask_cls]
        vid_ids_cls = all_train_vid_ids[mask_cls.numpy()]
        
        unique_vids = np.unique(vid_ids_cls)
        video_embeds = []
        for vid in unique_vids:
            vid_mask = (vid_ids_cls == vid)
            video_mean = embeds_cls[vid_mask].mean(dim=0)
            video_embeds.append(video_mean)
        centroid = torch.stack(video_embeds, dim=0).mean(dim=0)
        centroids[int(cls.item())] = centroid

    # ————— load model —————
    print("✅ Loading model...")
    model = TemporalTransformer(input_dim=(INPUT_DIM*2), latent_dim=LATENT_DIM).to(DEVICE)
    print(f"✅ Loading model from temporal_transformer_model_window_{WINDOW_SIZE}_stride_{STRIDE}.pt")
    model.load_state_dict(torch.load(f"/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/src/train/SAVE_NEW2/temporal_transformer_model_window_32_stride_8_valid_window.pt"))
    model.eval()

    # ————— generated dataset & loader —————
    class GeneratedVideoDataset(Dataset):
        def __init__(self, root, classes, window_size=WINDOW_SIZE, stride=STRIDE):
            self.samples, self.labels, self.video_names = [], [], []
            self.window_ids = []
            self.class_to_idx = {c: i for i, c in enumerate(classes)}
            for cls in classes:
                for idx, vid in enumerate(os.listdir(Path(root) / cls)):
                    seq = load_video_sequence(Path(root) / cls / vid)
                    if seq is None:
                        continue
                    wins = extract_windows(seq, window_size, stride)
                    self.samples.extend(wins)
                    self.labels.extend([self.class_to_idx[cls]] * len(wins))
                    self.video_names.extend([vid] * len(wins))
                    self.window_ids.extend([idx] * len(wins))
                    break
            print(f"✅ Loaded {len(self.samples)} windows from generated videos")

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            return self.samples[idx], self.labels[idx], self.video_names[idx], self.window_ids[idx]

    gen_dataset = GeneratedVideoDataset(GEN_ROOT, ALL_CLASSES, window_size=WINDOW_SIZE, stride=STRIDE)
    print(len(gen_dataset))

    with torch.no_grad():
        for idx in range(len(gen_dataset)):
            seqs, labels, vid_names, window_ids = gen_dataset[idx]
            seqs = seqs.to(DEVICE).unsqueeze(0)  # Add batch dimension
            embeddings, frame_embeddings = model(seqs)

            frame_smoothness_loss = second_order_steady_loss(frame_embeddings[:, 1:])
            print("Frame smoothness loss:", frame_smoothness_loss)


